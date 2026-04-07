package com.edgetutor.ingestion

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * Produces 384-dim L2-normalised sentence embeddings using
 * Snowflake/snowflake-arctic-embed-xs (ONNX backend).
 *
 * Arctic Embed requires a query prefix at retrieval time but NOT at ingestion time.
 * Pass [isQuery]=true when embedding a user question; leave false for document chunks.
 *
 * Holds an OrtSession; call [close] when the owning ViewModel is cleared.
 * All methods are thread-safe as long as [close] is not called concurrently with [embed].
 */
class Embedder(context: Context) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val tokenizer: WordPieceTokenizer

    /** Embedding dimensionality — 384 for snowflake-arctic-embed-xs. */
    val dim = 384

    companion object {
        private const val QUERY_PREFIX =
            "Represent this sentence for searching relevant passages: "
        private const val ASSET_MODEL_SIZE = 90395016L  // arctic.onnx size in bytes
    }

    init {
        // Copy model to filesDir so ORT can load from a file path.
        // Loading from path avoids holding the full 91 MB as a heap byte array.
        val modelFile = java.io.File(context.filesDir, "arctic.onnx")
        
        // Clean up orphaned external data files from previous export attempts
        context.filesDir.listFiles { _, name -> name.endsWith(".onnx.data") }
            ?.forEach { it.delete() }
        
        // Clean up old embedding model if present
        val oldModel = java.io.File(context.filesDir, "minilm.onnx")
        if (oldModel.exists()) oldModel.delete()
        
        // Re-copy if file missing or size mismatch (model was updated)
        val needsCopy = !modelFile.exists() || modelFile.length() != ASSET_MODEL_SIZE
        if (needsCopy) {
            if (modelFile.exists()) modelFile.delete()
            context.assets.open("arctic.onnx").use { it.copyTo(modelFile.outputStream()) }
        }
        val opts = OrtSession.SessionOptions().apply {
            // XNNPACK is ARM-optimized and manages its own thread pool.
            // When active, ORT's intra-op pool should be 1 to avoid contention.
            try {
                addXnnpack(emptyMap())
                setIntraOpNumThreads(1)
            } catch (_: Exception) {
                // XNNPACK unavailable — fall back to 2 ORT threads.
                setIntraOpNumThreads(2)
            }
            // Disable idle spinning to reduce CPU burn on battery-constrained devices.
            addConfigEntry("session.intra_op.allow_spinning", "0")
        }
        session   = env.createSession(modelFile.absolutePath, opts)
        tokenizer = WordPieceTokenizer(context)
    }

    /**
     * No-op — OrtSession and tokenizer are initialised eagerly in [init].
     * Call this instead of [embed] with an empty string to avoid a wasteful inference pass.
     */
    fun warmUp() = Unit

    /**
     * Embed a single string. Returns a L2-normalised [dim]-dim FloatArray.
     * Set [isQuery]=true when embedding a user question to apply the Arctic query prefix.
     */
    fun embed(text: String, isQuery: Boolean = false): FloatArray =
        embedBatch(listOf(text), isQuery)[0]

    /**
     * Embed a batch of strings in one ONNX call.
     * Returns Array<FloatArray> of shape [texts.size, dim], each L2-normalised.
     * Set [isQuery]=true when embedding user questions (applies Arctic query prefix).
     */
    fun embedBatch(texts: List<String>, isQuery: Boolean = false): Array<FloatArray> {
        val preparedTexts = if (isQuery) texts.map { QUERY_PREFIX + it } else texts
        require(texts.isNotEmpty()) { "texts must not be empty" }

        val maxLen   = 128
        val batch    = preparedTexts.size
        val encodings = preparedTexts.map { tokenizer.encode(it, maxLen) }

        val inputIds    = LongArray(batch * maxLen)
        val attnMask    = LongArray(batch * maxLen)
        val tokenTypes  = LongArray(batch * maxLen)
        encodings.forEachIndexed { b, enc ->
            enc.inputIds.copyInto(inputIds,   b * maxLen)
            enc.attentionMask.copyInto(attnMask,  b * maxLen)
            enc.tokenTypeIds.copyInto(tokenTypes, b * maxLen)
        }

        val shape = longArrayOf(batch.toLong(), maxLen.toLong())
        var tIds: OnnxTensor? = null
        var tMask: OnnxTensor? = null
        var tType: OnnxTensor? = null
        var outputs: OrtSession.Result? = null

        try {
            tIds  = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), shape)
            tMask = OnnxTensor.createTensor(env, LongBuffer.wrap(attnMask), shape)
            tType = OnnxTensor.createTensor(env, LongBuffer.wrap(tokenTypes), shape)

            outputs = session.run(mapOf(
                "input_ids"       to tIds,
                "attention_mask"  to tMask,
                "token_type_ids"  to tType,
            ))

            @Suppress("UNCHECKED_CAST")
            val hidden = outputs[0].value as Array<Array<FloatArray>>

            val result = Array(batch) { b ->
                meanPoolAndNorm(hidden[b], attnMask, maskOffset = b * maxLen, seqLen = maxLen)
            }

            return result
        } finally {
            tIds?.close()
            tMask?.close()
            tType?.close()
            outputs?.close()
        }
    }

    // ---------------------------------------------------------------------------
    // Mean pooling + L2 normalisation
    // ---------------------------------------------------------------------------

    private fun meanPoolAndNorm(
        hidden: Array<FloatArray>,
        mask: LongArray,
        maskOffset: Int,
        seqLen: Int,
    ): FloatArray {
        val pooled = FloatArray(dim)
        var count  = 0f

        for (t in 0 until seqLen) {
            if (mask[maskOffset + t] == 0L) continue
            val vec = hidden[t]
            for (d in 0 until dim) pooled[d] += vec[d]
            count++
        }
        if (count > 0f) for (d in 0 until dim) pooled[d] /= count

        // L2 normalise
        var norm = 0f
        for (v in pooled) norm += v * v
        norm = sqrt(norm)
        if (norm > 1e-9f) for (d in pooled.indices) pooled[d] /= norm

        return pooled
    }

    override fun close() = session.close()
}
