package com.edgetutor.ingestion

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * Produces 384-dim L2-normalised sentence embeddings using all-MiniLM-L6-v2 (ONNX backend).
 *
 * Holds an OrtSession; call [close] when the owning ViewModel is cleared.
 * All methods are thread-safe as long as [close] is not called concurrently with [embed].
 */
class Embedder(context: Context) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val tokenizer: WordPieceTokenizer

    /** Embedding dimensionality — 384 for all-MiniLM-L6-v2. */
    val dim = 384

    init {
        val bytes = context.assets.open("minilm.onnx").readBytes()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)          // conservative for low-RAM devices
        }
        session   = env.createSession(bytes, opts)
        tokenizer = WordPieceTokenizer(context)
    }

    /** Embed a single string. Returns a L2-normalised [dim]-dim FloatArray. */
    fun embed(text: String): FloatArray = embedBatch(listOf(text))[0]

    /**
     * Embed a batch of strings in one ONNX call.
     * Returns Array<FloatArray> of shape [texts.size, dim], each L2-normalised.
     */
    fun embedBatch(texts: List<String>): Array<FloatArray> {
        require(texts.isNotEmpty()) { "texts must not be empty" }

        val maxLen   = 128
        val batch    = texts.size
        val encodings = texts.map { tokenizer.encode(it, maxLen) }

        // Flatten to [batch × maxLen]
        val inputIds    = LongArray(batch * maxLen)
        val attnMask    = LongArray(batch * maxLen)
        val tokenTypes  = LongArray(batch * maxLen)
        encodings.forEachIndexed { b, enc ->
            enc.inputIds.copyInto(inputIds,   b * maxLen)
            enc.attentionMask.copyInto(attnMask,  b * maxLen)
            enc.tokenTypeIds.copyInto(tokenTypes, b * maxLen)
        }

        val shape = longArrayOf(batch.toLong(), maxLen.toLong())
        val tIds  = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds),   shape)
        val tMask = OnnxTensor.createTensor(env, LongBuffer.wrap(attnMask),   shape)
        val tType = OnnxTensor.createTensor(env, LongBuffer.wrap(tokenTypes), shape)

        val outputs = session.run(mapOf(
            "input_ids"       to tIds,
            "attention_mask"  to tMask,
            "token_type_ids"  to tType,
        ))

        // last_hidden_state shape: [batch, seq, 384]
        @Suppress("UNCHECKED_CAST")
        val hidden = outputs[0].value as Array<Array<FloatArray>>

        val result = Array(batch) { b ->
            meanPoolAndNorm(hidden[b], attnMask, maskOffset = b * maxLen, seqLen = maxLen)
        }

        tIds.close(); tMask.close(); tType.close(); outputs.close()
        return result
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
