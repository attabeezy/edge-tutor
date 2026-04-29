package com.edgetutor.llm

import android.content.Context
import android.util.Log
import com.edgetutor.perf.EdgeTutorPerf
import com.llamatik.library.platform.GenStream
import com.llamatik.library.platform.LlamaBridge
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * LLM engine backed by llama.cpp via Llamatik.
 *
 * Model file required in assets/:
 *   LFM2.5-350M-Q4_K_M.gguf  (~267 MB)
 *   Download: hf download LiquidAI/LFM2.5-350M-GGUF LFM2.5-350M-Q4_K_M.gguf
 *   Do NOT commit to git — copy manually before build.
 *
 * Switch to [MediaPipeEngine] if Llamatik is unavailable.
 */
class LlamaEngine(private val context: Context) : LlmEngine {

    private val modelLoaded = AtomicBoolean(false)
    private val warmUpComplete = AtomicBoolean(false)
    private val copyMutex   = Mutex()
    /** Ensures only one native generation runs at a time; released only after onComplete/onError. */
    private val genMutex    = Mutex()

    companion object {
        private const val TAG = "LlamaEngine"
        private const val MODEL_ASSET        = "LFM2.5-350M-Q4_K_M.gguf"
        private const val SYSTEM_PROMPT      = "Be concise. Use plain ASCII only. Avoid special symbols."
        private const val MAX_RESPONSE_CHARS = 3_000
        private const val WARM_UP_PROMPT     = "Reply with the word ready."
        private val STOP_SEQUENCES = listOf(
            "<|im_end|>", "<|im_start|>", "<|endoftext|>",
            "\nHuman:", "\nUser:", "\nQuestion:", "\nAssistant:",
        )
        /** Longest stop sequence length — used to bound the tail scan window. */
        private val MAX_STOP_SEQ_LEN = STOP_SEQUENCES.maxOf { it.length }
    }

    override suspend fun copyModelIfNeeded() = copyMutex.withLock {
        withContext(Dispatchers.IO) {
            val dest = File(context.filesDir, MODEL_ASSET)
            // If it exists and is not a tiny/empty file, assume it's okay
            if (dest.exists() && dest.length() > 1_000_000) {
                EdgeTutorPerf.log(
                    "llm_asset_check",
                    "status" to "hit",
                    "model_asset" to MODEL_ASSET,
                    "bytes" to dest.length(),
                )
                return@withContext
            }

            Log.d(TAG, "Copying model asset to internal storage...")
            val copyStartNs = System.nanoTime()
            val tmp = File(context.filesDir, "$MODEL_ASSET.tmp")
            try {
                if (!assetExists(MODEL_ASSET)) {
                    throw IllegalStateException(
                        "Missing model asset '$MODEL_ASSET' in app/src/main/assets/. " +
                        "Copy it from models/lfm2.5-350m/ before launching the app."
                    )
                }
                context.assets.open(MODEL_ASSET).use { src ->
                    tmp.outputStream().use { src.copyTo(it) }
                }
                if (!tmp.renameTo(dest)) {
                    throw RuntimeException("Failed to rename temporary model file")
                }
                Log.d(TAG, "Model copy complete: ${dest.length()} bytes")
                EdgeTutorPerf.log(
                    "llm_asset_copy",
                    "model_asset" to MODEL_ASSET,
                    "bytes" to dest.length(),
                    "duration_ms" to EdgeTutorPerf.elapsedMs(copyStartNs),
                )
            } finally {
                if (tmp.exists()) tmp.delete()
            }
        }
    }

    private fun assetExists(name: String): Boolean =
        context.assets.list("")?.contains(name) == true

    private suspend fun ensureModelLoaded() {
        if (modelLoaded.get()) return

        copyModelIfNeeded()

        withContext(Dispatchers.IO) {
            Log.d(TAG, "Initializing llama.cpp with model: $MODEL_ASSET")
            val initStartNs = System.nanoTime()
            try {
                LlamaBridge.initGenerateModel(File(context.filesDir, MODEL_ASSET).absolutePath)
                modelLoaded.set(true)
                Log.d(TAG, "llama.cpp initialization successful")
                EdgeTutorPerf.log(
                    "llm_native_init",
                    "model_asset" to MODEL_ASSET,
                    "duration_ms" to EdgeTutorPerf.elapsedMs(initStartNs),
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize llama.cpp", e)
                throw e
            }
        }
    }

    override suspend fun generate(prompt: String, onToken: (String) -> Unit): String =
        withGenerateLock(caller = "query") {
            withContext(Dispatchers.IO) {
                ensureModelLoaded()
                Log.d(TAG, "Starting generation...")
                generateInternal(
                    prompt = buildChatPrompt(prompt),
                    source = "query",
                    logMetrics = true,
                    onToken = onToken,
                )
            }
        }

    override suspend fun warmUp() {
        withGenerateLock(caller = "warm_up") {
            withContext(Dispatchers.IO) {
                ensureModelLoaded()
                if (warmUpComplete.get()) return@withContext
                generateInternal(
                    prompt = buildChatPrompt(WARM_UP_PROMPT),
                    source = "warm_up",
                    logMetrics = true,
                    onToken = {},
                )
                warmUpComplete.set(true)
            }
        }
    }

    override fun close() {
        modelLoaded.set(false)
        warmUpComplete.set(false)
    }

    private fun buildChatPrompt(userContent: String): String =
        "<|im_start|>system\n$SYSTEM_PROMPT<|im_end|>\n" +
        "<|im_start|>user\n$userContent<|im_end|>\n" +
        "<|im_start|>assistant\n"

    private suspend fun generateInternal(
        prompt: String,
        source: String,
        logMetrics: Boolean,
        onToken: (String) -> Unit,
    ): String =
        suspendCancellableCoroutine { cont ->
            val sb      = StringBuilder()
            var stopped = false
            val startNs = System.nanoTime()
            var firstTokenLogged = false
            LlamaBridge.generateStream(
                prompt,
                object : GenStream {
                    override fun onDelta(text: String) {
                        if (stopped) return
                        if (logMetrics && !firstTokenLogged && text.isNotEmpty()) {
                            firstTokenLogged = true
                            EdgeTutorPerf.log(
                                "llm_decode_first_token",
                                "source" to source,
                                "llm_native_ttft_ms" to EdgeTutorPerf.elapsedMs(startNs),
                            )
                        }
                        val prevLen = sb.length
                        sb.append(text)

                        // Only scan the tail (window = longest stop seq) to avoid
                        // O(response_length) search cost on every token.
                        val tailStart = maxOf(0, sb.length - MAX_STOP_SEQ_LEN - text.length)
                        val tail = sb.substring(tailStart)
                        val seqIdx = STOP_SEQUENCES
                            .mapNotNull { seq -> tail.indexOf(seq).takeIf { it >= 0 }?.let { tailStart + it } }
                            .minOrNull()

                        if (seqIdx != null || sb.length > MAX_RESPONSE_CHARS) {
                            stopped = true
                            val stopIdx = seqIdx ?: MAX_RESPONSE_CHARS
                            val newPart = sb.substring(prevLen, minOf(stopIdx, sb.length))
                            if (newPart.isNotEmpty()) onToken(newPart)
                        } else {
                            onToken(text)
                        }
                    }

                    override fun onComplete() {
                        if (logMetrics) {
                            EdgeTutorPerf.log(
                                "llm_decode_total",
                                "source" to source,
                                "duration_ms" to EdgeTutorPerf.elapsedMs(startNs),
                            )
                        }
                        val finalText = if (stopped) {
                            // Scan only the tail for the stop marker position.
                            val tailStart = maxOf(0, sb.length - MAX_STOP_SEQ_LEN)
                            val tail = sb.substring(tailStart)
                            val seqIdx = STOP_SEQUENCES
                                .mapNotNull { seq -> tail.indexOf(seq).takeIf { it >= 0 }?.let { tailStart + it } }
                                .minOrNull()
                            sb.substring(0, seqIdx ?: minOf(MAX_RESPONSE_CHARS, sb.length))
                        } else {
                            sb.toString()
                        }
                        if (cont.isActive) cont.resume(finalText)
                    }

                    override fun onError(message: String) {
                        Log.e(TAG, "Llamatik error: $message")
                        if (logMetrics) {
                            EdgeTutorPerf.log(
                                "llm_decode_total",
                                "source" to source,
                                "duration_ms" to EdgeTutorPerf.elapsedMs(startNs),
                                "status" to "error",
                            )
                        }
                        if (cont.isActive) cont.resumeWithException(RuntimeException(message))
                    }
                }
            )
        }

    private suspend fun <T> withGenerateLock(caller: String, block: suspend () -> T): T {
        val waitStartNs = System.nanoTime()
        return genMutex.withLock {
            val waitMs = EdgeTutorPerf.elapsedMs(waitStartNs)
            if (waitMs > 0L) {
                EdgeTutorPerf.log(
                    "llm_gen_mutex_wait",
                    "caller" to caller,
                    "wait_ms" to waitMs,
                )
            }
            block()
        }
    }
}
