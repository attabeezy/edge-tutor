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
import java.io.FileInputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * LLM engine backed by llama.cpp via Llamatik.
 *
 * Model provisioning:
 *   - bundledModel builds include Qwen_Qwen3-0.6B-Q4_K_M.gguf in assets.
 *   - downloadModel builds fetch the same file into app-private storage.
 *
 * Llamatik is the only packaged LLM runtime in the Android app.
 */
class LlamaEngine(private val context: Context) : LlmEngine {

    private val modelLoaded = AtomicBoolean(false)
    private val warmUpComplete = AtomicBoolean(false)
    private val copyMutex   = Mutex()
    /** Ensures only one native generation runs at a time; released only after onComplete/onError. */
    private val genMutex    = Mutex()

    companion object {
        private const val TAG = "LlamaEngine"
        private const val MODEL_ASSET        = "Qwen_Qwen3-0.6B-Q4_K_M.gguf"
        private const val MODEL_BYTES        = 484_220_320L
        private const val MODEL_SHA256       = "9acfc1e001311f34b4252001b626f2e466d592a42065f66571bff3790d4e1b14"
        private const val MODEL_URL          = "https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf?download=true"
        private const val SYSTEM_PROMPT      = "Be concise. ASCII only."
        private const val MAX_RESPONSE_CHARS = 3_000
        private const val WARM_UP_PROMPT     = "Reply with the word ready."
        private val STOP_SEQUENCES = listOf(
            "<|im_end|>", "<|im_start|>", "<|endoftext|>",
            "\nHuman:", "\nUser:", "\nQuestion:", "\nAssistant:",
        )
        /** Longest stop sequence length — used to bound the tail scan window. */
        private val MAX_STOP_SEQ_LEN = STOP_SEQUENCES.maxOf { it.length }
    }

    override suspend fun copyModelIfNeeded(onProgress: (String?) -> Unit) = copyMutex.withLock {
        withContext(Dispatchers.IO) {
            val dest = File(context.filesDir, MODEL_ASSET)
            onProgress("Preparing model")
            if (verifyModelFile(dest)) {
                EdgeTutorPerf.log(
                    "llm_model_check",
                    "status" to "hit",
                    "model_asset" to MODEL_ASSET,
                    "bytes" to dest.length(),
                )
                onProgress(null)
                return@withContext
            }
            if (dest.exists() && !dest.delete()) {
                throw IllegalStateException("Failed to remove invalid model file")
            }

            val tmp = File(context.filesDir, "$MODEL_ASSET.tmp")
            try {
                if (assetExists(MODEL_ASSET)) {
                    copyBundledModel(tmp, dest, onProgress)
                } else {
                    downloadModel(tmp, dest, onProgress)
                }
                onProgress(null)
            } finally {
                if (tmp.exists()) tmp.delete()
            }
        }
    }

    private fun copyBundledModel(
        tmp: File,
        dest: File,
        onProgress: (String?) -> Unit,
    ) {
        Log.d(TAG, "Copying bundled model asset to internal storage...")
        onProgress("Copying bundled model")
        val copyStartNs = System.nanoTime()
        context.assets.open(MODEL_ASSET).use { src ->
            tmp.outputStream().use { src.copyTo(it) }
        }
        moveVerifiedTempModel(tmp, dest)
        Log.d(TAG, "Bundled model copy complete: ${dest.length()} bytes")
        EdgeTutorPerf.log(
            "llm_asset_copy",
            "model_asset" to MODEL_ASSET,
            "bytes" to dest.length(),
            "duration_ms" to EdgeTutorPerf.elapsedMs(copyStartNs),
        )
    }

    private fun downloadModel(
        tmp: File,
        dest: File,
        onProgress: (String?) -> Unit,
    ) {
        Log.d(TAG, "Downloading model from Hugging Face...")
        val downloadStartNs = System.nanoTime()
        val connection = (URL(MODEL_URL).openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 60_000
            instanceFollowRedirects = true
        }
        try {
            val responseCode = connection.responseCode
            if (responseCode !in 200..299) {
                throw IllegalStateException("Model download failed with HTTP $responseCode")
            }
            val totalBytes = connection.contentLengthLong.takeIf { it > 0L }
            var copied = 0L
            var lastPercent = -1
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            connection.inputStream.use { src ->
                tmp.outputStream().use { out ->
                    while (true) {
                        val read = src.read(buffer)
                        if (read < 0) break
                        out.write(buffer, 0, read)
                        copied += read
                        if (totalBytes != null) {
                            val percent = ((copied * 100) / totalBytes).toInt().coerceIn(0, 100)
                            if (percent != lastPercent) {
                                lastPercent = percent
                                onProgress("Downloading model: $percent%")
                            }
                        } else if (copied % (8L * 1024L * 1024L) < read) {
                            onProgress("Downloading model: ${copied / (1024L * 1024L)} MB")
                        }
                    }
                }
            }
            moveVerifiedTempModel(tmp, dest)
            Log.d(TAG, "Model download complete: ${dest.length()} bytes")
            EdgeTutorPerf.log(
                "llm_model_download",
                "model_asset" to MODEL_ASSET,
                "bytes" to dest.length(),
                "duration_ms" to EdgeTutorPerf.elapsedMs(downloadStartNs),
            )
        } finally {
            connection.disconnect()
        }
    }

    private fun moveVerifiedTempModel(tmp: File, dest: File) {
        if (!verifyModelFile(tmp)) {
            throw IllegalStateException(
                "Model verification failed for $MODEL_ASSET. Expected $MODEL_BYTES bytes and SHA-256 $MODEL_SHA256."
            )
        }
        if (dest.exists() && !dest.delete()) {
            throw IllegalStateException("Failed to replace existing model file")
        }
        if (!tmp.renameTo(dest)) {
            throw RuntimeException("Failed to rename temporary model file")
        }
    }

    private fun verifyModelFile(file: File): Boolean {
        if (!file.exists() || file.length() != MODEL_BYTES) return false
        return sha256(file) == MODEL_SHA256
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun assetExists(name: String): Boolean =
        context.assets.list("")?.contains(name) == true

    private suspend fun ensureModelLoaded(onProgress: (String?) -> Unit = {}) {
        if (modelLoaded.get()) return

        copyModelIfNeeded(onProgress)

        withContext(Dispatchers.IO) {
            onProgress("Loading model")
            Log.d(TAG, "Initializing llama.cpp with model: $MODEL_ASSET")
            val initStartNs = System.nanoTime()
            try {
                LlamaBridge.initGenerateModel(File(context.filesDir, MODEL_ASSET).absolutePath)
                modelLoaded.set(true)
                onProgress(null)
                Log.d(TAG, "llama.cpp initialization successful")
                EdgeTutorPerf.log(
                    "llm_native_init",
                    "model_asset" to MODEL_ASSET,
                    "duration_ms" to EdgeTutorPerf.elapsedMs(initStartNs),
                )
            } catch (e: Exception) {
                onProgress(null)
                Log.e(TAG, "Failed to initialize llama.cpp", e)
                throw e
            }
        }
    }

    override suspend fun initNativeModel() {
        ensureModelLoaded()
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
            val safePrompt = PromptSanitizer.sanitize(prompt)
            logSanitizationChange("llm_prompt_sanitization", source, safePrompt)
            LlamaBridge.generateStream(
                safePrompt.value,
                object : GenStream {
                    override fun onDelta(text: String) {
                        if (stopped) return
                        val safeDelta = PromptSanitizer.sanitize(text)
                        logSanitizationChange("llm_output_sanitization", source, safeDelta)
                        val delta = safeDelta.value
                        if (delta.isEmpty()) return
                        if (logMetrics && !firstTokenLogged) {
                            firstTokenLogged = true
                            EdgeTutorPerf.log(
                                "llm_decode_first_token",
                                "source" to source,
                                "llm_native_ttft_ms" to EdgeTutorPerf.elapsedMs(startNs),
                            )
                        }
                        val prevLen = sb.length
                        sb.append(delta)

                        // Only scan the tail (window = longest stop seq) to avoid
                        // O(response_length) search cost on every token.
                        val tailStart = maxOf(0, sb.length - MAX_STOP_SEQ_LEN - delta.length)
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
                            onToken(delta)
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

    private fun logSanitizationChange(
        eventName: String,
        source: String,
        sanitizedText: PromptSanitizer.SanitizedText,
    ) {
        if (!sanitizedText.changed) return
        EdgeTutorPerf.log(
            eventName,
            "source" to source,
            "sanitized_chars" to sanitizedText.value.length,
            "replacement_count" to sanitizedText.replacementCount,
            "dropped_count" to sanitizedText.droppedCount,
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
