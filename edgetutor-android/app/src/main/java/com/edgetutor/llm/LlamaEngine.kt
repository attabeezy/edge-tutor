package com.edgetutor.llm

import android.content.Context
import com.llamatik.library.platform.GenStream
import com.llamatik.library.platform.LlamaBridge
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * LLM engine backed by llama.cpp via Llamatik.
 *
 * Model file required in assets/:
 *   gemma-3-270m-it-Q4_K_M.gguf  (~253 MB)
 *   Download: huggingface-cli download unsloth/gemma-3-270m-it-GGUF gemma-3-270m-it-Q4_K_M.gguf
 *   Do NOT commit to git — copy manually before build.
 *
 * Switch to [MediaPipeEngine] if Llamatik is unavailable.
 */
class LlamaEngine(private val context: Context) : LlmEngine {

    private var modelLoaded = false

    /**
     * Copies the GGUF asset to internal storage if not already present.
     * This is the slow one-time I/O step (~5–15s on first launch, ~0ms on subsequent launches).
     * Call this eagerly at app startup so the file is ready before the user picks a document.
     */
    override suspend fun copyModelIfNeeded() = withContext(Dispatchers.IO) {
        val dest = File(context.filesDir, MODEL_ASSET)
        if (!dest.exists()) {
            context.assets.open(MODEL_ASSET).use { src ->
                dest.outputStream().use { src.copyTo(it) }
            }
        }
    }

    private suspend fun ensureModelLoaded() {
        if (modelLoaded) return
        copyModelIfNeeded()          // no-op if already copied
        LlamaBridge.initGenerateModel(File(context.filesDir, MODEL_ASSET).absolutePath)
        modelLoaded = true
    }

    override suspend fun generate(prompt: String, onToken: (String) -> Unit): String =
        withContext(Dispatchers.IO) {
            ensureModelLoaded()
            suspendCancellableCoroutine { cont ->
                val sb      = StringBuilder()
                var stopped = false
                LlamaBridge.generateStreamWithContext(
                    SYSTEM_PROMPT,
                    "",
                    prompt,
                    object : GenStream {
                        override fun onDelta(text: String) {
                            if (stopped) return
                            val prevLen = sb.length
                            sb.append(text)
                            // Hard cap — safety net if no stop sequence fires
                            val capIdx = if (sb.length > MAX_RESPONSE_CHARS) MAX_RESPONSE_CHARS else -1
                            // Earliest stop sequence in the accumulated buffer
                            val seqIdx = STOP_SEQUENCES
                                .mapNotNull { seq -> sb.indexOf(seq).takeIf { it >= 0 } }
                                .minOrNull() ?: -1
                            val stopIdx = when {
                                seqIdx >= 0 && capIdx >= 0 -> minOf(seqIdx, capIdx)
                                seqIdx >= 0               -> seqIdx
                                capIdx >= 0               -> capIdx
                                else                      -> -1
                            }
                            if (stopIdx >= 0) {
                                stopped = true
                                val clean   = sb.substring(0, stopIdx)
                                val newPart = clean.drop(prevLen)
                                if (newPart.isNotEmpty()) onToken(newPart)
                                cont.resume(clean)
                            } else {
                                onToken(text)
                            }
                        }
                        override fun onComplete() {
                            if (!stopped) cont.resume(sb.toString())
                        }
                        override fun onError(message: String) {
                            if (!stopped) cont.resumeWithException(RuntimeException(message))
                        }
                    }
                )
            }
        }

    override suspend fun warmUp() { ensureModelLoaded() }

    override fun close() {
        // LlamaBridge is a process-scoped singleton; no explicit release in the API
        modelLoaded = false
    }

    companion object {
        private const val MODEL_ASSET        = "gemma-3-270m-it-Q4_K_M.gguf"
        private const val SYSTEM_PROMPT      = "Be concise."
        // Safety net: truncate if no stop sequence fires (~600 words)
        private const val MAX_RESPONSE_CHARS = 3_000
        // Llamatik formats prompts in Gemma chat template (<start_of_turn>/<end_of_turn>)
        // regardless of the model loaded. Stop sequences cover both Gemma format and
        // plain role-play prefixes as fallbacks.
        private val STOP_SEQUENCES = listOf(
            "<end_of_turn>",     // Gemma: end of assistant turn (primary)
            "<start_of_turn>",   // Gemma: opening of any new turn (catches looping)
            "<|im_end|>",        // ChatML fallback
            "<|im_start|>",      // ChatML fallback
            "<|endoftext|>",
            "\nHuman:",
            "\nUser:",
            "\nQuestion:",
            "\nAssistant:",
        )
    }
}
