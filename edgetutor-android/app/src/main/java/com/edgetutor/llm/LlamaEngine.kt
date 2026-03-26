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
 *   qwen2.5-0.5b-instruct-q4_k_m.gguf  (~350 MB)
 *   Download: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
 *   Do NOT commit to git — copy manually before build.
 *
 * Switch to [MediaPipeEngine] if Llamatik is unavailable.
 */
class LlamaEngine(private val context: Context) : LlmEngine {

    private var modelLoaded = false

    private suspend fun ensureModelLoaded() {
        if (modelLoaded) return
        val dest = File(context.filesDir, MODEL_ASSET)
        if (!dest.exists()) {
            context.assets.open(MODEL_ASSET).use { src ->
                dest.outputStream().use { src.copyTo(it) }
            }
        }
        LlamaBridge.initGenerateModel(dest.absolutePath)
        modelLoaded = true
    }

    override suspend fun generate(prompt: String, onToken: (String) -> Unit): String =
        withContext(Dispatchers.IO) {
            ensureModelLoaded()
            suspendCancellableCoroutine { cont ->
                val sb = StringBuilder()
                LlamaBridge.generateStreamWithContext(
                    SYSTEM_PROMPT,
                    "",
                    prompt,
                    object : GenStream {
                        override fun onDelta(text: String) {
                            onToken(text)
                            sb.append(text)
                        }
                        override fun onComplete() { cont.resume(sb.toString()) }
                        override fun onError(message: String) {
                            cont.resumeWithException(RuntimeException(message))
                        }
                    }
                )
            }
        }

    override fun close() {
        // LlamaBridge is a process-scoped singleton; no explicit release in the API
        modelLoaded = false
    }

    companion object {
        private const val MODEL_ASSET  = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        private const val SYSTEM_PROMPT = "Be concise."
    }
}
