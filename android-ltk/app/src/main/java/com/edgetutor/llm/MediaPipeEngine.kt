package com.edgetutor.llm

import android.content.Context
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * LLM engine backed by MediaPipe LLM Inference API (Google AI Edge).
 *
 * Model file required in assets/:
 *   gemma-3-270m-it-q4_k_m.task  (~253 MB, FlatBuffers format)
 *   Download from Hugging Face: search "gemma-3-270m LiteRT" or "gemma-3-270m .task"
 *   Do NOT commit to git — copy manually before build.
 *
 * Use this engine if Llamatik is unavailable. To switch, change [ChatViewModel]:
 *   private val llm: LlmEngine by lazy { MediaPipeEngine(app) }
 */
class MediaPipeEngine(context: Context) : LlmEngine {

    private val inference: LlmInference

    init {
        val dest = File(context.filesDir, MODEL_ASSET)
        if (!dest.exists()) {
            context.assets.open(MODEL_ASSET).use { src ->
                dest.outputStream().use { src.copyTo(it) }
            }
        }
        val options = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(dest.absolutePath)
            .setMaxTokens(512)
            .build()
        inference = LlmInference.createFromOptions(context, options)
    }

    override suspend fun generate(prompt: String, onToken: (String) -> Unit): String =
        withContext(Dispatchers.IO) {
            // MediaPipe has no separate system prompt slot — prepend to user turn
            val fullPrompt = "$SYSTEM_PROMPT\n\n$prompt"
            suspendCancellableCoroutine { cont ->
                val sb = StringBuilder()
                try {
                    inference.generateResponseAsync(fullPrompt) { partialResult, done ->
                        onToken(partialResult)
                        sb.append(partialResult)
                        if (done) cont.resume(sb.toString())
                    }
                } catch (e: Exception) {
                    cont.resumeWithException(e)
                }
            }
        }

    override fun close() = inference.close()

    companion object {
        private const val MODEL_ASSET   = "gemma-3-270m-it-q4_k_m.task"
        private const val SYSTEM_PROMPT = "Be concise."
    }
}
