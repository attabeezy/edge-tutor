package com.edgetutor.mnn.llm

/**
 * Common interface for on-device LLM inference.
 *
 * Implementations
 * ---------------
 *  - [MnnEngine]   — MNN-LLM via pre-built libMNN_LLM.so (primary for this module)
 *
 * Swap implementations in [com.edgetutor.mnn.viewmodel.ChatViewModel] without
 * changing any other code.
 */
interface LlmEngine : AutoCloseable {

    /**
     * Copy the model directory from assets (or another source) to internal
     * storage if it is not already present.
     * This is the slow I/O phase and should be started in the background at
     * app startup, well before the user selects a document.
     * Default implementation is a no-op.
     */
    suspend fun copyModelIfNeeded() {}

    /**
     * Load the native model weights into memory.
     * Called at app startup (after [copyModelIfNeeded]) so the expensive native
     * init is hidden behind the document-picker screen.
     * Safe to call multiple times — subsequent calls are no-ops.
     * Default implementation is a no-op.
     */
    suspend fun initNativeModel() {}

    /**
     * Run a short warm-up generation so the first real [generate] call has
     * no cold-start latency.
     * Safe to call multiple times.
     * Default implementation is a no-op.
     */
    suspend fun warmUp() {}

    /**
     * Generate a response for [prompt], calling [onToken] for each new token
     * as it is produced (streaming).
     *
     * Must be called from a coroutine (never the main thread).
     * Implementations dispatch internally via [kotlinx.coroutines.Dispatchers.IO].
     *
     * @return The complete response string after generation finishes.
     */
    suspend fun generate(prompt: String, onToken: (String) -> Unit): String =
        generateMeasured(prompt, onToken).text

    suspend fun generateMeasured(
        prompt: String,
        onToken: (String) -> Unit,
    ): GenerationResult
}

data class GenerationMetrics(
    val promptTokens: Long,
    val decodeTokens: Long,
    val prefillUs: Long,
    val decodeUs: Long,
    val visibleTtftMs: Long,
    val totalMs: Long,
)

data class GenerationResult(
    val text: String,
    val metrics: GenerationMetrics,
    val thinking: String = "",
)
