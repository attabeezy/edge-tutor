package com.edgetutor.llm

/**
 * Common interface for on-device LLM inference.
 *
 * Implementations
 * ---------------
 *  - [LlamaEngine]  — llama.cpp via Llamatik (primary, requires Llamatik AAR)
 *
 * Swap implementations in [com.edgetutor.viewmodel.ChatViewModel] without changing
 * any other code.
 */
interface LlmEngine : AutoCloseable {

    /**
     * Copy the model asset file to internal storage if it is not already there.
     * This is the slow I/O phase (~5–15s on first launch) and can be started in the
     * background at app startup, well before the user selects a document.
     * Default implementation is a no-op.
     */
    suspend fun copyModelIfNeeded() {}

    /**
     * Load the native model weights into memory without running a dummy decode.
     * Called at app startup (after [copyModelIfNeeded]) so the expensive native init
     * is hidden behind the document-picker screen rather than paid at first query.
     * Safe to call multiple times. Default implementation is a no-op.
     */
    suspend fun initNativeModel() {}

    /**
     * Pre-load the model into memory so the first [generate] call has no loading overhead.
     * Safe to call multiple times. Default implementation is a no-op.
     */
    suspend fun warmUp() {}

    /**
     * Generate a response for [prompt], calling [onToken] for each new token as it
     * is produced (streaming).
     *
     * Must be called from a coroutine (not the main thread). Implementations should
     * use [kotlinx.coroutines.Dispatchers.IO] internally.
     *
     * Returns the complete response string after generation finishes.
     */
    suspend fun generate(prompt: String, onToken: (String) -> Unit): String
}
