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
