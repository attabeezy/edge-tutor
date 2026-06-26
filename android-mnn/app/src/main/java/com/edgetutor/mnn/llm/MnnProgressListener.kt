package com.edgetutor.mnn.llm

/**
 * Callback interface invoked once per generated token by [MnnNativeBridge.submitPrompt].
 *
 * The native side calls [onProgress] on the JNI thread (background).
 * Implementations must NOT touch Android UI from this callback directly;
 * use a coroutine channel or StateFlow to propagate tokens to the UI.
 */
fun interface MnnProgressListener {
    /**
     * Called for each new token delta produced by the LLM.
     *
     * @param token  The token text (UTF-8).  Null signals end-of-generation.
     * @return true to stop generation early, false to continue.
     */
    fun onProgress(token: String?): Boolean
}
