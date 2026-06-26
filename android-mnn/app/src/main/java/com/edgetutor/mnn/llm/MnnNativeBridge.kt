package com.edgetutor.mnn.llm

/**
 * JNI bridge to the pre-built MNN native library (monolithic build with LLM).
 *
 * Required native library
 * -----------------------
 *  app/src/main/jniLibs/arm64-v8a/libMNN.so
 *
 * This is the single shared library produced by the MNN Android build with
 * MNN_BUILD_LLM=true.  The LLM engine is baked into libMNN.so rather than
 * emitted as a separate libMNN_LLM.so.
 *
 * Build the .so from the MNN source tree (MNN/ in the repo root) using:
 *
 *   cd MNN
 *   mkdir build-android && cd build-android
 *   cmake .. \
 *     -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
 *     -DANDROID_ABI=arm64-v8a \
 *     -DANDROID_PLATFORM=android-29 \
 *     -DMNN_BUILD_LLM=ON \
 *     -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
 *     -DMNN_ARM82=ON \
 *     -DMNN_LOW_MEMORY=ON
 *   make -j$(nproc) MNN_LLM
 *   cp libMNN.so \
 *       ../../android-mnn/app/src/main/jniLibs/arm64-v8a/libMNN.so
 *
 * The JNI method names below must match the native implementations compiled
 * into libMNN.so.  The MNN demo reference is:
 *   MNN/apps/Android/MnnLlmChat/app/src/main/cpp/llm_mnn_jni.cpp
 *
 * Native lifecycle
 * ----------------
 *  initSession  -> allocates an LlmSession on the heap, returns its pointer as Long
 *  submitPrompt -> runs one prefill+decode pass; calls progressListener per token
 *  releaseSession -> deletes the LlmSession and frees all native memory
 *
 * Threading
 * ---------
 *  All native calls must be made from a background thread (never the Main thread).
 *  MnnEngine enforces this by always dispatching via Dispatchers.IO.
 */
object MnnNativeBridge {

    init {
        // Load the monolithic MNN shared library.
        // MNN_BUILD_LLM=true bakes LLM support into libMNN.so directly.
        // The name here maps to libMNN.so in jniLibs/arm64-v8a/.
        // Loads libedgetutor_mnn.so — our thin JNI bridge compiled against libMNN.so.
        // libMNN.so (monolithic, MNN_BUILD_LLM=true) is packaged alongside in jniLibs/.
        System.loadLibrary("edgetutor_mnn")
    }

    /**
     * Initialise an MNN LLM session and return a native pointer.
     *
     * @param modelDir   Absolute path to the directory containing config.json
     *                   and the .mnn weight files.
     * @param configJson JSON string to merge into the model's default config.
     *                   Pass "{}" to use the model's own config.json unchanged.
     * @return Native pointer (cast to Long) for the created LlmSession.
     *         Returns 0 on failure (native side throws IllegalStateException).
     */
    @JvmStatic
    external fun initSession(modelDir: String, configJson: String): Long

    /**
     * Run one full prefill+decode pass for [prompt].
     *
     * The [progressListener] is called once per generated token with the token
     * text.  Pass null to suppress per-token callbacks and receive only the
     * final result.
     *
     * Blocks until generation completes or is stopped by the listener.
     *
     * @param sessionPtr     Pointer returned by [initSession].
     * @param prompt         The full prompt string (UTF-8, already formatted).
     * @param keepHistory    If true, the session retains KV-cache state for
     *                       subsequent turns.  Pass false for single-turn use.
     * @param progressListener  Called on each token delta (may be null).
     * @return Map of timing metrics (prompt_len, decode_len, prefill_time_us,
     *         decode_time_us) as returned by the native layer.
     */
    @JvmStatic
    external fun submitPrompt(
        sessionPtr: Long,
        prompt: String,
        keepHistory: Boolean,
        progressListener: MnnProgressListener?,
    ): Map<String, Long>

    /**
     * Release all native resources held by the session.
     * Must be called exactly once when the session is no longer needed.
     *
     * @param sessionPtr  Pointer returned by [initSession].
     */
    @JvmStatic
    external fun releaseSession(sessionPtr: Long)
}
