package com.edgetutor.mnn.llm

import android.content.Context
import android.util.Log
import com.edgetutor.mnn.perf.EdgeTutorPerf
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import org.json.JSONObject

/**
 * LLM engine backed by MNN-LLM via the pre-built libMNN_LLM.so.
 *
 * Model: Qwen3.5-0.8B-MNN
 * -------------------------
 * Located at: models/Qwen3.5-0.8B-MNN/ in the project root.
 *
 * Required files in the model directory (pushed to device):
 *   config.json         — MNN-LLM session config (required by native layer)
 *   llm.mnn             — model graph (~2 MB)
 *   llm.mnn.weight      — quantised weights (~448 MB)
 *   llm.mnn.json        — supplementary graph metadata
 *   llm_config.json     — architecture config
 *   tokenizer.txt       — BPE tokenizer (used by MNN internally)
 *   visual.mnn          — vision encoder graph (present but unused for text-only RAG)
 *   visual.mnn.weight   — vision encoder weights
 *
 * Push to device:
 *   adb push models\Qwen3.5-0.8B-MNN\ /sdcard/Download/mnn_model/
 *   adb shell run-as com.edgetutor.mnn \
 *     cp -r /sdcard/Download/mnn_model /data/data/com.edgetutor.mnn/files/mnn_model
 *
 * Thinking mode
 * -------------
 * Qwen3.5 defaults to enable_thinking=true which prepends <think>...</think>
 * blocks before answers. For a RAG tutor we want direct answers, so we
 * override via the configJson merge: {"jinja":{"context":{"enable_thinking":false}}}.
 * <think> / </think> are also added to STOP_SEQUENCES as a defence-in-depth.
 *
 * Prompt format
 * -------------
 * Qwen3.5 uses the same ChatML wrapping as Qwen2.5:
 *   <|im_start|>system\n...<|im_end|>\n
 *   <|im_start|>user\n...<|im_end|>\n
 *   <|im_start|>assistant\n
 *
 * Thread safety
 * -------------
 * Only one native generation runs at a time — enforced by [genMutex].
 * All native calls are dispatched on Dispatchers.IO.
 */
class MnnEngine(private val context: Context) : LlmEngine {

    private val modelLoaded   = AtomicBoolean(false)
    private val warmUpDone    = AtomicBoolean(false)
    private val copyMutex     = Mutex()
    /** Serializes native session initialization so warm-up/generation can await it. */
    private val initMutex     = Mutex()
    /** Ensures only one native generation runs at a time. */
    private val genMutex      = Mutex()

    /** Opaque pointer to the native LlmSession allocated by [MnnNativeBridge.initSession]. */
    @Volatile private var sessionPtr: Long = 0L

    companion object {
        private const val TAG                = "MnnEngine"
        private const val SYSTEM_PROMPT      = "Be concise. ASCII only."
        private const val MAX_RESPONSE_CHARS = 3_000
        private const val WARM_UP_PROMPT     = "Reply with the word ready."

        /**
         * Config JSON merged into the model's config.json at session init.
         *
         * enable_thinking=false: Qwen3.5-0.8B defaults to chain-of-thought
         * thinking mode which prepends <think>...</think> blocks.  For a RAG
         * tutor we want direct, concise answers so we disable thinking here.
         * The native layer merges this over the model's own config.json.
         */
        private val SESSION_CONFIG_JSON: String = JSONObject()
            .put("jinja", JSONObject()
                .put("context", JSONObject()
                    .put("enable_thinking", false)))
            .toString()

        /**
         * Qwen3.5 + ChatML stop sequences.
         * Thinking tags are filtered from the stream separately. They must not
         * be stop sequences because Qwen can emit <think> before the answer.
         */
        private val STOP_SEQUENCES = listOf(
            "<|im_end|>", "<|im_start|>", "<|endoftext|>",
            "\nHuman:", "\nUser:", "\nQuestion:", "\nAssistant:",
        )
        private val MAX_STOP_SEQ_LEN = STOP_SEQUENCES.maxOf { it.length }
    }

    // -------------------------------------------------------------------------
    // LlmEngine implementation
    // -------------------------------------------------------------------------

    // Resolves model directory: internal storage first, then app-specific external
    // storage (getExternalFilesDir — no permission needed on Android 10+).
    private fun resolveModelDir(): File {
        MnnModelManager.resolveReadyModelDir(context)?.let { return it }
        return MnnModelManager.internalModelDir(context)  // default; initNativeModel will throw a clear error
    }

    override suspend fun copyModelIfNeeded() = copyMutex.withLock {
        withContext(Dispatchers.IO) {
            val modelDir = MnnModelManager.resolveReadyModelDir(context)
            if (modelDir != null) {
                EdgeTutorPerf.log("llm_asset_check", "status" to "hit", "model_dir" to modelDir.absolutePath)
                return@withContext
            }
            val state = MnnModelManager.validate(context)
            EdgeTutorPerf.log(
                "llm_asset_check",
                "status" to state.kind,
                "model_dir" to (state.modelDir ?: ""),
                "missing" to state.missingFiles.joinToString(","),
            )
        }
    }

    override suspend fun initNativeModel() {
        if (modelLoaded.get()) return
        initMutex.withLock {
            if (modelLoaded.get()) return
            withContext(Dispatchers.IO) {
                val modelDir = resolveModelDir()
                val configFile = File(modelDir, MnnModelManager.CONFIG_FILE)
                if (!configFile.exists()) {
                    throw IllegalStateException(
                        "MNN config.json not found at ${configFile.absolutePath}. " +
                        "Import the MNN model folder and ensure all required files are present."
                    )
                }
                Log.d(TAG, "Initialising MNN-LLM session with model at: ${modelDir.absolutePath}")
                val startNs = System.nanoTime()
                val ptr = MnnNativeBridge.initSession(
                    modelDir   = modelDir.absolutePath,
                    configJson = SESSION_CONFIG_JSON,  // disable thinking, keep all other defaults
                )
                if (ptr == 0L) {
                    throw IllegalStateException("MNN initSession returned null pointer — check logcat for native errors.")
                }
                sessionPtr = ptr
                modelLoaded.set(true)
                EdgeTutorPerf.log(
                    "llm_native_init",
                    "model_dir"   to MnnModelManager.MODEL_DIR_NAME,
                    "duration_ms" to EdgeTutorPerf.elapsedMs(startNs),
                )
                Log.d(TAG, "MNN-LLM session initialised successfully (ptr=$ptr)")
            }
        }
    }

    override suspend fun warmUp() {
        withGenerateLock("warm_up") {
            initNativeModel()
            withContext(Dispatchers.IO) {
                ensureSessionReady()
                if (warmUpDone.get()) return@withContext
                generateInternal(
                    prompt      = buildChatPrompt(WARM_UP_PROMPT),
                    source      = "warm_up",
                    logMetrics  = true,
                    onToken     = {},
                )
                warmUpDone.set(true)
            }
        }
    }

    override suspend fun generate(prompt: String, onToken: (String) -> Unit): String =
        withGenerateLock("query") {
            initNativeModel()
            withContext(Dispatchers.IO) {
                ensureSessionReady()
                generateInternal(
                    prompt     = buildChatPrompt(prompt),
                    source     = "query",
                    logMetrics = true,
                    onToken    = onToken,
                )
            }
        }

    override fun close() {
        val ptr = sessionPtr
        if (ptr != 0L) {
            sessionPtr = 0L
            modelLoaded.set(false)
            warmUpDone.set(false)
            try {
                MnnNativeBridge.releaseSession(ptr)
                Log.d(TAG, "MNN-LLM session released (ptr=$ptr)")
            } catch (e: Exception) {
                Log.e(TAG, "Error releasing MNN session", e)
            }
        }
    }

    // -------------------------------------------------------------------------
    // Internal generation
    // -------------------------------------------------------------------------

    private fun ensureSessionReady() {
        if (sessionPtr == 0L) {
            throw IllegalStateException("MnnEngine: session not initialised. Call initNativeModel() first.")
        }
    }

    private fun generateInternal(
        prompt: String,
        source: String,
        logMetrics: Boolean,
        onToken: (String) -> Unit,
    ): String {
        val sb      = StringBuilder()
        var stopped = false
        val thinkingFilter = ThinkingTagFilter()
        val startNs = System.nanoTime()
        var firstTokenLogged = false
        val safePrompt = PromptSanitizer.sanitize(prompt)
        if (safePrompt.changed) {
            EdgeTutorPerf.log(
                "llm_prompt_sanitization",
                "source"            to source,
                "sanitized_chars"   to safePrompt.value.length,
                "replacement_count" to safePrompt.replacementCount,
                "dropped_count"     to safePrompt.droppedCount,
            )
        }

        val metrics = MnnNativeBridge.submitPrompt(
            sessionPtr = sessionPtr,
            prompt     = safePrompt.value,
            keepHistory = false,   // single-turn; multi-turn via prompt construction
            progressListener = MnnProgressListener { rawToken ->
                if (stopped || rawToken == null) return@MnnProgressListener stopped
                val safeDelta = PromptSanitizer.sanitize(rawToken)
                val delta = thinkingFilter.filter(safeDelta.value)
                if (delta.isEmpty()) return@MnnProgressListener false

                if (logMetrics && !firstTokenLogged) {
                    firstTokenLogged = true
                    EdgeTutorPerf.log(
                        "llm_decode_first_token",
                        "source"           to source,
                        "llm_native_ttft_ms" to EdgeTutorPerf.elapsedMs(startNs),
                    )
                }
                val prevLen = sb.length
                sb.append(delta)

                // Tail-scan only (O(1) per token) for stop sequences.
                val tailStart = maxOf(0, sb.length - MAX_STOP_SEQ_LEN - delta.length)
                val tail      = sb.substring(tailStart)
                val seqIdx    = STOP_SEQUENCES
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
                stopped   // return true to stop native generation when we hit a stop sequence
            }
        )

        if (logMetrics) {
            EdgeTutorPerf.log(
                "llm_decode_total",
                "source"       to source,
                "duration_ms"  to EdgeTutorPerf.elapsedMs(startNs),
                "prompt_len"   to (metrics["prompt_len"] ?: 0L),
                "decode_len"   to (metrics["decode_len"] ?: 0L),
                "prefill_us"   to (metrics["prefill_time"] ?: 0L),
                "decode_us"    to (metrics["decode_time"] ?: 0L),
            )
        }

        return if (stopped) {
            val tailStart = maxOf(0, sb.length - MAX_STOP_SEQ_LEN)
            val tail      = sb.substring(tailStart)
            val seqIdx    = STOP_SEQUENCES
                .mapNotNull { seq -> tail.indexOf(seq).takeIf { it >= 0 }?.let { tailStart + it } }
                .minOrNull()
            sb.substring(0, seqIdx ?: minOf(MAX_RESPONSE_CHARS, sb.length))
        } else {
            sb.toString()
        }
    }

    private fun buildChatPrompt(userContent: String): String =
        "<|im_start|>system\n$SYSTEM_PROMPT<|im_end|>\n" +
        "<|im_start|>user\n$userContent<|im_end|>\n" +
        "<|im_start|>assistant\n"

    private suspend fun <T> withGenerateLock(caller: String, block: suspend () -> T): T {
        val waitStartNs = System.nanoTime()
        return genMutex.withLock {
            val waitMs = EdgeTutorPerf.elapsedMs(waitStartNs)
            if (waitMs > 0L) {
                EdgeTutorPerf.log("llm_gen_mutex_wait", "caller" to caller, "wait_ms" to waitMs)
            }
            block()
        }
    }
}
