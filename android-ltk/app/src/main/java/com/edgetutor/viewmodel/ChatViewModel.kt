package com.edgetutor.viewmodel

import android.app.ActivityManager
import android.app.Application
import android.os.SystemClock
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.edgetutor.data.db.DocumentEntity
import com.edgetutor.ingestion.Embedder
import com.edgetutor.llm.LlamaEngine
import com.edgetutor.llm.LlmEngine
import com.edgetutor.llm.PromptSanitizer
import com.edgetutor.perf.EdgeTutorPerf
import com.edgetutor.store.FlatIndex
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File
import kotlin.math.ceil

data class ChatMessage(
    val role: Role,
    val text: String,
    /** Short excerpts of the retrieved chunks — shown as source attribution in the UI. */
    val sources: List<String> = emptyList(),
)

enum class Role { USER, ASSISTANT }

class ChatViewModel(app: Application) : AndroidViewModel(app) {

    private val llm: LlmEngine by lazy { LlamaEngine(app) }

    init {
        // Start copying the model asset to internal storage immediately so the file
        // is ready on disk by the time the user selects a document and warmUp() fires.
        // On subsequent launches the file already exists and this completes instantly.
        viewModelScope.launch(Dispatchers.IO) {
            try { llm.copyModelIfNeeded() }
            catch (e: Exception) { _errorMessage.value = "Model unavailable: ${e.message}" }
        }
    }

    private val _messages    = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages

    private val _isThinking  = MutableStateFlow(false)
    val isThinking: StateFlow<Boolean> = _isThinking

    private val _lastThinkingDurationMs = MutableStateFlow<Long?>(null)
    val lastThinkingDurationMs: StateFlow<Long?> = _lastThinkingDurationMs.asStateFlow()

    private val _isWarmingUp = MutableStateFlow(false)
    val isWarmingUp: StateFlow<Boolean> = _isWarmingUp

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage

    /** Emits true when the active document was flagged as a likely scanned PDF. */
    private val _isLikelyScanned = MutableStateFlow(false)
    val isLikelyScanned: StateFlow<Boolean> = _isLikelyScanned

    private val _activeDocumentId = MutableStateFlow<Long?>(null)
    val activeDocumentId: StateFlow<Long?> = _activeDocumentId.asStateFlow()

    companion object {
        /**
         * Below this free-memory threshold we keep the previous aggressive behavior
         * of releasing the embedder around generation to reduce memory pressure.
         */
        private const val LOW_MEM_THRESHOLD_MB = 120L
        /**
         * Cosine similarity floor for in-scope queries.
         * Derived from Python MAX_RELEVANT_DISTANCE=1.4 on normalized vectors:
         *   cos_sim = 1 - (dist² / 2) = 1 - (1.96 / 2) ≈ 0.02
         */
        private const val MIN_COSINE_SIM = 0.02f
        private const val MIN_LEXICAL_OVERLAP = 1

        private val STOPWORDS = setOf(
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "i", "you", "he", "she",
            "it", "we", "they", "what", "how", "why", "when", "where", "who",
            "which", "this", "that", "these", "those", "of", "in", "on", "at",
            "to", "for", "with", "by", "from", "about", "into", "than", "or",
            "and", "but", "if", "not", "no", "so",
        )
    }

    private var index: FlatIndex?    = null
    var activeDoc: DocumentEntity?   = null
        private set
    /**
     * Reused across queries to avoid paying ORT session init cost per turn.
     * Closed just before llm.generate() fires (to free ~23 MB), then reopened
     * after generation completes so the next query finds it ready.
     */
    private var embedder: Embedder?  = null

    private fun getAvailableMemoryMB(): Long {
        val am = getApplication<Application>().getSystemService(Application.ACTIVITY_SERVICE) as ActivityManager
        val info = ActivityManager.MemoryInfo()
        am.getMemoryInfo(info)
        return info.availMem / (1024 * 1024)
    }

    private fun shouldReleaseEmbedderForGeneration(): Boolean =
        getAvailableMemoryMB() < LOW_MEM_THRESHOLD_MB

    // ---------------------------------------------------------------------------
    // Document loading
    // ---------------------------------------------------------------------------

    /**
     * Load the FAISS-equivalent flat index for [doc] from disk.
     * Must be called before [ask]; safe to call multiple times (switches context).
     */
    fun loadDocument(doc: DocumentEntity) {
        viewModelScope.launch(Dispatchers.IO) {
            _isWarmingUp.value = true
            _lastThinkingDurationMs.value = null
            try {
                val file = File(getApplication<Application>().filesDir, "${doc.id}.idx")
                if (!file.exists()) { _isWarmingUp.value = false; return@launch }
                val app = getApplication<Application>()
                val idx = EdgeTutorPerf.trace("index_load", "doc_id" to doc.id) {
                    FlatIndex().also { it.load(file) }
                }
                index = idx
                activeDoc = doc
                _activeDocumentId.value = doc.id
                _isLikelyScanned.value = doc.isLikelyScanned
                _messages.value = emptyList()

                // Close any embedder from a previously loaded document.
                embedder?.close()
                EdgeTutorPerf.snapshot(app, "embed_warmup_before", "doc_id" to doc.id)
                val emb = EdgeTutorPerf.trace("embed_warmup", "doc_id" to doc.id) {
                    Embedder(app).also { it.warmUp() }
                }
                embedder = emb
                EdgeTutorPerf.snapshot(app, "embed_warmup_after", "doc_id" to doc.id)

                EdgeTutorPerf.snapshot(app, "llm_warmup_before", "doc_id" to doc.id)
                EdgeTutorPerf.traceSuspend("llm_warmup", "doc_id" to doc.id) {
                    llm.warmUp()
                }
                EdgeTutorPerf.snapshot(app, "llm_warmup_after", "doc_id" to doc.id)
                _isWarmingUp.value = false
            } catch (e: Exception) {
                _isWarmingUp.value = false
                _errorMessage.value = "Failed to load model: ${e.message}"
            }
        }
    }

    fun clearError() { _errorMessage.value = null }

    // ---------------------------------------------------------------------------
    // Querying
    // ---------------------------------------------------------------------------

    fun ask(question: String) {
        val idx = index ?: return
        if (_isWarmingUp.value) return

        viewModelScope.launch(Dispatchers.IO) {
            val thinkingStartMs = SystemClock.elapsedRealtime()
            val queryStartNs = System.nanoTime()
            val docId = activeDoc?.id ?: -1L
            var shouldPersistThinkingDuration = false
            _lastThinkingDurationMs.value = null
            _isThinking.value = true
            _messages.value  += ChatMessage(Role.USER, question)
            try {
                val app = getApplication<Application>()
                // 1. Embed question using the cached ORT session.
                //    Normal-memory devices keep it hot across turns; low-memory
                //    devices still release it around generation to reduce pressure.
                val emb = embedder ?: Embedder(app).also { it.warmUp(); embedder = it }
                val releaseEmbedder = shouldReleaseEmbedderForGeneration()
                EdgeTutorPerf.log(
                    "query_memory_policy",
                    "doc_id" to docId,
                    "avail_mem_mb" to getAvailableMemoryMB(),
                    "low_mem_threshold_mb" to LOW_MEM_THRESHOLD_MB,
                    "release_embedder" to releaseEmbedder,
                )
                EdgeTutorPerf.snapshot(app, "query_embed_before", "doc_id" to docId)
                val queryEmbedStartNs = System.nanoTime()
                val qVec = EdgeTutorPerf.trace("query_embed", "doc_id" to docId) {
                    emb.embed(question, isQuery = true)
                }
                val queryEmbedMs = EdgeTutorPerf.elapsedMs(queryEmbedStartNs)
                EdgeTutorPerf.snapshot(app, "query_embed_after", "doc_id" to docId)
                EdgeTutorPerf.log("query_stage_timing", "doc_id" to docId, "query_embed_ms" to queryEmbedMs)

                // 2. Retrieve top-3 chunks with similarity scores
                val retrievalStartNs = System.nanoTime()
                val searchResults = EdgeTutorPerf.trace("retrieval_search", "doc_id" to docId, "k" to 3) {
                    idx.searchWithScores(qVec, k = 3)
                }
                val retrievalSearchMs = EdgeTutorPerf.elapsedMs(retrievalStartNs)
                EdgeTutorPerf.log("query_stage_timing", "doc_id" to docId, "retrieval_search_ms" to retrievalSearchMs)
                val topChunks = searchResults.map { (entry, _) -> entry }
                val topK = topChunks.size

                // 3. Out-of-scope gate — reject before hitting the LLM
                val inScope = isInScope(question, searchResults)
                if (!inScope.allowed) {
                    EdgeTutorPerf.log(
                        "query_out_of_scope",
                        "doc_id" to docId,
                        "max_sim" to inScope.maxSim,
                        "lexical_overlap" to inScope.lexicalOverlap,
                        "required_overlap" to inScope.requiredOverlap,
                    )
                    _messages.value += ChatMessage(
                        role = Role.ASSISTANT,
                        text = "This doesn't appear to be covered in the loaded document.",
                    )
                    shouldPersistThinkingDuration = true
                    return@launch  // finally still runs and resets _isThinking
                }

                // 4. Build prompt (matches Python src/rag/query.py system prompt)
                val promptBuildStartNs = System.nanoTime()
                val rawContextText = topChunks.joinToString("\n---\n") { it.text }
                val sanitizedContext = PromptSanitizer.sanitize(rawContextText)
                val sanitizedQuestion = PromptSanitizer.sanitize(question)
                val contextText = sanitizedContext.value
                val prompt = buildPrompt(
                    context = contextText,
                    question = sanitizedQuestion.value,
                )
                val sanitizedPrompt = PromptSanitizer.sanitize(prompt)
                val promptBuildMs = EdgeTutorPerf.elapsedMs(promptBuildStartNs)
                EdgeTutorPerf.log("query_stage_timing", "doc_id" to docId, "prompt_build_ms" to promptBuildMs)
                EdgeTutorPerf.log(
                    "prompt_sanitization",
                    "doc_id" to docId,
                    "raw_context_chars" to rawContextText.length,
                    "sanitized_context_chars" to sanitizedContext.value.length,
                    "raw_question_chars" to question.length,
                    "sanitized_question_chars" to sanitizedQuestion.value.length,
                    "raw_prompt_chars" to prompt.length,
                    "sanitized_prompt_chars" to sanitizedPrompt.value.length,
                    "had_non_ascii" to (
                        sanitizedContext.hadNonAscii ||
                            sanitizedQuestion.hadNonAscii ||
                            sanitizedPrompt.hadNonAscii
                        ),
                    "replacement_count" to (
                        sanitizedContext.replacementCount +
                            sanitizedQuestion.replacementCount +
                            sanitizedPrompt.replacementCount
                        ),
                    "dropped_count" to (
                        sanitizedContext.droppedCount +
                            sanitizedQuestion.droppedCount +
                            sanitizedPrompt.droppedCount
                        ),
                )
                EdgeTutorPerf.log(
                    "prompt_metrics",
                    "doc_id" to docId,
                    "prompt_chars" to sanitizedPrompt.value.length,
                    "estimated_prompt_tokens" to estimatePromptTokens(sanitizedPrompt.value),
                    "top_k" to topK,
                    "chunk_char_counts" to topChunks.joinToString(",") { it.text.length.toString() },
                    "final_context_chars" to contextText.length,
                )

                // 5. Add a placeholder ASSISTANT message; stream tokens into it.
                //    Tokens are buffered and flushed to StateFlow every 50 ms to
                //    avoid a full list copy + String allocation on every token.
                _messages.value += ChatMessage(
                    role    = Role.ASSISTANT,
                    text    = "",
                    sources = topChunks.map { it.text.take(120) + "…" },
                )

                val tokenBuf = StringBuilder()
                var uiVisibleTtftLogged = false
                fun flushAssistantChunk(chunk: String, source: String) {
                    if (chunk.isEmpty()) return
                    val list = _messages.value.toMutableList()
                    val last = list.last()
                    val updatedText = last.text + chunk
                    list[list.lastIndex] = last.copy(text = updatedText)
                    _messages.value = list
                    if (!uiVisibleTtftLogged && updatedText.isNotBlank()) {
                        uiVisibleTtftLogged = true
                        EdgeTutorPerf.log(
                            "query_stage_timing",
                            "doc_id" to docId,
                            "ui_visible_ttft_ms" to EdgeTutorPerf.elapsedMs(queryStartNs),
                            "source" to source,
                        )
                    }
                }
                var flushJob: Job? = viewModelScope.launch(Dispatchers.Main) {
                    while (isActive) {
                        delay(50)
                        val chunk = synchronized(tokenBuf) {
                            if (tokenBuf.isEmpty()) null
                            else { val s = tokenBuf.toString(); tokenBuf.clear(); s }
                        }
                        if (chunk != null) {
                            flushAssistantChunk(chunk, source = "buffered_flush")
                        }
                    }
                }

                if (releaseEmbedder) {
                    EdgeTutorPerf.log("embedder_release_before_generation", "doc_id" to docId)
                    emb.close()
                    embedder = null
                }

                try {
                    llm.generate(sanitizedPrompt.value) { token ->
                        synchronized(tokenBuf) { tokenBuf.append(token) }
                    }
                } finally {
                    flushJob?.cancel()
                    flushJob = null
                    // Final flush — emit any tokens that arrived in the last <50 ms window.
                    val remaining = synchronized(tokenBuf) { tokenBuf.toString() }
                    if (remaining.isNotEmpty()) {
                        flushAssistantChunk(remaining, source = "final_flush")
                    }
                    if (releaseEmbedder) {
                        // On low-memory devices, recreate lazily on the next query
                        // instead of immediately paying the session startup cost here.
                        embedder = null
                    }
                }
                EdgeTutorPerf.log(
                    "query_stage_timing",
                    "doc_id" to docId,
                    "total_answer_ms" to EdgeTutorPerf.elapsedMs(queryStartNs),
                )

                val lastAssistantMessage = _messages.value.lastOrNull()
                shouldPersistThinkingDuration =
                    lastAssistantMessage != null &&
                    lastAssistantMessage.role == Role.ASSISTANT &&
                    lastAssistantMessage.text.isNotBlank()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _errorMessage.value = "Query failed: ${e.message}"
            } finally {
                val thinkingDurationMs =
                    if (shouldPersistThinkingDuration) SystemClock.elapsedRealtime() - thinkingStartMs
                    else null
                _lastThinkingDurationMs.value = thinkingDurationMs
                _isThinking.value = false
            }
        }
    }

    fun resetHistory() {
        _messages.value = emptyList()
        _lastThinkingDurationMs.value = null
    }

    // ---------------------------------------------------------------------------
    // Out-of-scope gate - mirrors Python src/rag/query.py
    // ---------------------------------------------------------------------------

    /**
     * Returns true if the question appears to be covered by the retrieved chunks.
     *
     * Two conditions must both pass:
     *  1. At least one chunk has cosine similarity >= [MIN_COSINE_SIM] (equivalent
     *     to L2 distance <= 1.4 on normalized vectors: cos = 1 - dist²/2).
     *  2. The question shares >= [MIN_LEXICAL_OVERLAP] content words with at least
     *     one chunk (stopwords excluded).
     */
    private fun isInScope(question: String, results: List<Pair<FlatIndex.Entry, Float>>): ScopeCheckResult {
        val maxSim = results.maxOfOrNull { (_, sim) -> sim } ?: return ScopeCheckResult(
            allowed = false,
            maxSim = 0f,
            lexicalOverlap = 0,
            requiredOverlap = MIN_LEXICAL_OVERLAP,
        )
        if (maxSim < MIN_COSINE_SIM) {
            return ScopeCheckResult(
                allowed = false,
                maxSim = maxSim,
                lexicalOverlap = 0,
                requiredOverlap = MIN_LEXICAL_OVERLAP,
            )
        }
        val overlap = lexicalOverlap(question, results.map { (e, _) -> e.text })
        return ScopeCheckResult(
            allowed = overlap.bestOverlap >= overlap.requiredOverlap,
            maxSim = maxSim,
            lexicalOverlap = overlap.bestOverlap,
            requiredOverlap = overlap.requiredOverlap,
        )
    }

    private fun lexicalOverlap(question: String, chunks: List<String>): LexicalOverlapResult {
        val qTokens = normalizedTokens(question)
        val required = minOf(MIN_LEXICAL_OVERLAP, maxOf(1, qTokens.size))
        val bestOverlap = chunks.maxOfOrNull { chunk ->
            val chunkTokens = normalizedTokens(chunk)
            qTokens.count { it in chunkTokens }
        } ?: 0
        return LexicalOverlapResult(
            bestOverlap = bestOverlap,
            requiredOverlap = required,
        )
    }

    private fun normalizedTokens(text: String): Set<String> =
        text.lowercase()
            .split(Regex("[^a-z]+"))
            .asSequence()
            .map(::normalizeToken)
            .filter { it.isNotEmpty() && it !in STOPWORDS }
            .toSet()

    private fun normalizeToken(token: String): String {
        var normalized = token
        val suffixes = listOf("ation", "ition", "ment", "ingly", "edly", "ing", "ed", "es", "s")
        for (suffix in suffixes) {
            if (normalized.length > suffix.length + 2 && normalized.endsWith(suffix)) {
                normalized = normalized.removeSuffix(suffix)
                break
            }
        }
        if (normalized.endsWith("i") && normalized.length > 3) {
            normalized = normalized.dropLast(1) + "y"
        }
        return normalized
    }

    // ---------------------------------------------------------------------------
    // Prompt template — mirrors EdgeTutor_MVP_Unified.md §8.1
    // ---------------------------------------------------------------------------

    private fun buildPrompt(context: String, question: String): String = """
Context passages from the document:

$context

Answer using ONLY the passages above.
Question: $question
""".trimIndent()

    // ---------------------------------------------------------------------------

    private fun estimatePromptTokens(prompt: String): Int =
        ceil(prompt.length / 4.0).toInt()

    private data class ScopeCheckResult(
        val allowed: Boolean,
        val maxSim: Float,
        val lexicalOverlap: Int,
        val requiredOverlap: Int,
    )

    private data class LexicalOverlapResult(
        val bestOverlap: Int,
        val requiredOverlap: Int,
    )

    override fun onCleared() {
        super.onCleared()
        llm.close()
        embedder?.close()
        embedder = null
    }
}
