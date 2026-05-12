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
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
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

sealed class ThinkingUiState {
    object Idle : ThinkingUiState()
    object Active : ThinkingUiState()
    data class Done(val durationMs: Long) : ThinkingUiState()
}

class ChatViewModel(app: Application) : AndroidViewModel(app) {

    private val llm: LlmEngine by lazy { LlamaEngine(app) }

    init {
        // Start copying the model asset to internal storage immediately so the file
        // is ready on disk by the time the user selects a document and warmUp() fires.
        // On subsequent launches the file already exists and this completes instantly.
        viewModelScope.launch(Dispatchers.IO) {
            try {
                llm.copyModelIfNeeded()
                // Load native weights immediately after copy so the expensive
                // LlamaBridge.initGenerateModel() is hidden behind the document-picker
                // screen rather than paid at document selection time.
                llm.initNativeModel()
            }
            catch (e: Exception) { _errorMessage.value = "Model unavailable: ${e.message}" }
        }
    }

    private val _messages    = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages

    private val _thinkingUiState = MutableStateFlow<ThinkingUiState>(ThinkingUiState.Idle)
    val thinkingUiState: StateFlow<ThinkingUiState> = _thinkingUiState.asStateFlow()
    /** Derived for canSend / inputEnabled logic — no change needed in UI. */
    val isThinking: StateFlow<Boolean> = _thinkingUiState
        .map { it is ThinkingUiState.Active }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

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
        private const val RETRIEVAL_CANDIDATE_K = 5
        private const val MAX_KEPT_CONTEXT_CHUNKS = 2
        private const val MAX_CONTEXT_CHARS_PER_CHUNK = 800
        private const val MAX_FOLLOWUP_CONTEXT_CHARS = 180
        private const val MAX_ANSWER_CONTEXT_CHARS = 250
        private const val STRONG_SEMANTIC_MATCH_SIM = 0.70f

        private val STOPWORDS = setOf(
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "i", "you", "he", "she",
            "it", "we", "they", "what", "how", "why", "when", "where", "who",
            "which", "this", "that", "these", "those", "of", "in", "on", "at",
            "to", "for", "with", "by", "from", "about", "into", "than", "or",
            "and", "but", "if", "not", "no", "so",
        )

        private val ACADEMIC_TERMS = setOf(
            "academic", "algebra", "analyze", "answer", "biology", "calculate",
            "calculus", "chemistry", "compare", "concept", "define", "derivative",
            "differentiate", "differential", "equation", "example", "explain",
            "factor", "formula", "function", "geometry", "graph", "history",
            "homework", "integral", "interpret", "lesson", "limit", "math",
            "physics", "practice", "problem", "proof", "reading", "science",
            "slope", "solve", "study", "summarize", "teach", "theorem", "tutor",
            "understand", "word",
        )

        private val FOLLOWUP_TERMS = setOf(
            "again", "also", "another", "back", "example", "it", "more", "that",
            "this", "those", "time",
        )

        private val WORKED_EXAMPLE_TERMS = setOf(
            "calculate", "differentiate", "example", "integrate", "problem", "show",
            "solve", "step", "steps", "work",
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
            _thinkingUiState.value = ThinkingUiState.Idle
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
            _thinkingUiState.value = ThinkingUiState.Active
            val priorMessages = _messages.value
            _messages.value  += ChatMessage(Role.USER, question)
            try {
                val app = getApplication<Application>()
                val retrievalQuestion = buildRetrievalQuestion(question, priorMessages)
                // Only carry conversation context into the prompt for genuine follow-up
                // questions. Fresh questions don't need it, and the extra tokens push
                // prefill time past the 30s TTFT exit gate (O(n²) attention cost).
                val conversationContext = if (isFollowUpQuestion(question)) {
                    buildConversationContext(priorMessages)
                } else {
                    ""
                }
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
                    emb.embed(retrievalQuestion, isQuery = true)
                }
                val queryEmbedMs = EdgeTutorPerf.elapsedMs(queryEmbedStartNs)
                EdgeTutorPerf.snapshot(app, "query_embed_after", "doc_id" to docId)
                EdgeTutorPerf.log("query_stage_timing", "doc_id" to docId, "query_embed_ms" to queryEmbedMs)
                EdgeTutorPerf.log(
                    "query_rewrite",
                    "doc_id" to docId,
                    "used_followup_context" to (retrievalQuestion != question),
                    "retrieval_query_chars" to retrievalQuestion.length,
                    "conversation_context_chars" to conversationContext.length,
                )

                // 2. Retrieve candidate chunks with similarity scores, then keep a small prompt budget.
                val retrievalStartNs = System.nanoTime()
                val searchResults = EdgeTutorPerf.trace("retrieval_search", "doc_id" to docId, "k" to RETRIEVAL_CANDIDATE_K) {
                    idx.searchWithScores(qVec, k = RETRIEVAL_CANDIDATE_K)
                }
                val retrievalSearchMs = EdgeTutorPerf.elapsedMs(retrievalStartNs)
                EdgeTutorPerf.log("query_stage_timing", "doc_id" to docId, "retrieval_search_ms" to retrievalSearchMs)

                // 3. Route the query and build the shortest prompt that can answer it.
                val promptBuildStartNs = System.nanoTime()
                val contextSelection = selectContext(searchResults)
                val routeDecision = routeQuery(question, contextSelection)
                val sanitizedQuestion = PromptSanitizer.sanitize(question)
                val rawContextText = contextSelection.keptChunks.joinToString("\n\n") { (_, chunk) -> chunk }
                val sanitizedContext = PromptSanitizer.sanitize(rawContextText)
                val prompt = when (routeDecision.route) {
                    QueryRoute.GROUNDED -> buildGroundedPrompt(
                        passages = sanitizedContext.value,
                        conversationContext = PromptSanitizer.sanitize(conversationContext).value,
                        question = sanitizedQuestion.value,
                        wantsWorkedExample = wantsWorkedExample(question),
                    )
                    QueryRoute.GENERAL_REASONING -> buildGeneralReasoningPrompt(
                        conversationContext = PromptSanitizer.sanitize(conversationContext).value,
                        question = sanitizedQuestion.value,
                        wantsWorkedExample = wantsWorkedExample(question),
                    )
                    QueryRoute.UNRELATED -> ""
                }
                val sanitizedPrompt = PromptSanitizer.sanitize(prompt)
                val promptBuildMs = EdgeTutorPerf.elapsedMs(promptBuildStartNs)
                EdgeTutorPerf.log("query_stage_timing", "doc_id" to docId, "prompt_build_ms" to promptBuildMs)
                EdgeTutorPerf.log(
                    "query_route",
                    "doc_id" to docId,
                    "query_route" to routeDecision.route.name,
                    "route_reason" to routeDecision.reason,
                    "max_sim" to contextSelection.maxSimilarity,
                    "lexical_overlap" to routeDecision.lexicalOverlap,
                    "required_overlap" to routeDecision.requiredOverlap,
                )
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
                    "retrieved_k" to contextSelection.retrievedCount,
                    "kept_k" to contextSelection.keptChunks.size,
                    "max_sim" to contextSelection.maxSimilarity,
                    "kept_sim_scores" to formatScores(contextSelection.keptScores),
                    "dropped_sim_scores" to formatScores(contextSelection.droppedScores),
                    "context_char_cap" to contextSelection.contextCharCap,
                    "final_context_chars" to contextSelection.finalContextChars,
                    "prompt_chars" to sanitizedPrompt.value.length,
                    "estimated_prompt_tokens" to estimatePromptTokens(sanitizedPrompt.value),
                    "query_route" to routeDecision.route.name,
                )
                if (routeDecision.route == QueryRoute.UNRELATED) {
                    _messages.value += ChatMessage(
                        role = Role.ASSISTANT,
                        text = "This doesn't appear to be covered in the loaded document.",
                    )
                    shouldPersistThinkingDuration = true
                    EdgeTutorPerf.log(
                        "query_stage_timing",
                        "doc_id" to docId,
                        "total_answer_ms" to EdgeTutorPerf.elapsedMs(queryStartNs),
                    )
                    return@launch
                }

                // 4. Add a placeholder ASSISTANT message; stream tokens into it.
                //    The first visible token is flushed immediately for perceived TTFT;
                //    later tokens are buffered every 50 ms to avoid Compose churn.
                _messages.value += ChatMessage(
                    role    = Role.ASSISTANT,
                    text    = "",
                    sources = if (routeDecision.route == QueryRoute.GROUNDED) {
                        contextSelection.keptChunks.map { (_, chunk) -> chunk.take(120) + "." }
                    } else {
                        emptyList()
                    },
                )

                val tokenBuf = StringBuilder()
                var firstTokenFlushed = false
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
                    // Memory Handover: explicit GC and small delay to let the OS
                    // reclaim the ~23MB ONNX session memory before the LLM demands 300MB+.
                    System.gc()
                    delay(200)
                }

                try {
                    llm.generate(sanitizedPrompt.value) { token ->
                        val immediateChunk = synchronized(tokenBuf) {
                            if (!firstTokenFlushed && token.isNotBlank()) {
                                firstTokenFlushed = true
                                val chunk = tokenBuf.toString() + token
                                tokenBuf.clear()
                                chunk
                            } else {
                                tokenBuf.append(token)
                                null
                            }
                        }
                        if (immediateChunk != null) {
                            flushAssistantChunk(immediateChunk, source = "first_token_flush")
                        }
                    }
                } finally {
                    flushJob?.cancel()
                    flushJob = null
                    // Final flush - emit any tokens that arrived in the last <50 ms window.
                    val remaining = synchronized(tokenBuf) {
                        val chunk = tokenBuf.toString()
                        tokenBuf.clear()
                        chunk
                    }
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
                _thinkingUiState.value =
                    if (shouldPersistThinkingDuration)
                        ThinkingUiState.Done(SystemClock.elapsedRealtime() - thinkingStartMs)
                    else
                        ThinkingUiState.Idle
            }
        }
    }

    fun resetHistory() {
        _messages.value = emptyList()
        _thinkingUiState.value = ThinkingUiState.Idle
    }

    // ---------------------------------------------------------------------------
    // Query routing and context budgeting
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
    private fun selectContext(results: List<Pair<FlatIndex.Entry, Float>>): ContextSelection {
        val sortedResults = results.sortedByDescending { (_, sim) -> sim }
        val keptCandidates = sortedResults
            .filter { (_, sim) -> sim >= MIN_COSINE_SIM }
            .take(MAX_KEPT_CONTEXT_CHUNKS)

        val keptEntries = keptCandidates.mapIndexed { index, (entry, _) ->
            entry to "${index + 1}. ${entry.text.take(MAX_CONTEXT_CHARS_PER_CHUNK)}"
        }
        val keptIds = keptCandidates.map { (entry, _) -> entry.id }.toSet()
        val keptScores = keptCandidates.map { (_, sim) -> sim }
        val droppedScores = sortedResults
            .filterIndexed { index, (entry, sim) ->
                sim < MIN_COSINE_SIM || index >= MAX_KEPT_CONTEXT_CHUNKS || entry.id !in keptIds
            }
            .map { (_, sim) -> sim }

        return ContextSelection(
            retrievedCount = results.size,
            keptChunks = keptEntries,
            keptScores = keptScores,
            droppedScores = droppedScores,
            maxSimilarity = sortedResults.firstOrNull()?.second ?: 0f,
            contextCharCap = MAX_CONTEXT_CHARS_PER_CHUNK,
            finalContextChars = keptEntries.sumOf { (_, chunk) -> chunk.length },
        )
    }

    private fun routeQuery(question: String, contextSelection: ContextSelection): RouteDecision {
        if (contextSelection.maxSimilarity < MIN_COSINE_SIM) {
            return weakDocumentRoute(
                question = question,
                reason = "similarity_failure",
                lexicalOverlap = 0,
                requiredOverlap = MIN_LEXICAL_OVERLAP,
            )
        }

        val overlap = lexicalOverlap(question, contextSelection.keptChunks.map { (_, chunk) -> chunk })
        if (
            overlap.bestOverlap >= overlap.requiredOverlap ||
            contextSelection.maxSimilarity >= STRONG_SEMANTIC_MATCH_SIM
        ) {
            return RouteDecision(
                route = QueryRoute.GROUNDED,
                reason = if (overlap.bestOverlap >= overlap.requiredOverlap) {
                    "grounded"
                } else {
                    "strong_semantic_match"
                },
                lexicalOverlap = overlap.bestOverlap,
                requiredOverlap = overlap.requiredOverlap,
            )
        }

        return weakDocumentRoute(
            question = question,
            reason = "lexical_overlap_failure",
            lexicalOverlap = overlap.bestOverlap,
            requiredOverlap = overlap.requiredOverlap,
        )
    }

    private fun weakDocumentRoute(
        question: String,
        reason: String,
        lexicalOverlap: Int,
        requiredOverlap: Int,
    ): RouteDecision {
        val academicFallback = isAcademicQuestion(question)
        return RouteDecision(
            route = if (academicFallback) QueryRoute.GENERAL_REASONING else QueryRoute.UNRELATED,
            reason = if (academicFallback) "academic_fallback_after_$reason" else "unrelated_refusal_after_$reason",
            lexicalOverlap = lexicalOverlap,
            requiredOverlap = requiredOverlap,
        )
    }

    private fun lexicalOverlap(question: String, chunks: List<String>): LexicalOverlapResult {
        val qTokens = normalizedTokens(question)
        val required = when {
            qTokens.size >= 4 -> 2
            else -> minOf(MIN_LEXICAL_OVERLAP, maxOf(1, qTokens.size))
        }
        val bestOverlap = chunks.maxOfOrNull { chunk ->
            val chunkTokens = normalizedTokens(chunk)
            qTokens.count { it in chunkTokens }
        } ?: 0
        return LexicalOverlapResult(
            bestOverlap = bestOverlap,
            requiredOverlap = required,
        )
    }

    private fun isAcademicQuestion(question: String): Boolean {
        val tokens = normalizedTokens(question)
        if (tokens.any { it in ACADEMIC_TERMS }) return true
        return question.contains("?") && tokens.any { it.length >= 5 && it in ACADEMIC_TERMS }
    }

    private fun buildRetrievalQuestion(question: String, priorMessages: List<ChatMessage>): String {
        if (!isFollowUpQuestion(question)) return question
        val previousUserQuestion = priorMessages
            .lastOrNull { it.role == Role.USER }
            ?.text
            ?.take(MAX_FOLLOWUP_CONTEXT_CHARS)
            ?: return question
        return "$previousUserQuestion $question"
    }

    private fun buildConversationContext(priorMessages: List<ChatMessage>): String {
        val lastUser = priorMessages.lastOrNull { it.role == Role.USER } ?: return ""
        val lastAssistant = priorMessages.lastOrNull { it.role == Role.ASSISTANT }
        val sb = StringBuilder("Previous question: ${lastUser.text.take(MAX_FOLLOWUP_CONTEXT_CHARS)}")
        if (lastAssistant != null && lastAssistant.text.isNotBlank()) {
            sb.append("\nPrevious answer: ${lastAssistant.text.take(MAX_ANSWER_CONTEXT_CHARS)}")
        }
        return sb.toString()
    }

    private fun isFollowUpQuestion(question: String): Boolean {
        val tokens = normalizedTokens(question)
        val raw = question.lowercase()
        return tokens.any { it in FOLLOWUP_TERMS } ||
            raw.contains("this") ||
            raw.contains("that") ||
            raw.contains(" it ") ||
            raw.startsWith("tell me more") ||
            raw.startsWith("explain more")
    }

    private fun wantsWorkedExample(question: String): Boolean {
        val tokens = normalizedTokens(question)
        return tokens.any { it in WORKED_EXAMPLE_TERMS }
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

    private fun buildGroundedPrompt(
        passages: String,
        conversationContext: String,
        question: String,
        wantsWorkedExample: Boolean,
    ): String = """
$passages

$conversationContext

${answerInstruction(wantsWorkedExample)}
Question: $question
""".trimIndent()

    private fun buildGeneralReasoningPrompt(
        conversationContext: String,
        question: String,
        wantsWorkedExample: Boolean,
    ): String = """
The loaded document did not provide strong support. Answer as a concise tutor.
$conversationContext
${answerInstruction(wantsWorkedExample)}
Question: $question
""".trimIndent()

    private fun answerInstruction(wantsWorkedExample: Boolean): String =
        if (wantsWorkedExample) {
            "Give a small worked example. Show the derivative, integrate it back, and check the result. Do not reply with only a constant or equation."
        } else {
            "Answer using these passages. Be direct and explain the relationship, not a loose scenario."
        }

    // ---------------------------------------------------------------------------

    private fun estimatePromptTokens(prompt: String): Int =
        ceil(prompt.length / 4.0).toInt()

    private fun formatScores(scores: List<Float>): String =
        scores.joinToString(",") { "%.4f".format(it) }

    private enum class QueryRoute { GROUNDED, GENERAL_REASONING, UNRELATED }

    private data class ContextSelection(
        val retrievedCount: Int,
        val keptChunks: List<Pair<FlatIndex.Entry, String>>,
        val keptScores: List<Float>,
        val droppedScores: List<Float>,
        val maxSimilarity: Float,
        val contextCharCap: Int,
        val finalContextChars: Int,
    )

    private data class RouteDecision(
        val route: QueryRoute,
        val reason: String,
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
