package com.edgetutor.mnn.viewmodel

import android.app.ActivityManager
import android.app.Application
import android.net.Uri
import android.os.SystemClock
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.edgetutor.mnn.data.db.DocumentEntity
import com.edgetutor.mnn.data.db.AppDatabase
import com.edgetutor.mnn.data.db.ChatSessionEntity
import com.edgetutor.mnn.data.db.MessageEntity
import com.edgetutor.mnn.data.db.SessionListItem
import com.edgetutor.mnn.ingestion.Embedder
import com.edgetutor.mnn.llm.LlmEngine
import com.edgetutor.mnn.llm.MnnEngine
import com.edgetutor.mnn.llm.MnnModelManager
import com.edgetutor.mnn.llm.ModelReadinessKind
import com.edgetutor.mnn.llm.ModelReadinessState
import com.edgetutor.mnn.llm.PromptSanitizer
import com.edgetutor.mnn.perf.EdgeTutorPerf
import com.edgetutor.mnn.perf.EdgeTutorValidationSuite
import com.edgetutor.mnn.perf.ValidationResult
import com.edgetutor.mnn.store.FlatIndex
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
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
import com.google.gson.Gson
import kotlin.math.ceil

data class ChatMessage(
    val role: Role,
    val text: String,
    /** Short excerpts of the retrieved chunks — shown as source attribution. */
    val sources: List<String> = emptyList(),
    val id: Long = System.nanoTime(),
    val thinking: String? = null,
    val imagePath: String? = null,
    val metricsText: String? = null,
    val completionState: String = "complete",
)

data class QueryExecutionResult(
    val policyId: String,
    val question: String,
    val answer: String,
    val sources: List<String>,
    val promptChars: Int,
    val promptTokens: Long,
    val prefillUs: Long,
    val decodeUs: Long,
    val visibleTtftMs: Long,
    val totalMs: Long,
    val availableMemoryMb: Long,
    val route: QueryRoute,
    val routeReason: String,
    val routeMarkerValid: Boolean,
    val maxSimilarity: Float,
    val secondSimilarity: Float,
    val meanTop5Similarity: Float,
)

enum class Role { USER, ASSISTANT }

sealed class ThinkingUiState {
    object Idle   : ThinkingUiState()
    object Active : ThinkingUiState()
    data class Done(val durationMs: Long) : ThinkingUiState()
}

/**
 * Product Chat ViewModel for the MNN-LLM application.
 *
 * Owns retrieval, model-routed prompting, source attribution, performance
 * logging, and streamed UI updates.
 */
class ChatViewModel(app: Application) : AndroidViewModel(app) {
    private val db = AppDatabase.get(app)
    private val gson = Gson()

    private val mnnEngine: MnnEngine by lazy { MnnEngine(app) }
    // ► Swap MnnEngine for any other LlmEngine implementation here.
    private val llm: LlmEngine by lazy { mnnEngine }

    private val _modelReadiness = MutableStateFlow(MnnModelManager.validate(app))
    val modelReadiness: StateFlow<ModelReadinessState> = _modelReadiness.asStateFlow()

    init {
        // Start native init immediately so it is hidden behind the document-picker.
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val state = llm.copyModelIfNeeded { progress ->
                    _modelReadiness.value = progress
                }
                _modelReadiness.value = state
                if (state.isReady) {
                    llm.initNativeModel()
                } else {
                    _errorMessage.value =
                        "Bundled model installation failed: ${state.message ?: "unknown error"}. " +
                            "You can import the model folder manually in Settings."
                }
            } catch (e: Exception) {
                _errorMessage.value = "Model unavailable: ${e.message}"
            }
        }
    }

    private val _messages    = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages
    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()
    private val _pendingImagePath = MutableStateFlow<String?>(null)
    val pendingImagePath: StateFlow<String?> = _pendingImagePath.asStateFlow()
    private var generationJob: Job? = null

    private val _thinkingUiState = MutableStateFlow<ThinkingUiState>(ThinkingUiState.Idle)
    val thinkingUiState: StateFlow<ThinkingUiState> = _thinkingUiState.asStateFlow()
    val isThinking: StateFlow<Boolean> = _thinkingUiState
        .map { it is ThinkingUiState.Active }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    private val _isWarmingUp = MutableStateFlow(false)
    val isWarmingUp: StateFlow<Boolean> = _isWarmingUp

    /** User-controlled Thinking pill: when on, the model emits a <think> block. */
    private val _thinkingEnabled = MutableStateFlow(false)
    val thinkingEnabled: StateFlow<Boolean> = _thinkingEnabled.asStateFlow()

    fun setThinkingEnabled(enabled: Boolean) { _thinkingEnabled.value = enabled }
    fun toggleThinking() { _thinkingEnabled.value = !_thinkingEnabled.value }

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage

    private val _isLikelyScanned = MutableStateFlow(false)
    val isLikelyScanned: StateFlow<Boolean> = _isLikelyScanned

    private val _activeDocumentId = MutableStateFlow<Long?>(null)
    val activeDocumentId: StateFlow<Long?> = _activeDocumentId.asStateFlow()

    private val _activeSessionId = MutableStateFlow<Long?>(null)
    val activeSessionId: StateFlow<Long?> = _activeSessionId.asStateFlow()

    /** All chat sessions across textbooks, newest first — backs the history drawer. */
    val sessions: StateFlow<List<SessionListItem>> =
        db.chatSessionDao().observeAll()
            .stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())

    private val _validationStatus = MutableStateFlow<String?>(null)
    val validationStatus: StateFlow<String?> = _validationStatus.asStateFlow()
    private var lastQueryResult: QueryExecutionResult? = null

    companion object {
        private const val LOW_MEM_THRESHOLD_MB        = 200L
        private const val EMBEDDER_CLOSE_DELAY_MS     = 30_000L
        private const val RETRIEVAL_CANDIDATE_K       = 5
        private const val MAX_FOLLOWUP_CONTEXT_CHARS  = 180
        private const val MAX_ANSWER_CONTEXT_CHARS    = 250

        private val STOPWORDS = setOf(
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "i", "you", "he", "she",
            "it", "we", "they", "what", "how", "why", "when", "where", "who",
            "which", "this", "that", "these", "those", "of", "in", "on", "at",
            "to", "for", "with", "by", "from", "about", "into", "than", "or",
            "and", "but", "if", "not", "no", "so",
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

    private var index: FlatIndex?   = null
    var activeDoc: DocumentEntity?  = null
        private set
    private var embedder: Embedder? = null
    private var embedderCloseJob: Job? = null
    @Volatile private var promptBudgetPolicy: PromptBudgetPolicy = PromptBudgetPolicy.DEFAULT

    fun setPromptBudgetPolicy(policy: PromptBudgetPolicy) {
        promptBudgetPolicy = policy
    }

    private fun getAvailableMemoryMB(): Long {
        val am = getApplication<Application>().getSystemService(Application.ACTIVITY_SERVICE) as ActivityManager
        val info = ActivityManager.MemoryInfo()
        am.getMemoryInfo(info)
        return info.availMem / (1024 * 1024)
    }

    private fun shouldReleaseEmbedderForGeneration(): Boolean =
        getAvailableMemoryMB() < LOW_MEM_THRESHOLD_MB

    // -------------------------------------------------------------------------
    // Document loading
    // -------------------------------------------------------------------------

    fun loadDocument(doc: DocumentEntity) {
        viewModelScope.launch(Dispatchers.IO) {
            if (!loadIndexFor(doc)) return@launch
            val session = db.chatSessionDao().latestForDocument(doc.id) ?: createSessionFor(doc.id)
            _activeSessionId.value = session.id
            _messages.value = db.messageDao().getForSession(session.id).map(::toChatMessage)
        }
    }

    fun clearDocument() {
        if (_isGenerating.value) return
        viewModelScope.launch(Dispatchers.IO) {
            mnnEngine.reset()
            index = null
            activeDoc = null
            _activeDocumentId.value = null
            _activeSessionId.value = null
            _messages.value = emptyList()
            _thinkingUiState.value = ThinkingUiState.Idle
        }
    }

    /** Loads a document's FAISS index and warms the embedder + LLM. */
    private suspend fun loadIndexFor(doc: DocumentEntity): Boolean {
        _isWarmingUp.value = true
        _thinkingUiState.value = ThinkingUiState.Idle
        return try {
            val app = getApplication<Application>()
            val file = File(app.filesDir, "${doc.id}.idx")
            if (!file.exists()) { _isWarmingUp.value = false; return false }
            val idx = EdgeTutorPerf.trace("index_load", "doc_id" to doc.id) {
                FlatIndex().also { it.load(file) }
            }
            index = idx
            activeDoc = doc
            _activeDocumentId.value = doc.id
            _isLikelyScanned.value = doc.isLikelyScanned

            embedderCloseJob?.cancel()
            embedderCloseJob = null
            embedder?.close()
            embedder = null

            EdgeTutorPerf.snapshot(app, "embed_warmup_before", "doc_id" to doc.id)
            EdgeTutorPerf.snapshot(app, "llm_warmup_before",   "doc_id" to doc.id)

            coroutineScope {
                val embJob = async {
                    EdgeTutorPerf.trace("embed_warmup", "doc_id" to doc.id) {
                        Embedder(app).also { it.warmUp() }
                    }
                }
                val llmJob = async {
                    EdgeTutorPerf.traceSuspend("llm_warmup", "doc_id" to doc.id) {
                        llm.warmUp()
                    }
                }
                embedder = embJob.await()
                llmJob.await()
            }

            EdgeTutorPerf.snapshot(app, "embed_warmup_after", "doc_id" to doc.id)
            EdgeTutorPerf.snapshot(app, "llm_warmup_after",   "doc_id" to doc.id)
            _isWarmingUp.value = false
            true
        } catch (e: Exception) {
            _isWarmingUp.value = false
            _errorMessage.value = "Failed to load model: ${e.message}"
            false
        }
    }

    private suspend fun createSessionFor(documentId: Long): ChatSessionEntity {
        val now = System.currentTimeMillis()
        val entity = ChatSessionEntity(documentId = documentId, createdAt = now, updatedAt = now)
        return entity.copy(id = db.chatSessionDao().insert(entity))
    }

    /** Starts a fresh chat on the current textbook. */
    fun newSession() {
        val doc = activeDoc ?: return
        if (_isGenerating.value) return
        viewModelScope.launch(Dispatchers.IO) {
            mnnEngine.reset()
            val session = createSessionFor(doc.id)
            _activeSessionId.value = session.id
            _messages.value = emptyList()
            _thinkingUiState.value = ThinkingUiState.Idle
        }
    }

    /** Opens an existing chat from history, switching textbook index if needed. */
    fun openSession(item: SessionListItem) {
        if (_isGenerating.value) return
        viewModelScope.launch(Dispatchers.IO) {
            if (activeDoc?.id != item.documentId) {
                val doc = db.documentDao().getById(item.documentId) ?: return@launch
                if (!loadIndexFor(doc)) return@launch
            }
            mnnEngine.reset()
            _activeSessionId.value = item.id
            _messages.value = db.messageDao().getForSession(item.id).map(::toChatMessage)
            _thinkingUiState.value = ThinkingUiState.Idle
        }
    }

    fun deleteSession(item: SessionListItem) {
        viewModelScope.launch(Dispatchers.IO) {
            db.chatSessionDao().delete(item.id)
            if (_activeSessionId.value != item.id) return@launch
            mnnEngine.reset()
            val doc = activeDoc
            if (doc == null) {
                _activeSessionId.value = null
                _messages.value = emptyList()
                return@launch
            }
            val next = db.chatSessionDao().latestForDocument(doc.id) ?: createSessionFor(doc.id)
            _activeSessionId.value = next.id
            _messages.value = db.messageDao().getForSession(next.id).map(::toChatMessage)
        }
    }

    fun clearError() { _errorMessage.value = null }
    fun reportError(message: String) { _errorMessage.value = message }
    fun setPendingImage(path: String?) { _pendingImagePath.value = path }
    fun stopGeneration() {
        mnnEngine.cancel()
        generationJob?.cancel()
    }

    fun refreshModelReadiness() {
        _modelReadiness.value = MnnModelManager.validate(getApplication())
    }

    fun importModel(treeUri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            llm.close()
            try {
                val state = MnnModelManager.importFromTreeUri(getApplication(), treeUri) { progress ->
                    _modelReadiness.value = progress
                }
                _modelReadiness.value = state
                if (state.kind == ModelReadinessKind.READY) {
                    llm.copyModelIfNeeded()
                    llm.initNativeModel()
                } else {
                    _errorMessage.value = state.message ?: "Model import did not complete."
                }
            } catch (e: Exception) {
                _modelReadiness.value = ModelReadinessState(
                    kind = ModelReadinessKind.ERROR,
                    message = e.message ?: e.javaClass.simpleName,
                )
                _errorMessage.value = "Model import failed: ${e.message}"
            }
        }
    }

    // -------------------------------------------------------------------------
    // Querying
    // -------------------------------------------------------------------------

    fun ask(question: String): Job? {
        val idx = index ?: return null
        val imagePath = _pendingImagePath.value
        val effectiveQuestion = question.ifBlank {
            if (imagePath != null) "Explain the problem or diagram shown in this image." else return null
        }
        if (_isWarmingUp.value) return null
        if (_isGenerating.value) return null
        if (!_modelReadiness.value.isReady) {
            _errorMessage.value = "Import the MNN model before asking questions."
            return null
        }

        _pendingImagePath.value = null
        val job = viewModelScope.launch(Dispatchers.IO) {
            _isGenerating.value = true
            generationJob = coroutineContext[Job]
            embedderCloseJob?.cancel()
            embedderCloseJob = null

            val thinkingStartMs = SystemClock.elapsedRealtime()
            val queryStartNs    = System.nanoTime()
            val docId           = activeDoc?.id ?: -1L
            val thinkingEnabled = _thinkingEnabled.value
            var shouldPersistThinkingDuration = false
            var activeRoute: QueryRoute? = null
            var activeRouteReason = "not_available"
            _thinkingUiState.value = ThinkingUiState.Active
            val priorMessages = _messages.value
            val sessionId = _activeSessionId.value
                ?: createSessionFor(docId).also { _activeSessionId.value = it.id }.id
            lastQueryResult = null
            val user = ChatMessage(Role.USER, effectiveQuestion, imagePath = imagePath)
            _messages.value += user
            val userId = db.messageDao().insert(
                MessageEntity(
                    documentId = docId,
                    sessionId = sessionId,
                    role = Role.USER.name,
                    text = effectiveQuestion,
                    imagePath = imagePath,
                    thinkingEnabled = thinkingEnabled,
                )
            )
            _messages.value = _messages.value.dropLast(1) + user.copy(id = userId)
            // First message names the session; later ones just bump its recency.
            if (priorMessages.isEmpty()) {
                val title = effectiveQuestion.trim().replace(Regex("\\s+"), " ").take(60)
                db.chatSessionDao().updateMeta(
                    sessionId, title.ifBlank { "New chat" }, System.currentTimeMillis(),
                )
            } else {
                db.chatSessionDao().touch(sessionId, System.currentTimeMillis())
            }
            try {
                val app = getApplication<Application>()
                val retrievalQuestion = buildRetrievalQuestion(effectiveQuestion, priorMessages)
                val conversationContext = if (isFollowUpQuestion(effectiveQuestion)) {
                    buildConversationContext(priorMessages)
                } else {
                    ""
                }

                // 1. Embed the query.
                val emb = embedder ?: Embedder(app).also { it.warmUp(); embedder = it }
                val releaseEmbedder = shouldReleaseEmbedderForGeneration()
                EdgeTutorPerf.log(
                    "query_memory_policy",
                    "doc_id"              to docId,
                    "avail_mem_mb"        to getAvailableMemoryMB(),
                    "low_mem_threshold_mb" to LOW_MEM_THRESHOLD_MB,
                    "release_embedder"    to releaseEmbedder,
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
                    "doc_id"                   to docId,
                    "used_followup_context"    to (retrievalQuestion != question),
                    "retrieval_query_chars"    to retrievalQuestion.length,
                    "conversation_context_chars" to conversationContext.length,
                )

                // 2. Retrieve candidate chunks.
                val retrievalStartNs = System.nanoTime()
                val searchResults = EdgeTutorPerf.trace("retrieval_search", "doc_id" to docId, "k" to RETRIEVAL_CANDIDATE_K) {
                    idx.searchWithScores(qVec, k = RETRIEVAL_CANDIDATE_K)
                }
                val retrievalSearchMs = EdgeTutorPerf.elapsedMs(retrievalStartNs)
                EdgeTutorPerf.log("query_stage_timing", "doc_id" to docId, "retrieval_search_ms" to retrievalSearchMs)

                // 3. Select context and build prompt.
                val promptBuildStartNs = System.nanoTime()
                val contextSelection = selectContext(searchResults, promptBudgetPolicy)
                val sanitizedQuestion = PromptSanitizer.sanitize(effectiveQuestion)
                val rawContextText    = contextSelection.keptChunks.joinToString("\n\n") { (_, chunk) -> chunk }
                val sanitizedContext  = PromptSanitizer.sanitize(rawContextText)
                val prompt = buildRoutedPrompt(
                    passages = sanitizedContext.value,
                    conversationContext = PromptSanitizer.sanitize(conversationContext).value,
                    question = sanitizedQuestion.value,
                    wantsWorkedExample = wantsWorkedExample(effectiveQuestion),
                )
                val sanitizedPrompt = PromptSanitizer.sanitize(prompt)
                val promptBuildMs   = EdgeTutorPerf.elapsedMs(promptBuildStartNs)
                EdgeTutorPerf.log("query_stage_timing", "doc_id" to docId, "prompt_build_ms" to promptBuildMs)
                EdgeTutorPerf.log(
                    "query_route",
                    "doc_id"       to docId,
                    "answer_route" to "PENDING_MODEL_MARKER",
                    "max_sim"      to contextSelection.maxSimilarity,
                    "second_sim" to contextSelection.droppedScores.firstOrNull(),
                    "mean_top5_sim" to (
                        contextSelection.keptScores + contextSelection.droppedScores
                    ).take(5).average(),
                )
                EdgeTutorPerf.log(
                    "prompt_metrics",
                    "doc_id"                    to docId,
                    "prompt_policy"             to promptBudgetPolicy.id,
                    "retrieved_k"               to contextSelection.retrievedCount,
                    "kept_k"                    to contextSelection.keptChunks.size,
                    "max_sim"                   to contextSelection.maxSimilarity,
                    "kept_sim_scores"           to formatScores(contextSelection.keptScores),
                    "dropped_sim_scores"        to formatScores(contextSelection.droppedScores),
                    "context_char_cap"          to contextSelection.contextCharCap,
                    "final_context_chars"       to contextSelection.finalContextChars,
                    "prompt_chars"              to sanitizedPrompt.value.length,
                    "estimated_prompt_tokens"   to estimatePromptTokens(sanitizedPrompt.value),
                    "answer_route"              to "PENDING_MODEL_MARKER",
                )

                // 4. Add placeholder ASSISTANT message; stream tokens into it.
                _messages.value += ChatMessage(
                    role    = Role.ASSISTANT,
                    text    = "",
                    sources = emptyList(),
                )
                val assistantId = db.messageDao().insert(
                    MessageEntity(
                        documentId = docId,
                        sessionId = sessionId,
                        role = Role.ASSISTANT.name,
                        text = "",
                        sourcesJson = gson.toJson(_messages.value.last().sources),
                        completionState = "streaming",
                        thinkingEnabled = thinkingEnabled,
                    )
                )
                _messages.value = _messages.value.dropLast(1) + _messages.value.last().copy(id = assistantId)

                val sourceExcerpts =
                    contextSelection.keptChunks.map { (_, chunk) -> chunk.take(120) + "." }
                val routeParser = AnswerRouteParser()
                val tokenBuf = StringBuilder()
                var firstTokenFlushed = false
                var uiVisibleTtftLogged = false
                var uiVisibleTtftMs = -1L
                fun flushAssistantChunk(chunk: String, source: String) {
                    if (chunk.isEmpty()) return
                    val list = _messages.value.toMutableList()
                    val last = list.last()
                    val updatedText = last.text + chunk
                    list[list.lastIndex] = last.copy(text = updatedText)
                    _messages.value = list
                    if (!uiVisibleTtftLogged && updatedText.isNotBlank()) {
                        uiVisibleTtftLogged = true
                        uiVisibleTtftMs = EdgeTutorPerf.elapsedMs(queryStartNs)
                        EdgeTutorPerf.log(
                            "query_stage_timing",
                            "doc_id"             to docId,
                            "ui_visible_ttft_ms" to EdgeTutorPerf.elapsedMs(queryStartNs),
                            "source"             to source,
                        )
                    }
                }
                fun applyResolvedRoute() {
                    val route = routeParser.route ?: return
                    activeRoute = route
                    activeRouteReason =
                        if (routeParser.markerValid) "model_route_marker"
                        else "invalid_or_missing_route_marker"
                    val list = _messages.value.toMutableList()
                    val last = list.lastOrNull() ?: return
                    if (last.role == Role.ASSISTANT) {
                        list[list.lastIndex] = last.copy(
                            sources = if (route == QueryRoute.TEXTBOOK) sourceExcerpts else emptyList(),
                        )
                        _messages.value = list
                    }
                }
                var flushJob: Job? = viewModelScope.launch(Dispatchers.Main) {
                    while (isActive) {
                        delay(50)
                        val chunk = synchronized(tokenBuf) {
                            if (tokenBuf.isEmpty()) null
                            else { val s = tokenBuf.toString(); tokenBuf.clear(); s }
                        }
                        if (chunk != null) flushAssistantChunk(chunk, "buffered_flush")
                    }
                }

                if (releaseEmbedder) {
                    EdgeTutorPerf.log("embedder_release_before_generation", "doc_id" to docId)
                    emb.close()
                    embedder = null
                    System.gc()
                    delay(200)
                }

                try {
                    mnnEngine.setThinkingEnabled(thinkingEnabled)
                    val inferencePrompt = if (imagePath != null) {
                        "${sanitizedPrompt.value}\n<img>$imagePath</img>"
                    } else sanitizedPrompt.value
                    val generation = llm.generateMeasured(inferencePrompt) { token ->
                        val parsed = routeParser.consume(token)
                        if (parsed.routeResolvedNow) applyResolvedRoute()
                        if (parsed.visibleText.isEmpty()) return@generateMeasured
                        val immediateChunk = synchronized(tokenBuf) {
                            if (!firstTokenFlushed && parsed.visibleText.isNotBlank()) {
                                firstTokenFlushed = true
                                val chunk = tokenBuf.toString() + parsed.visibleText
                                tokenBuf.clear()
                                chunk
                            } else {
                                tokenBuf.append(parsed.visibleText)
                                null
                            }
                        }
                        if (immediateChunk != null) {
                            flushAssistantChunk(immediateChunk, "first_token_flush")
                        }
                    }
                    val finalRouteChunk = routeParser.finish()
                    if (finalRouteChunk.routeResolvedNow) applyResolvedRoute()
                    synchronized(tokenBuf) { tokenBuf.append(finalRouteChunk.visibleText) }
                    val finalVisibleChunk = synchronized(tokenBuf) {
                        val chunk = tokenBuf.toString()
                        tokenBuf.clear()
                        chunk
                    }
                    if (finalVisibleChunk.isNotEmpty()) {
                        flushAssistantChunk(finalVisibleChunk, "route_final_flush")
                    }
                    val assistant = _messages.value.lastOrNull { it.role == Role.ASSISTANT }
                    if (assistant != null && generation.thinking.isNotBlank()) {
                        _messages.value = _messages.value.dropLast(1) +
                            _messages.value.last().copy(thinking = generation.thinking)
                    }
                    lastQueryResult = QueryExecutionResult(
                        policyId = promptBudgetPolicy.id,
                        question = effectiveQuestion,
                        answer = assistant?.text.orEmpty(),
                        sources = assistant?.sources.orEmpty(),
                        promptChars = sanitizedPrompt.value.length,
                        promptTokens = generation.metrics.promptTokens,
                        prefillUs = generation.metrics.prefillUs,
                        decodeUs = generation.metrics.decodeUs,
                        visibleTtftMs = uiVisibleTtftMs,
                        totalMs = EdgeTutorPerf.elapsedMs(queryStartNs),
                        availableMemoryMb = getAvailableMemoryMB(),
                        route = routeParser.route ?: QueryRoute.FALLBACK_GENERAL,
                        routeReason = activeRouteReason,
                        routeMarkerValid = routeParser.markerValid,
                        maxSimilarity = contextSelection.maxSimilarity,
                        secondSimilarity = (
                            contextSelection.keptScores + contextSelection.droppedScores
                        ).getOrNull(1) ?: Float.NEGATIVE_INFINITY,
                        meanTop5Similarity = (
                            contextSelection.keptScores + contextSelection.droppedScores
                        ).take(5).average().toFloat(),
                    )
                    EdgeTutorPerf.log(
                        "query_complete",
                        "doc_id" to docId,
                        "answer_route" to (routeParser.route?.name ?: "FALLBACK_GENERAL"),
                        "route_reason" to activeRouteReason,
                        "route_marker_valid" to routeParser.markerValid,
                        "max_sim" to contextSelection.maxSimilarity,
                        "ui_visible_ttft_ms" to uiVisibleTtftMs,
                        "total_answer_ms" to EdgeTutorPerf.elapsedMs(queryStartNs),
                        "prompt_tokens" to generation.metrics.promptTokens,
                        "decode_tokens" to generation.metrics.decodeTokens,
                        "answer_chars" to generation.text.length,
                        "blank_visible_answer" to generation.text.isBlank(),
                    )
                    val completed = _messages.value.last()
                    db.messageDao().update(
                        MessageEntity(
                            id = assistantId,
                            documentId = docId,
                            sessionId = sessionId,
                            role = Role.ASSISTANT.name,
                            text = completed.text,
                            thinking = completed.thinking,
                            sourcesJson = gson.toJson(completed.sources),
                            completionState = "complete",
                            thinkingEnabled = thinkingEnabled,
                            promptTokens = generation.metrics.promptTokens,
                            answerTokens = generation.metrics.decodeTokens,
                            prefillUs = generation.metrics.prefillUs,
                            decodeUs = generation.metrics.decodeUs,
                            ttftMs = uiVisibleTtftMs,
                        )
                    )
                    val metricsText =
                        "${generation.metrics.promptTokens} prompt · ${generation.metrics.decodeTokens} answer · TTFT ${uiVisibleTtftMs} ms"
                    _messages.value = _messages.value.dropLast(1) +
                        _messages.value.last().copy(
                            completionState = "complete",
                            metricsText = metricsText,
                        )
                } finally {
                    flushJob?.cancel()
                    flushJob = null
                    val remaining = synchronized(tokenBuf) {
                        val chunk = tokenBuf.toString(); tokenBuf.clear(); chunk
                    }
                    if (remaining.isNotEmpty()) flushAssistantChunk(remaining, "final_flush")
                    if (releaseEmbedder) {
                        val freshEmb = Embedder(app).also { it.warmUp(); embedder = it }
                        embedderCloseJob = viewModelScope.launch {
                            delay(EMBEDDER_CLOSE_DELAY_MS)
                            freshEmb.close()
                            if (embedder === freshEmb) embedder = null
                            embedderCloseJob = null
                        }
                    }
                }
                EdgeTutorPerf.log(
                    "query_stage_timing",
                    "doc_id"          to docId,
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
                EdgeTutorPerf.log(
                    "query_failed",
                    "doc_id" to docId,
                    "answer_route" to (activeRoute?.name ?: "UNDECIDED"),
                    "route_reason" to activeRouteReason,
                    "elapsed_ms" to EdgeTutorPerf.elapsedMs(queryStartNs),
                    "error_type" to e.javaClass.simpleName,
                    "error_message" to (e.message ?: ""),
                )
                _errorMessage.value = "Query failed: ${e.message}"
            } finally {
                _isGenerating.value = false
                generationJob = null
                _thinkingUiState.value =
                    if (shouldPersistThinkingDuration)
                        ThinkingUiState.Done(SystemClock.elapsedRealtime() - thinkingStartMs)
                    else
                        ThinkingUiState.Idle
            }
        }
        generationJob = job
        return job
    }

    fun runValidationSuite() {
        if (_validationStatus.value?.startsWith("running") == true) return
        viewModelScope.launch(Dispatchers.IO) {
            val results = mutableListOf<ValidationResult>()
            _validationStatus.value = "running 0/${EdgeTutorValidationSuite.cases.size}"
            EdgeTutorPerf.log("validation_suite_start", "cases" to EdgeTutorValidationSuite.cases.size)
            try {
                for ((index, case) in EdgeTutorValidationSuite.cases.withIndex()) {
                    resetHistory()
                    case.setupQuestion?.let { setup ->
                        ask(setup)?.join()
                        if (lastQueryResult == null) error("Setup query failed for ${case.id}")
                    }
                    ask(case.question)?.join()
                    val query = lastQueryResult ?: error("Query failed for ${case.id}")
                    results += ValidationResult(
                        caseId = case.id,
                        category = case.category,
                        policyId = query.policyId,
                        question = case.question,
                        answer = query.answer,
                        sources = query.sources,
                        promptChars = query.promptChars,
                        promptTokens = query.promptTokens,
                        prefillUs = query.prefillUs,
                        decodeUs = query.decodeUs,
                        visibleTtftMs = query.visibleTtftMs,
                        totalMs = query.totalMs,
                        availableMemoryMb = query.availableMemoryMb,
                        route = query.route,
                        routeReason = query.routeReason,
                        routeMarkerValid = query.routeMarkerValid,
                        maxSimilarity = query.maxSimilarity,
                        secondSimilarity = query.secondSimilarity,
                        meanTop5Similarity = query.meanTop5Similarity,
                    )
                    _validationStatus.value = "running ${index + 1}/${EdgeTutorValidationSuite.cases.size}"
                    EdgeTutorPerf.log(
                        "validation_suite_progress",
                        "completed" to (index + 1),
                        "total" to EdgeTutorValidationSuite.cases.size,
                        "case_id" to case.id,
                    )
                }
                val outputDir = File(
                    getApplication<Application>().getExternalFilesDir(null),
                    "reports",
                )
                EdgeTutorValidationSuite.writeReports(outputDir, results)
                _validationStatus.value = "complete: ${outputDir.absolutePath}"
                EdgeTutorPerf.log(
                    "validation_suite_complete",
                    "cases" to results.size,
                    "output_dir" to outputDir.absolutePath,
                )
            } catch (e: Exception) {
                _validationStatus.value = "failed: ${e.message}"
                EdgeTutorPerf.log(
                    "validation_suite_failed",
                    "completed" to results.size,
                    "error_type" to e.javaClass.simpleName,
                    "error_message" to (e.message ?: ""),
                )
            } finally {
                resetHistory()
            }
        }
    }

    fun runPromptPolicyBenchmark() {
        if (_validationStatus.value?.startsWith("running") == true) return
        viewModelScope.launch(Dispatchers.IO) {
            val questions = EdgeTutorValidationSuite.cases
                .filter { it.category == com.edgetutor.mnn.perf.ValidationCategory.GROUNDED }
            val total = PromptBudgetPolicy.BENCHMARK_POLICIES.size * questions.size * 3
            val results = mutableListOf<ValidationResult>()
            var completed = 0
            _validationStatus.value = "running benchmark 0/$total"
            try {
                for (policy in PromptBudgetPolicy.BENCHMARK_POLICIES) {
                    setPromptBudgetPolicy(policy)
                    for (case in questions) {
                        repeat(3) { repeatIndex ->
                            resetHistory()
                            ask(case.question)?.join()
                            val query = lastQueryResult ?: error("Query failed for ${case.id}")
                            results += ValidationResult(
                                caseId = "${case.id}-r${repeatIndex + 1}",
                                category = case.category,
                                policyId = query.policyId,
                                question = case.question,
                                answer = query.answer,
                                sources = query.sources,
                                promptChars = query.promptChars,
                                promptTokens = query.promptTokens,
                                prefillUs = query.prefillUs,
                                decodeUs = query.decodeUs,
                                visibleTtftMs = query.visibleTtftMs,
                                totalMs = query.totalMs,
                                availableMemoryMb = query.availableMemoryMb,
                                route = query.route,
                                routeReason = query.routeReason,
                                routeMarkerValid = query.routeMarkerValid,
                                maxSimilarity = query.maxSimilarity,
                                secondSimilarity = query.secondSimilarity,
                                meanTop5Similarity = query.meanTop5Similarity,
                            )
                            completed++
                            _validationStatus.value = "running benchmark $completed/$total"
                        }
                    }
                }
                val outputDir = File(
                    getApplication<Application>().getExternalFilesDir(null),
                    "reports/prompt-benchmark",
                )
                EdgeTutorValidationSuite.writeReports(outputDir, results)
                _validationStatus.value = "benchmark complete: ${outputDir.absolutePath}"
            } catch (e: Exception) {
                _validationStatus.value = "benchmark failed: ${e.message}"
            } finally {
                setPromptBudgetPolicy(PromptBudgetPolicy.DEFAULT)
                resetHistory()
            }
        }
    }

    private suspend fun resetHistory() {
        // submitMessages() also resets natively before every generation. Keep an
        // explicit boundary here so validation cases remain isolated if the
        // native submission policy changes later.
        mnnEngine.reset()
        _messages.value = emptyList()
        _thinkingUiState.value = ThinkingUiState.Idle
    }

    private fun toChatMessage(entity: MessageEntity): ChatMessage = ChatMessage(
        role = Role.valueOf(entity.role),
        text = entity.text,
        sources = runCatching {
            gson.fromJson(entity.sourcesJson, Array<String>::class.java).toList()
        }.getOrDefault(emptyList()),
        id = entity.id,
        thinking = entity.thinking,
        imagePath = entity.imagePath,
        completionState = entity.completionState,
        metricsText = if (entity.promptTokens > 0 || entity.answerTokens > 0) {
            "${entity.promptTokens} prompt · ${entity.answerTokens} answer · TTFT ${entity.ttftMs} ms"
        } else null,
    )

    // -------------------------------------------------------------------------
    // Query routing and context budgeting
    // -------------------------------------------------------------------------

    private fun selectContext(
        results: List<Pair<FlatIndex.Entry, Float>>,
        policy: PromptBudgetPolicy,
    ): ContextSelection {
        val sortedResults = results.sortedByDescending { (_, sim) -> sim }
        val keptCandidates = sortedResults.take(policy.maxKeptChunks)

        val keptEntries = keptCandidates.mapIndexed { index, (entry, _) ->
            entry to "${index + 1}. ${entry.text.take(policy.maxCharsPerChunk)}"
        }
        val keptIds     = keptCandidates.map { (entry, _) -> entry.id }.toSet()
        val keptScores  = keptCandidates.map { (_, sim) -> sim }
        val droppedScores = sortedResults
            .filterIndexed { index, (entry, _) ->
                index >= policy.maxKeptChunks || entry.id !in keptIds
            }
            .map { (_, sim) -> sim }

        return ContextSelection(
            retrievedCount    = results.size,
            keptChunks        = keptEntries,
            keptScores        = keptScores,
            droppedScores     = droppedScores,
            maxSimilarity     = sortedResults.firstOrNull()?.second ?: 0f,
            contextCharCap    = policy.maxCharsPerChunk,
            finalContextChars = keptEntries.sumOf { (_, chunk) -> chunk.length },
        )
    }

    private fun buildRetrievalQuestion(question: String, priorMessages: List<ChatMessage>): String {
        if (!isFollowUpQuestion(question)) return question
        val groundedAssistantIndex = priorMessages.indexOfLast {
            it.role == Role.ASSISTANT && it.sources.isNotEmpty()
        }
        if (groundedAssistantIndex < 1) return question
        val previousUserQuestion = priorMessages
            .subList(0, groundedAssistantIndex)
            .lastOrNull { it.role == Role.USER }
            ?.text
            ?.take(MAX_FOLLOWUP_CONTEXT_CHARS)
            ?: return question
        return "$previousUserQuestion $question"
    }

    private fun buildConversationContext(priorMessages: List<ChatMessage>): String {
        val groundedAssistantIndex = priorMessages.indexOfLast {
            it.role == Role.ASSISTANT && it.sources.isNotEmpty()
        }
        if (groundedAssistantIndex < 1) return ""
        val lastAssistant = priorMessages[groundedAssistantIndex]
        val lastUser = priorMessages
            .subList(0, groundedAssistantIndex)
            .lastOrNull { it.role == Role.USER }
            ?: return ""
        val sb = StringBuilder("Previous question: ${lastUser.text.take(MAX_FOLLOWUP_CONTEXT_CHARS)}")
        if (lastAssistant.text.isNotBlank()) {
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

    // -------------------------------------------------------------------------
    // Prompt template
    // -------------------------------------------------------------------------

    private fun buildRoutedPrompt(
        passages: String,
        conversationContext: String,
        question: String,
        wantsWorkedExample: Boolean,
    ): String = """
Decide whether the textbook passage directly supports answering the question.
Your first output must be exactly one route marker:
- [TEXTBOOK] if the passage supports the answer. Answer using the passage.
- [GENERAL] if it does not. Answer from general knowledge.
Never output both markers. Put the answer immediately after the marker.

TEXTBOOK PASSAGE:
$passages

${if (conversationContext.isBlank()) "" else "PREVIOUS GROUNDED EXCHANGE:\n$conversationContext\n"}

${answerInstruction(wantsWorkedExample)}
Question: $question
""".trimIndent()

    private fun answerInstruction(wantsWorkedExample: Boolean): String =
        if (wantsWorkedExample) {
            "If answering from the textbook, give a small worked example when the passage supports it. " +
                "Write the answer as plain Markdown prose. Only put actual mathematical formulas in LaTeX " +
                "(inline \$...\$ or display \$\$...\$\$); never wrap ordinary words or whole sentences in math delimiters."
        } else {
            "Be direct and explain the relationship. " +
                "Write the answer as plain Markdown prose. Only put actual mathematical formulas in LaTeX " +
                "(inline \$...\$ or display \$\$...\$\$); never wrap ordinary words or whole sentences in math delimiters."
        }

    // -------------------------------------------------------------------------

    private fun estimatePromptTokens(prompt: String): Int = ceil(prompt.length / 4.0).toInt()

    private fun formatScores(scores: List<Float>): String =
        scores.joinToString(",") { "%.4f".format(it) }

    private data class ContextSelection(
        val retrievedCount: Int,
        val keptChunks: List<Pair<FlatIndex.Entry, String>>,
        val keptScores: List<Float>,
        val droppedScores: List<Float>,
        val maxSimilarity: Float,
        val contextCharCap: Int,
        val finalContextChars: Int,
    )

    override fun onCleared() {
        super.onCleared()
        embedderCloseJob?.cancel()
        embedderCloseJob = null
        llm.close()
        embedder?.close()
        embedder = null
    }
}
