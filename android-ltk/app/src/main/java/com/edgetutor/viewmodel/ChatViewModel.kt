package com.edgetutor.viewmodel

import android.app.ActivityManager
import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.edgetutor.data.db.DocumentEntity
import com.edgetutor.ingestion.Embedder
import com.edgetutor.llm.LlamaEngine
import com.edgetutor.llm.LlmEngine
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
        private const val MIN_LEXICAL_OVERLAP = 2

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
                    "doc_id" to (activeDoc?.id ?: -1L),
                    "avail_mem_mb" to getAvailableMemoryMB(),
                    "release_embedder" to releaseEmbedder,
                )
                EdgeTutorPerf.snapshot(app, "query_embed_before", "doc_id" to (activeDoc?.id ?: -1L))
                val qVec = EdgeTutorPerf.trace("query_embed", "doc_id" to (activeDoc?.id ?: -1L)) {
                    emb.embed(question, isQuery = true)
                }
                EdgeTutorPerf.snapshot(app, "query_embed_after", "doc_id" to (activeDoc?.id ?: -1L))

                // 2. Retrieve top-3 chunks with similarity scores
                val searchResults = EdgeTutorPerf.trace("retrieval_search", "doc_id" to (activeDoc?.id ?: -1L), "k" to 3) {
                    idx.searchWithScores(qVec, k = 3)
                }
                val topChunks = searchResults.map { (entry, _) -> entry }

                // 3. Out-of-scope gate — reject before hitting the LLM
                if (!isInScope(question, searchResults)) {
                    _messages.value += ChatMessage(
                        role = Role.ASSISTANT,
                        text = "This doesn't appear to be covered in the loaded document.",
                    )
                    return@launch  // finally still runs and resets _isThinking
                }

                // 4. Build prompt (matches Python src/rag/query.py system prompt)
                val contextText = topChunks.joinToString("\n---\n") { it.text }
                val prompt      = buildPrompt(contextText, question)

                // 5. Add a placeholder ASSISTANT message; stream tokens into it.
                //    Tokens are buffered and flushed to StateFlow every 50 ms to
                //    avoid a full list copy + String allocation on every token.
                _messages.value += ChatMessage(
                    role    = Role.ASSISTANT,
                    text    = "",
                    sources = topChunks.map { it.text.take(120) + "…" },
                )

                val tokenBuf = StringBuilder()
                var flushJob: Job? = viewModelScope.launch(Dispatchers.Main) {
                    while (isActive) {
                        delay(50)
                        val chunk = synchronized(tokenBuf) {
                            if (tokenBuf.isEmpty()) null
                            else { val s = tokenBuf.toString(); tokenBuf.clear(); s }
                        }
                        if (chunk != null) {
                            val list = _messages.value.toMutableList()
                            val last = list.last()
                            list[list.lastIndex] = last.copy(text = last.text + chunk)
                            _messages.value = list
                        }
                    }
                }

                if (releaseEmbedder) {
                    emb.close()
                    embedder = null
                }

                try {
                    llm.generate(prompt) { token ->
                        synchronized(tokenBuf) { tokenBuf.append(token) }
                    }
                } finally {
                    flushJob?.cancel()
                    flushJob = null
                    // Final flush — emit any tokens that arrived in the last <50 ms window.
                    val remaining = synchronized(tokenBuf) { tokenBuf.toString() }
                    if (remaining.isNotEmpty()) {
                        val list = _messages.value.toMutableList()
                        val last = list.last()
                        list[list.lastIndex] = last.copy(text = last.text + remaining)
                        _messages.value = list
                    }
                    if (releaseEmbedder) {
                        // On low-memory devices, recreate lazily on the next query
                        // instead of immediately paying the session startup cost here.
                        embedder = null
                    }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _errorMessage.value = "Query failed: ${e.message}"
            } finally {
                _isThinking.value = false
            }
        }
    }

    fun resetHistory() {
        _messages.value = emptyList()
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
    private fun isInScope(question: String, results: List<Pair<FlatIndex.Entry, Float>>): Boolean {
        val maxSim = results.maxOfOrNull { (_, sim) -> sim } ?: return false
        if (maxSim < MIN_COSINE_SIM) return false
        return hasLexicalOverlap(question, results.map { (e, _) -> e.text })
    }

    private fun hasLexicalOverlap(question: String, chunks: List<String>): Boolean {
        val qTokens = question.lowercase()
            .split(Regex("[^a-z]+"))
            .filter { it.isNotEmpty() && it !in STOPWORDS }
            .toSet()
        val required = minOf(MIN_LEXICAL_OVERLAP, maxOf(1, qTokens.size))
        return chunks.any { chunk ->
            val chunkTokens = chunk.lowercase().split(Regex("[^a-z]+")).toSet()
            qTokens.count { it in chunkTokens } >= required
        }
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

    override fun onCleared() {
        super.onCleared()
        llm.close()
        embedder?.close()
        embedder = null
    }
}
