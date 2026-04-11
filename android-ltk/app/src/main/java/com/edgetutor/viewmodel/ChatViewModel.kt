package com.edgetutor.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.edgetutor.data.db.DocumentEntity
import com.edgetutor.ingestion.Embedder
import com.edgetutor.llm.LlamaEngine
import com.edgetutor.llm.LlmEngine
import com.edgetutor.store.FlatIndex
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
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

    private var index: FlatIndex?    = null
    var activeDoc: DocumentEntity?   = null
        private set

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
                val idx = FlatIndex()
                idx.load(file)
                index            = idx
                activeDoc        = doc
                _isLikelyScanned.value = doc.isLikelyScanned
                _messages.value  = emptyList()
                llm.warmUp()      // eagerly load LLM weights
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

        viewModelScope.launch(Dispatchers.IO) {
            _isThinking.value = true
            _messages.value  += ChatMessage(Role.USER, question)
            try {
                // 1. Embed question; close the ORT session immediately after to free ~23 MB
                //    before the LLM needs its full allocation.
                val qVec = Embedder(getApplication<Application>()).use { it.embed(question, isQuery = true) }

                // 2. Retrieve top-3 chunks
                val topChunks = idx.search(qVec, k = 3)

                // 3. Build prompt (matches Python src/rag/query.py system prompt)
                val contextText = topChunks.joinToString("\n---\n") { it.text }
                val prompt      = buildPrompt(contextText, question)

                // 4. Add a placeholder ASSISTANT message; stream tokens into it
                _messages.value += ChatMessage(
                    role    = Role.ASSISTANT,
                    text    = "",
                    sources = topChunks.map { it.text.take(120) + "…" },
                )

                llm.generate(prompt) { token ->
                    val list    = _messages.value.toMutableList()
                    val last    = list.last()
                    list[list.lastIndex] = last.copy(text = last.text + token)
                    _messages.value = list
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
    }
}
