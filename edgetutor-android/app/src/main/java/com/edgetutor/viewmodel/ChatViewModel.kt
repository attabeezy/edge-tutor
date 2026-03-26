package com.edgetutor.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.edgetutor.data.db.DocumentEntity
import com.edgetutor.ingestion.Embedder
import com.edgetutor.llm.LlamaEngine
import com.edgetutor.llm.LlmEngine
import com.edgetutor.store.FlatIndex
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

    private val embedder: Embedder   by lazy { Embedder(app) }
    private val llm: LlmEngine       by lazy { LlamaEngine(app) }

    private val _messages    = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages

    private val _isThinking  = MutableStateFlow(false)
    val isThinking: StateFlow<Boolean> = _isThinking

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
            val file = File(getApplication<Application>().filesDir, "${doc.id}.idx")
            if (!file.exists()) return@launch
            val idx = FlatIndex()
            idx.load(file)
            index     = idx
            activeDoc = doc
            _messages.value = emptyList()
        }
    }

    // ---------------------------------------------------------------------------
    // Querying
    // ---------------------------------------------------------------------------

    fun ask(question: String) {
        val idx = index ?: return

        viewModelScope.launch(Dispatchers.IO) {
            _isThinking.value = true
            _messages.value  += ChatMessage(Role.USER, question)

            // 1. Embed question
            val qVec = embedder.embed(question)

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

            _isThinking.value = false
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
        embedder.close()
        llm.close()
    }
}
