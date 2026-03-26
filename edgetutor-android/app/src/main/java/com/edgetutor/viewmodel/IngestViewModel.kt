package com.edgetutor.viewmodel

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.edgetutor.data.db.AppDatabase
import com.edgetutor.data.db.DocumentEntity
import com.edgetutor.data.db.IngestionStatus
import com.edgetutor.ingestion.Embedder
import com.edgetutor.ingestion.PdfExtractor
import com.edgetutor.ingestion.TextChunker
import com.edgetutor.store.FlatIndex
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.io.File

class IngestViewModel(app: Application) : AndroidViewModel(app) {

    private val db      = AppDatabase.get(app)
    private val embedder by lazy { Embedder(app) }

    /** All documents, latest-first. Exposed as StateFlow for Compose. */
    val documents = db.documentDao().observeAll()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    /**
     * Start background ingestion for [uri].
     * Creates a PENDING row immediately (so the UI shows it right away), then
     * runs PDF → chunk → embed → index → DONE on IO threads.
     *
     * @param uri         Content URI from the file picker.
     * @param displayName Human-readable filename shown in the library.
     */
    fun ingest(uri: Uri, displayName: String) {
        viewModelScope.launch(Dispatchers.IO) {
            // 1. Insert PENDING row
            val docId = db.documentDao().insert(
                DocumentEntity(displayName = displayName, uriString = uri.toString())
            )
            val doc = db.documentDao().getById(docId)!!
            db.documentDao().update(doc.copy(status = IngestionStatus.RUNNING))

            try {
                // 2. Extract text
                val extracted = PdfExtractor.extract(getApplication(), uri)
                // Scanned-PDF warning is surfaced by observing IngestionStatus.DONE
                // with pageCount > 0 but chunkCount very low (callers can check).

                // 3. Chunk
                val chunks = TextChunker.chunk(extracted.text)

                // 4. Embed (batch call — avoids repeated ONNX session overhead)
                val vectors = embedder.embedBatch(chunks.map { it.text })

                // 5. Build flat index
                val index = FlatIndex()
                chunks.forEachIndexed { i, chunk ->
                    index.add(chunk.index.toLong(), chunk.text, vectors[i])
                }

                // 6. Persist to filesDir/<docId>.idx
                val indexFile = File(getApplication<Application>().filesDir, "$docId.idx")
                index.save(indexFile)

                // 7. Mark DONE
                db.documentDao().update(
                    doc.copy(
                        status     = IngestionStatus.DONE,
                        pageCount  = extracted.pageCount,
                        chunkCount = chunks.size,
                    )
                )
            } catch (e: Exception) {
                db.documentDao().update(
                    doc.copy(status = IngestionStatus.ERROR, errorMessage = e.message)
                )
            }
        }
    }

    fun delete(doc: DocumentEntity) {
        viewModelScope.launch(Dispatchers.IO) {
            File(getApplication<Application>().filesDir, "${doc.id}.idx").delete()
            db.documentDao().delete(doc)
        }
    }

    override fun onCleared() {
        super.onCleared()
        embedder.close()
    }
}
