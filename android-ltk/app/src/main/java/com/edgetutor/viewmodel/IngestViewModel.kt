package com.edgetutor.viewmodel

import android.app.ActivityManager
import android.app.Application
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.edgetutor.data.db.AppDatabase
import com.edgetutor.data.db.DocumentEntity
import com.edgetutor.data.db.IngestionStatus
import com.edgetutor.ingestion.Embedder
import com.edgetutor.ingestion.PdfExtractor
import com.edgetutor.ingestion.TextChunker
import com.edgetutor.perf.EdgeTutorPerf
import com.edgetutor.store.FlatIndex
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.ConcurrentHashMap

data class IngestionProgress(
    val phase: String,
    val current: Int,
    val total: Int,
)

class IngestViewModel(app: Application) : AndroidViewModel(app) {

    private val db       = AppDatabase.get(app)
    private val activeJobs = ConcurrentHashMap<Long, Job>()
    val documents = db.documentDao().observeAll()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    private val _progress = MutableStateFlow<Map<Long, IngestionProgress>>(emptyMap())
    val progress: StateFlow<Map<Long, IngestionProgress>> = _progress.asStateFlow()

    companion object {
        private const val TAG = "IngestViewModel"
        private const val DEFAULT_EMBED_BATCH = 32
        private const val DEFAULT_PAGE_WINDOW = 20
        private const val LOW_MEM_EMBED_BATCH = 8
        private const val LOW_MEM_PAGE_WINDOW = 5
        private const val LOW_MEM_THRESHOLD_MB = 50L
    }

    private fun getAvailableMemoryMB(): Long {
        val am = getApplication<Application>().getSystemService(Application.ACTIVITY_SERVICE) as ActivityManager
        val info = ActivityManager.MemoryInfo()
        am.getMemoryInfo(info)
        return info.availMem / (1024 * 1024)
    }

    private fun isLowMemory(): Boolean = getAvailableMemoryMB() < LOW_MEM_THRESHOLD_MB

    private fun computeAdaptiveBatchSize(): Int =
        if (isLowMemory()) LOW_MEM_EMBED_BATCH else DEFAULT_EMBED_BATCH

    private fun computeAdaptivePageWindow(): Int =
        if (isLowMemory()) LOW_MEM_PAGE_WINDOW else DEFAULT_PAGE_WINDOW

    private suspend fun replaceAllDocuments() {
        activeJobs.values.forEach { it.cancel() }
        activeJobs.clear()
        db.documentDao().getAll().forEach { doc ->
            File(getApplication<Application>().filesDir, "${doc.id}.idx").delete()
            db.documentDao().delete(doc)
        }
    }

    fun ingest(uri: Uri, displayName: String) {
        viewModelScope.launch(Dispatchers.IO) {
            replaceAllDocuments()
            val docId = db.documentDao().insert(
                DocumentEntity(displayName = displayName, uriString = uri.toString())
            )
            val doc = db.documentDao().getById(docId)!!
            db.documentDao().update(doc.copy(status = IngestionStatus.RUNNING))
            activeJobs[docId] = coroutineContext[Job]!!

            var index: FlatIndex? = null
            var indexFile: File? = null
            val embedder = Embedder(getApplication())

            try {
                index = FlatIndex()
                indexFile = File(getApplication<Application>().filesDir, "$docId.idx")
                var totalPages = 0
                var isScanned = false
                var globalChunkIdx = 0
                val ingestStartNs = System.nanoTime()
                val app = getApplication<Application>()

                val embedBatch = computeAdaptiveBatchSize()
                val pageWindow = computeAdaptivePageWindow()
                Log.d(TAG, "Starting ingestion: embedBatch=$embedBatch, pageWindow=$pageWindow, freeMem=${getAvailableMemoryMB()}MB")
                EdgeTutorPerf.snapshot(
                    app,
                    "ingest_start",
                    "doc_id" to docId,
                    "embed_batch" to embedBatch,
                    "page_window" to pageWindow,
                )

                index.startAppend(indexFile, embedder.dim)

                PdfExtractor.extractPages(getApplication(), uri, pageWindow).collect { window ->
                    ensureActive()
                    totalPages = window.totalPages
                    isScanned = window.isLikelyScanned
                    setProgress(docId, IngestionProgress("Extracting", window.startPage, window.totalPages))

                    val chunks = TextChunker.chunk(window.text)
                    chunks.chunked(embedBatch).forEach { batch ->
                        ensureActive()
                        setProgress(docId, IngestionProgress("Embedding", globalChunkIdx, -1))

                        val vectors = EdgeTutorPerf.trace(
                            "embed_batch",
                            "doc_id" to docId,
                            "batch_size" to batch.size,
                        ) {
                            embedder.embedBatch(batch.map { it.text })
                        }
                        batch.forEachIndexed { i, chunk ->
                            requireNotNull(index).append(FlatIndex.Entry((globalChunkIdx + chunk.index).toLong(), chunk.text, vectors[i]))
                        }
                    }

                    globalChunkIdx += chunks.size
                }

                index.finishAppend()

                db.documentDao().update(
                    doc.copy(
                        status = IngestionStatus.DONE,
                        pageCount = totalPages,
                        chunkCount = globalChunkIdx,
                        isLikelyScanned = isScanned,
                    )
                )
                val ingestDurationMs = (System.nanoTime() - ingestStartNs) / 1_000_000
                EdgeTutorPerf.log(
                    "ingest_total",
                    "doc_id" to docId,
                    "duration_ms" to ingestDurationMs,
                    "pages" to totalPages,
                    "chunks" to globalChunkIdx,
                )
                EdgeTutorPerf.log(
                    "ingest_pages_per_sec",
                    "doc_id" to docId,
                    "pages" to totalPages,
                    "pages_per_sec" to if (ingestDurationMs > 0) (totalPages * 1000.0 / ingestDurationMs) else 0.0,
                )
                EdgeTutorPerf.snapshot(
                    app,
                    "ingest_end",
                    "doc_id" to docId,
                    "pages" to totalPages,
                    "chunks" to globalChunkIdx,
                )
            } catch (e: CancellationException) {
                db.documentDao().update(doc.copy(status = IngestionStatus.ERROR, errorMessage = "Cancelled"))
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Ingestion failed for doc $docId", e)
                db.documentDao().update(
                    doc.copy(
                        status = IngestionStatus.ERROR,
                        errorMessage = e.message ?: e.javaClass.simpleName,
                    )
                )
                indexFile?.delete()
            } finally {
                embedder.close()  // free 91 MB ORT session immediately after ingestion
                index?.let {
                    try { it.finishAppend() } catch (_: Exception) {}
                }
                activeJobs.remove(docId)
                clearProgress(docId)
            }
        }
    }

    fun cancelIngest(docId: Long) {
        activeJobs[docId]?.cancel()
    }

    fun delete(doc: DocumentEntity) {
        viewModelScope.launch(Dispatchers.IO) {
            File(getApplication<Application>().filesDir, "${doc.id}.idx").delete()
            db.documentDao().delete(doc)
        }
    }

    override fun onCleared() {
        super.onCleared()
    }

    private fun setProgress(docId: Long, p: IngestionProgress) {
        _progress.value = _progress.value + (docId to p)
    }

    private fun clearProgress(docId: Long) {
        _progress.value = _progress.value - docId
    }
}
