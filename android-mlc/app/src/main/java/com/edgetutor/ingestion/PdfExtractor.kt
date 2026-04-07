package com.edgetutor.ingestion

import android.content.Context
import android.net.Uri
import android.util.Log
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.io.File

object PdfExtractor {

    private const val TAG = "PdfExtractor"

    /**
     * Call once at app startup (before any [extract] or [extractPages] call).
     * PdfBox Android needs access to Android font resources.
     */
    fun init(context: Context) = PDFBoxResourceLoader.init(context)

    data class Result(
        val text: String,
        val pageCount: Int,
        val avgWordsPerPage: Double,
        val isLikelyScanned: Boolean,
    )

    data class PageWindow(
        val text: String,
        val startPage: Int,
        val totalPages: Int,
        val isLikelyScanned: Boolean,
    )

    /**
     * Copy content URI to a temp file for memory-mapped access.
     * PdfBox loads file-backed documents more efficiently than stream-backed ones.
     */
    private fun copyToCache(context: Context, uri: Uri): File {
        val tempFile = File(context.cacheDir, "pdf_${System.nanoTime()}.tmp")
        context.contentResolver.openInputStream(uri)?.use { input ->
            tempFile.outputStream().use { output ->
                input.copyTo(output)
            }
        } ?: error("Cannot open URI: $uri")
        return tempFile
    }

    /**
     * Extract plain text from a PDF [uri] in one pass.
     * Uses file-backed loading to reduce memory pressure.
     */
    fun extract(context: Context, uri: Uri): Result {
        val tempFile = copyToCache(context, uri)
        try {
            val doc = PDDocument.load(tempFile)
            try {
                val text = PDFTextStripper().getText(doc)
                val pages = doc.numberOfPages
                val words = text.trim().split(Regex("\\s+")).count { it.isNotEmpty() }.toDouble()
                val avgWords = if (pages > 0) words / pages else 0.0
                return Result(
                    text = text,
                    pageCount = pages,
                    avgWordsPerPage = avgWords,
                    isLikelyScanned = avgWords < 100.0,
                )
            } finally {
                doc.close()
            }
        } finally {
            if (!tempFile.delete()) {
                Log.w(TAG, "Failed to delete temp file: ${tempFile.absolutePath}")
            }
        }
    }

    /**
     * Streaming extraction: loads PDF from file-backed storage and emits one [PageWindow]
     * per [pageWindow] pages. Uses file-backed loading instead of stream-backed to enable
     * PdfBox's random-access optimization and reduce memory pressure.
     *
     * The flow is cancellation-cooperative: cancelling the collector stops extraction
     * between window emissions (at each [emit] suspension point).
     *
     * Scanned-PDF detection is done cheaply on the first 5 pages and propagated to every
     * emitted window via [PageWindow.isLikelyScanned].
     */
    fun extractPages(context: Context, uri: Uri, pageWindow: Int = 20): Flow<PageWindow> = flow {
        val tempFile = copyToCache(context, uri)
        try {
            val doc = PDDocument.load(tempFile)
            try {
                val totalPages = doc.numberOfPages

                val sampleEnd = minOf(5, totalPages)
                val sampleWords = PDFTextStripper().apply {
                    setStartPage(1); setEndPage(sampleEnd)
                }.getText(doc).trim().split(Regex("\\s+")).count { it.isNotEmpty() }
                val isLikelyScanned = (sampleWords.toDouble() / sampleEnd.coerceAtLeast(1)) < 100.0

                var startPage = 1
                while (startPage <= totalPages) {
                    val endPage = minOf(startPage + pageWindow - 1, totalPages)
                    val pageText = PDFTextStripper().apply {
                        setStartPage(startPage); setEndPage(endPage)
                    }.getText(doc)
                    emit(PageWindow(pageText, startPage, totalPages, isLikelyScanned))
                    startPage += pageWindow
                }
            } finally {
                doc.close()
            }
        } finally {
            if (!tempFile.delete()) {
                Log.w(TAG, "Failed to delete temp file: ${tempFile.absolutePath}")
            }
        }
    }
}
