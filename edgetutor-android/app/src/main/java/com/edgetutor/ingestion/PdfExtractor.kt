package com.edgetutor.ingestion

import android.content.Context
import android.net.Uri
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper

object PdfExtractor {

    /**
     * Call once at app startup (before any [extract] call).
     * PdfBox Android needs access to Android font resources.
     */
    fun init(context: Context) = PDFBoxResourceLoader.init(context)

    data class Result(
        val text: String,
        val pageCount: Int,
        val avgWordsPerPage: Double,
        /** True if the PDF appears to be a scanned image (very low word density). */
        val isLikelyScanned: Boolean,
    )

    /**
     * Extract plain text from a PDF [uri].
     * Returns a [Result]; [Result.isLikelyScanned] is set when avg words/page < 100,
     * which callers should surface as a warning to the user.
     */
    fun extract(context: Context, uri: Uri): Result {
        val stream = context.contentResolver.openInputStream(uri)
            ?: error("Cannot open URI: $uri")

        val doc     = PDDocument.load(stream)
        val text    = PDFTextStripper().getText(doc)
        val pages   = doc.numberOfPages
        doc.close()
        stream.close()

        val words   = text.trim().split(Regex("\\s+")).count { it.isNotEmpty() }.toDouble()
        val avgWords = if (pages > 0) words / pages else 0.0

        return Result(
            text             = text,
            pageCount        = pages,
            avgWordsPerPage  = avgWords,
            isLikelyScanned  = avgWords < 100.0,
        )
    }
}
