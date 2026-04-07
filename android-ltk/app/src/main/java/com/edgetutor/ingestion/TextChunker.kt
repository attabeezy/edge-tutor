package com.edgetutor.ingestion

/**
 * Paragraph-aware sliding-window text chunker.
 *
 * Mirrors src/ingestion/pipeline.py::chunk_text():
 *   - CHUNK_TOKENS = 400 words (whitespace-split approximation)
 *   - OVERLAP_TOKENS = 50 words
 *   - Splits on blank lines first; then slides a window word-by-word.
 *
 * Token count is approximated as whitespace-word count. The ONNX tokenizer
 * may produce a slightly different count for the same text, but the difference
 * is small enough not to matter for retrieval quality at this chunk size.
 */
object TextChunker {

    private const val CHUNK_WORDS   = 400
    private const val OVERLAP_WORDS = 50

    data class Chunk(
        /** Zero-based sequential index within the document. */
        val index: Int,
        val text: String,
    )

    fun chunk(text: String): List<Chunk> {
        // Split into paragraphs, filter blanks
        val paragraphs = text.split(Regex("\\n\\s*\\n+"))
            .map { it.trim() }
            .filter { it.isNotEmpty() }

        // Flatten into a word list, inserting a sentinel for paragraph breaks
        val words = mutableListOf<String>()
        for (para in paragraphs) {
            words.addAll(para.split(Regex("\\s+")).filter { it.isNotEmpty() })
            words.add("\u0000")   // paragraph boundary sentinel
        }

        val chunks = mutableListOf<Chunk>()
        var start  = 0
        var idx    = 0

        while (start < words.size) {
            val end       = minOf(start + CHUNK_WORDS, words.size)
            val slice     = words.subList(start, end)
            val chunkText = slice
                .joinToString(" ")
                .replace(" \u0000 ", "\n\n")   // restore paragraph breaks
                .replace("\u0000", "")          // strip trailing sentinels
                .trim()

            if (chunkText.isNotBlank()) {
                chunks.add(Chunk(idx++, chunkText))
            }

            if (end >= words.size) break
            start += CHUNK_WORDS - OVERLAP_WORDS
        }

        return chunks
    }
}
