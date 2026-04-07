package com.edgetutor.ingestion

import android.content.Context

/**
 * Minimal WordPiece tokenizer for all-MiniLM-L6-v2 (BERT vocab, 30k tokens).
 *
 * Loads vocab.txt from assets — one token per line, line index == token ID.
 * Thread-safe after construction (read-only maps).
 *
 * Mirrors the Python tokenizer closely enough for MVP; edge cases (e.g. CJK,
 * accent stripping) are not handled but are irrelevant for English engineering text.
 */
class WordPieceTokenizer(context: Context) {

    private val vocab: Map<String, Int>

    val clsId: Int
    val sepId: Int
    val unkId: Int
    val padId: Int

    init {
        val lines = context.assets.open("vocab.txt").bufferedReader().readLines()
        val map = HashMap<String, Int>(lines.size * 2)
        lines.forEachIndexed { i, token -> map[token] = i }
        vocab  = map
        clsId  = map["[CLS]"] ?: error("vocab.txt missing [CLS]")
        sepId  = map["[SEP]"] ?: error("vocab.txt missing [SEP]")
        unkId  = map["[UNK]"] ?: error("vocab.txt missing [UNK]")
        padId  = map["[PAD]"] ?: error("vocab.txt missing [PAD]")
    }

    data class Encoding(
        val inputIds: LongArray,
        val attentionMask: LongArray,
        val tokenTypeIds: LongArray,
    )

    /**
     * Tokenize [text] and return a fixed-length [maxLen] encoding (padded / truncated).
     * Output shape matches what the ONNX model expects: [1, maxLen].
     */
    fun encode(text: String, maxLen: Int = 128): Encoding {
        val tokens = wordPiece(text.lowercase().trim())
        val truncated = tokens.take(maxLen - 2)          // room for [CLS] + [SEP]

        val ids   = LongArray(maxLen) { padId.toLong() }
        val mask  = LongArray(maxLen) { 0L }
        val types = LongArray(maxLen) { 0L }

        ids[0] = clsId.toLong(); mask[0] = 1L
        truncated.forEachIndexed { i, id ->
            ids[i + 1]  = id.toLong()
            mask[i + 1] = 1L
        }
        val sepPos = truncated.size + 1
        if (sepPos < maxLen) { ids[sepPos] = sepId.toLong(); mask[sepPos] = 1L }

        return Encoding(ids, mask, types)
    }

    // ---------------------------------------------------------------------------
    // WordPiece algorithm
    // ---------------------------------------------------------------------------

    private fun wordPiece(text: String): List<Int> {
        val result = mutableListOf<Int>()
        // Split on whitespace; basic punctuation splitting
        val words = splitOnPunct(text).filter { it.isNotEmpty() }

        for (word in words) {
            if (word.length > 100) { result.add(unkId); continue }

            var start = 0
            var isBad = false
            val wordTokens = mutableListOf<Int>()

            while (start < word.length) {
                var end = word.length
                var foundId = -1
                while (start < end) {
                    val sub = if (start == 0) word.substring(start, end)
                               else "##" + word.substring(start, end)
                    val id = vocab[sub]
                    if (id != null) { foundId = id; break }
                    end--
                }
                if (foundId == -1) { isBad = true; break }
                wordTokens.add(foundId)
                start = end
            }

            if (isBad) result.add(unkId) else result.addAll(wordTokens)
        }
        return result
    }

    /** Split text on whitespace, also inserting punctuation chars as separate tokens. */
    private fun splitOnPunct(text: String): List<String> {
        val tokens = mutableListOf<String>()
        val buf = StringBuilder()
        for (ch in text) {
            if (ch.isWhitespace()) {
                if (buf.isNotEmpty()) { tokens.add(buf.toString()); buf.clear() }
            } else if (isPunct(ch)) {
                if (buf.isNotEmpty()) { tokens.add(buf.toString()); buf.clear() }
                tokens.add(ch.toString())
            } else {
                buf.append(ch)
            }
        }
        if (buf.isNotEmpty()) tokens.add(buf.toString())
        return tokens
    }

    private fun isPunct(ch: Char): Boolean =
        ch in "!\"#\$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
}
