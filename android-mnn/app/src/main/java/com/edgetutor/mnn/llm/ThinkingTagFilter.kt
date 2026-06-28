package com.edgetutor.mnn.llm

class ThinkingTagFilter {
    private var suppressing = false
    private var carry = ""
    private val captured = StringBuilder()
    val thinkingText: String get() = captured.toString().trim()

    fun filter(delta: String): String {
        if (delta.isEmpty()) return ""

        val input = carry + delta
        carry = ""
        val out = StringBuilder()
        var cursor = 0

        while (cursor < input.length) {
            if (suppressing) {
                val end = input.indexOf("</think>", cursor)
                if (end < 0) {
                    val remainder = input.substring(cursor)
                    carry = remainder.takeLastPartialPrefixOf("</think>")
                    captured.append(remainder.dropLast(carry.length))
                    return out.toString()
                }
                captured.append(input.substring(cursor, end))
                cursor = end + "</think>".length
                suppressing = false
            } else {
                val start = input.indexOf("<think>", cursor)
                if (start < 0) {
                    val text = input.substring(cursor)
                    val partial = text.takeLastPartialPrefixOf("<think>")
                    if (partial.isNotEmpty()) {
                        out.append(text.dropLast(partial.length))
                        carry = partial
                    } else {
                        out.append(text)
                    }
                    break
                }
                out.append(input.substring(cursor, start))
                cursor = start + "<think>".length
                suppressing = true
            }
        }

        return out.toString()
    }

    private fun String.takeLastPartialPrefixOf(tag: String): String {
        val maxLen = minOf(length, tag.length - 1)
        for (len in maxLen downTo 1) {
            val suffix = takeLast(len)
            if (tag.startsWith(suffix)) return suffix
        }
        return ""
    }
}
