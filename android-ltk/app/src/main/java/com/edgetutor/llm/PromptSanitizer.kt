package com.edgetutor.llm

object PromptSanitizer {

    fun sanitize(text: String): SanitizedText {
        val out = StringBuilder(text.length)
        var replacementCount = 0
        var droppedCount = 0
        var hadNonAscii = false

        var index = 0
        while (index < text.length) {
            val codePoint = text.codePointAt(index)
            index += Character.charCount(codePoint)

            when {
                codePoint in 0x20..0x7E || codePoint == '\n'.code || codePoint == '\r'.code || codePoint == '\t'.code -> {
                    out.appendCodePoint(codePoint)
                }
                codePoint < 0x20 -> {
                    droppedCount += 1
                }
                else -> {
                    hadNonAscii = true
                    val replacement = REPLACEMENTS[codePoint]
                    if (replacement != null) {
                        out.append(replacement)
                        replacementCount += 1
                    } else {
                        droppedCount += 1
                    }
                }
            }
        }

        return SanitizedText(
            value = out.toString(),
            hadNonAscii = hadNonAscii,
            replacementCount = replacementCount,
            droppedCount = droppedCount,
        )
    }

    data class SanitizedText(
        val value: String,
        val hadNonAscii: Boolean,
        val replacementCount: Int,
        val droppedCount: Int,
    ) {
        val changed: Boolean
            get() = hadNonAscii || replacementCount > 0 || droppedCount > 0
    }

    private val REPLACEMENTS = mapOf(
        0x00A0 to " ",
        0x2010 to "-",
        0x2011 to "-",
        0x2012 to "-",
        0x2013 to "-",
        0x2014 to "-",
        0x2015 to "-",
        0x2018 to "'",
        0x2019 to "'",
        0x201A to "'",
        0x201B to "'",
        0x201C to "\"",
        0x201D to "\"",
        0x201E to "\"",
        0x2022 to "-",
        0x2026 to "...",
        0x2032 to "'",
        0x2033 to "\"",
        0x2043 to "-",
        0x2212 to "-",
        0x2217 to "*",
        0x221A to "sqrt",
        0x221E to "infinity",
        0x2248 to "~",
        0x2260 to "!=",
        0x2264 to "<=",
        0x2265 to ">=",
    )
}
