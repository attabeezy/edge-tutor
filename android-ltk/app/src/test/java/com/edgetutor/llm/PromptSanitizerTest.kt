package com.edgetutor.llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PromptSanitizerTest {

    @Test
    fun asciiTextPassesThroughUnchanged() {
        val result = PromptSanitizer.sanitize("Line 1\nLine 2\tOK.")

        assertEquals("Line 1\nLine 2\tOK.", result.value)
        assertFalse(result.changed)
        assertEquals(0, result.replacementCount)
        assertEquals(0, result.droppedCount)
    }

    @Test
    fun knownNonAsciiCharactersAreReplaced() {
        val result = PromptSanitizer.sanitize(
            "\u201CSlope\u201D \u2014 x \u2265 2 \u2212 1 \u2248 0",
        )

        assertEquals("\"Slope\" - x >= 2 - 1 ~ 0", result.value)
        assertTrue(result.changed)
        assertEquals(6, result.replacementCount)
        assertEquals(0, result.droppedCount)
    }

    @Test
    fun unsupportedNonAsciiCharactersAreDropped() {
        val result = PromptSanitizer.sanitize("Price \u20AC5 and emoji \uD83D\uDE00")

        assertEquals("Price 5 and emoji ", result.value)
        assertTrue(result.changed)
        assertEquals(0, result.replacementCount)
        assertEquals(2, result.droppedCount)
    }

    @Test
    fun controlCharactersAreDroppedExceptWhitespaceAllowedByPrompt() {
        val result = PromptSanitizer.sanitize("A\u0000B\nC\rD\tE\u001FF")

        assertEquals("AB\nC\rD\tEF", result.value)
        assertTrue(result.changed)
        assertEquals(0, result.replacementCount)
        assertEquals(2, result.droppedCount)
    }
}
