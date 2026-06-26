package com.edgetutor.mnn.llm

import org.junit.Assert.assertEquals
import org.junit.Test

class PromptSanitizerTest {

    @Test
    fun `plain ASCII passes through unchanged`() {
        val result = PromptSanitizer.sanitize("Hello World 123")
        assertEquals("Hello World 123", result.value)
        assert(!result.changed)
    }

    @Test
    fun `unicode dashes are replaced with ASCII hyphens`() {
        val result = PromptSanitizer.sanitize("a\u2013b")   // en dash
        assertEquals("a-b", result.value)
        assertEquals(1, result.replacementCount)
    }

    @Test
    fun `curly quotes are replaced`() {
        val result = PromptSanitizer.sanitize("\u201CHello\u201D")
        assertEquals("\"Hello\"", result.value)
    }

    @Test
    fun `control characters are dropped`() {
        val result = PromptSanitizer.sanitize("a\u0001b")
        assertEquals("ab", result.value)
        assertEquals(1, result.droppedCount)
    }
}
