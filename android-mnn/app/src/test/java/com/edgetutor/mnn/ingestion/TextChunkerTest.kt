package com.edgetutor.mnn.ingestion

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TextChunkerTest {
    @Test
    fun `short text becomes one chunk`() {
        val chunks = TextChunker.chunk("One paragraph.\n\nSecond paragraph.")

        assertEquals(1, chunks.size)
        assertTrue(chunks.single().text.contains("One paragraph."))
    }

    @Test
    fun `long text uses overlapping chunks`() {
        val text = (1..760).joinToString(" ") { "w$it" }

        val chunks = TextChunker.chunk(text)

        assertEquals(3, chunks.size)
        assertTrue(chunks[1].text.startsWith("w351 "))
        assertTrue(chunks[2].text.startsWith("w701 "))
    }
}
