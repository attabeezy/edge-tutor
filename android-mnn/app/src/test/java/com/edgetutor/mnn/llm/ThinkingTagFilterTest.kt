package com.edgetutor.mnn.llm

import org.junit.Assert.assertEquals
import org.junit.Test

class ThinkingTagFilterTest {
    @Test
    fun `removes complete thinking block`() {
        val filter = ThinkingTagFilter()

        assertEquals("answer", filter.filter("<think>hidden</think>answer"))
    }

    @Test
    fun `removes thinking block split across deltas`() {
        val filter = ThinkingTagFilter()

        assertEquals("before ", filter.filter("before <thi"))
        assertEquals("", filter.filter("nk>hidden"))
        assertEquals("", filter.filter("</thi"))
        assertEquals(" after", filter.filter("nk> after"))
    }

    @Test
    fun `preserves normal text when no thinking tags exist`() {
        val filter = ThinkingTagFilter()

        assertEquals("plain text", filter.filter("plain text"))
    }
}
