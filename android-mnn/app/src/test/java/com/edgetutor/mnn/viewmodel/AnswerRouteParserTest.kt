package com.edgetutor.mnn.viewmodel

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AnswerRouteParserTest {
    @Test
    fun parsesTextbookMarkerSplitAcrossChunks() {
        val parser = AnswerRouteParser()

        assertEquals("", parser.consume("  [TEXT").visibleText)
        assertEquals("A derivative is a slope.", parser.consume("BOOK] A derivative is a slope.").visibleText)
        assertEquals(QueryRoute.TEXTBOOK, parser.route)
        assertTrue(parser.markerValid)
    }

    @Test
    fun labelsGeneralAnswerExactlyOnce() {
        val parser = AnswerRouteParser()

        val first = parser.consume("[GENERAL] Tokyo.")
        val second = parser.consume(" It is Japan's capital.")

        assertEquals(AnswerRouteParser.GENERAL_LABEL + "Tokyo.", first.visibleText)
        assertEquals(" It is Japan's capital.", second.visibleText)
        assertEquals(1, (first.visibleText + second.visibleText).windowed(
            AnswerRouteParser.GENERAL_LABEL.length,
        ).count { it == AnswerRouteParser.GENERAL_LABEL })
        assertEquals(QueryRoute.GENERAL, parser.route)
    }

    @Test
    fun malformedMarkerFailsClosedWithoutDroppingAnswer() {
        val parser = AnswerRouteParser()
        val result = parser.consume("The passage says calculus studies change.")

        assertEquals(
            AnswerRouteParser.GENERAL_LABEL + "The passage says calculus studies change.",
            result.visibleText,
        )
        assertEquals(QueryRoute.FALLBACK_GENERAL, parser.route)
        assertFalse(parser.markerValid)
    }

    @Test
    fun missingMarkerAtEndFailsClosed() {
        val parser = AnswerRouteParser()

        assertEquals("", parser.consume("[TEXT").visibleText)
        val final = parser.finish()

        assertEquals(AnswerRouteParser.GENERAL_LABEL + "[TEXT", final.visibleText)
        assertEquals(QueryRoute.FALLBACK_GENERAL, parser.route)
        assertFalse(parser.markerValid)
    }

    @Test
    fun markerOnlyOutputIsResolvedAndHidden() {
        val parser = AnswerRouteParser()
        val result = parser.consume("[TEXTBOOK]")

        assertEquals("", result.visibleText)
        assertEquals(QueryRoute.TEXTBOOK, parser.route)
        assertTrue(parser.markerValid)
    }
}
