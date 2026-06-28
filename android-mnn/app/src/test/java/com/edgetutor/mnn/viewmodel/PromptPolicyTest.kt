package com.edgetutor.mnn.viewmodel

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PromptPolicyTest {
    @Test
    fun benchmarkPoliciesCoverPlannedMatrix() {
        assertEquals(
            listOf("2x800", "2x500", "1x800", "1x500"),
            PromptBudgetPolicy.BENCHMARK_POLICIES.map { it.id },
        )
    }

    @Test
    fun generalAnswerDetectionIgnoresLeadingWhitespaceAndCase() {
        assertTrue(
            AnswerAttributionPolicy.isGeneralKnowledgeAnswer(
                "  the textbook does not cover this. general answer: Tokyo",
            ),
        )
        assertFalse(AnswerAttributionPolicy.isGeneralKnowledgeAnswer("Calculus studies change."))
    }
}
