package com.edgetutor.mnn.perf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ValidationSuiteTest {
    @Test
    fun suiteHasFourCasesPerCategory() {
        assertEquals(40, EdgeTutorValidationSuite.cases.size)
        ValidationCategory.entries.forEach { category ->
            assertEquals(4, EdgeTutorValidationSuite.cases.count { it.category == category })
        }
    }

    @Test
    fun reportsContainMetricsAndManualRubricColumns() {
        val result = ValidationResult(
            caseId = "g1",
            category = ValidationCategory.GROUNDED,
            policyId = "1x500",
            question = "What is calculus?",
            answer = "It studies change.",
            sources = listOf("A source"),
            promptChars = 500,
            promptTokens = 125,
            prefillUs = 100,
            decodeUs = 200,
            visibleTtftMs = 300,
            totalMs = 400,
            availableMemoryMb = 800,
        )
        val csv = listOf(result).toCsv()
        assertTrue(csv.contains("correctness_0_2"))
        assertTrue(csv.contains("max_similarity"))
        assertTrue(csv.contains("second_similarity"))
        assertTrue(csv.contains("mean_top5_similarity"))
        assertTrue(csv.contains("answer_route"))
        assertTrue(csv.contains("route_marker_valid"))
        assertTrue(csv.contains("tutor_move_0_2"))
        assertTrue(csv.contains("answer_restraint_0_2"))
        assertTrue(csv.contains("adaptation_0_2"))
        assertTrue(csv.contains("corrective_feedback_0_2"))
        assertTrue(csv.contains("\"1x500\""))
        assertTrue(csv.contains("\"100\""))
    }
}
