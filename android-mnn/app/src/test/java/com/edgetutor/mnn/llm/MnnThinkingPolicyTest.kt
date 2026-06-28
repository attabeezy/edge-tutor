package com.edgetutor.mnn.llm

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MnnThinkingPolicyTest {
    @Test
    fun disabledOverrideRequestsFalse() {
        assertTrue(MnnThinkingPolicy.isDisabled(MnnThinkingPolicy.disabledConfigJson))
    }

    @Test
    fun effectiveConfigRejectsEnabledThinking() {
        assertFalse(
            MnnThinkingPolicy.isDisabled(
                """{"jinja":{"context":{"enable_thinking":true}}}""",
            ),
        )
    }

    @Test
    fun effectiveConfigRejectsMissingThinkingFlag() {
        assertFalse(MnnThinkingPolicy.isDisabled("""{"jinja":{"context":{}}}"""))
    }

    @Test
    fun detectsHiddenOnlyGeneration() {
        assertTrue(MnnThinkingPolicy.isHiddenOnlyAnswer("", "internal reasoning"))
        assertFalse(MnnThinkingPolicy.isHiddenOnlyAnswer("answer", "internal reasoning"))
        assertFalse(MnnThinkingPolicy.isHiddenOnlyAnswer("", ""))
    }
}
