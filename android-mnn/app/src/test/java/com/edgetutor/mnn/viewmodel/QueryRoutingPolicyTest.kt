package com.edgetutor.mnn.viewmodel

import org.junit.Assert.assertEquals
import org.junit.Test

class QueryRoutingPolicyTest {
    @Test
    fun routesHighSemanticSimilarityToGrounded() {
        val decision = QueryRoutingPolicy.decide(
            QueryRoutingPolicy.MIN_GROUNDED_COSINE_SIMILARITY + 0.01f,
        )

        assertEquals(QueryRoute.GROUNDED, decision.route)
        assertEquals("semantic_threshold_met", decision.reason)
    }

    @Test
    fun routesLowSemanticSimilarityToGeneral() {
        val decision = QueryRoutingPolicy.decide(
            QueryRoutingPolicy.MIN_GROUNDED_COSINE_SIMILARITY - 0.01f,
        )

        assertEquals(QueryRoute.GENERAL, decision.route)
        assertEquals("semantic_threshold_not_met", decision.reason)
    }

    @Test
    fun routesMissingRetrievalToGeneral() {
        assertEquals(QueryRoute.GENERAL, QueryRoutingPolicy.decide(null).route)
    }

    @Test
    fun groundedFollowUpInheritsPreviousRouteDespiteLowScore() {
        val decision = QueryRoutingPolicy.inheritFollowUp(
            isFollowUp = true,
            previousAnswerWasGrounded = true,
            maxSimilarity = 0.01f,
        )

        assertEquals(QueryRoute.GROUNDED, decision.route)
        assertEquals("followup_inherited_grounded", decision.reason)
    }

    @Test
    fun generalFollowUpInheritsPreviousRouteDespiteHighScore() {
        val decision = QueryRoutingPolicy.inheritFollowUp(
            isFollowUp = true,
            previousAnswerWasGrounded = false,
            maxSimilarity = 0.99f,
        )

        assertEquals(QueryRoute.GENERAL, decision.route)
        assertEquals("followup_inherited_general", decision.reason)
    }

    @Test
    fun newQuestionDoesNotInheritPreviousRoute() {
        val decision = QueryRoutingPolicy.inheritFollowUp(
            isFollowUp = false,
            previousAnswerWasGrounded = false,
            maxSimilarity = 0.99f,
        )

        assertEquals(QueryRoute.GROUNDED, decision.route)
        assertEquals("semantic_threshold_met", decision.reason)
    }
}
