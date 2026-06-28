package com.edgetutor.mnn.viewmodel

import org.junit.Assert.assertEquals
import org.junit.Test

class QueryRoutingPolicyTest {
    @Test
    fun routesHighMeanTop5SimilarityToGrounded() {
        val decision = QueryRoutingPolicy.decide(List(5) { 0.7f })

        assertEquals(QueryRoute.GROUNDED, decision.route)
        assertEquals("mean_top5_threshold_met", decision.reason)
        assertEquals(0.7f, decision.meanTop5Similarity, 0.0001f)
    }

    @Test
    fun routesLowMeanTop5SimilarityToGeneral() {
        val decision = QueryRoutingPolicy.decide(List(5) { 0.6f })

        assertEquals(QueryRoute.GENERAL, decision.route)
        assertEquals("mean_top5_threshold_not_met", decision.reason)
    }

    @Test
    fun routesFewerThanFiveScoresToGeneral() {
        val decision = QueryRoutingPolicy.decide(listOf(0.9f, 0.8f, 0.7f, 0.6f))
        assertEquals(QueryRoute.GENERAL, decision.route)
        assertEquals("insufficient_similarity_distribution", decision.reason)
    }

    @Test
    fun thresholdValueRoutesGrounded() {
        val decision = QueryRoutingPolicy.decide(
            List(5) { QueryRoutingPolicy.MIN_MEAN_TOP5_SIMILARITY },
        )
        assertEquals(QueryRoute.GROUNDED, decision.route)
    }

    @Test
    fun sortsScoresBeforeSelectingTopFive() {
        val decision = QueryRoutingPolicy.decide(
            listOf(0.4f, 0.7f, 0.68f, 0.66f, 0.64f, 0.62f),
        )
        assertEquals(QueryRoute.GROUNDED, decision.route)
        assertEquals(0.7f, decision.maxSimilarity, 0.0001f)
        assertEquals(0.68f, decision.secondSimilarity, 0.0001f)
        assertEquals(0.66f, decision.meanTop5Similarity, 0.0001f)
    }
}
