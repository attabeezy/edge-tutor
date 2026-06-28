package com.edgetutor.mnn.viewmodel

enum class QueryRoute { GROUNDED, GENERAL }

data class QueryRouteDecision(
    val route: QueryRoute,
    val reason: String,
    val maxSimilarity: Float,
)

/**
 * Routes a query from semantic retrieval confidence, never lexical overlap.
 *
 * The default threshold is intentionally isolated here because cosine-score
 * distributions depend on the embedding model and document corpus. Device
 * validation reports must be used to calibrate it for Arctic Embed XS.
 */
object QueryRoutingPolicy {
    const val MIN_GROUNDED_COSINE_SIMILARITY = 0.35f

    fun decide(maxSimilarity: Float?): QueryRouteDecision {
        val score = maxSimilarity ?: Float.NEGATIVE_INFINITY
        return if (score >= MIN_GROUNDED_COSINE_SIMILARITY) {
            QueryRouteDecision(QueryRoute.GROUNDED, "semantic_threshold_met", score)
        } else {
            QueryRouteDecision(QueryRoute.GENERAL, "semantic_threshold_not_met", score)
        }
    }

    fun inheritFollowUp(
        isFollowUp: Boolean,
        previousAnswerWasGrounded: Boolean?,
        maxSimilarity: Float?,
    ): QueryRouteDecision {
        if (!isFollowUp || previousAnswerWasGrounded == null) return decide(maxSimilarity)
        return QueryRouteDecision(
            route = if (previousAnswerWasGrounded) QueryRoute.GROUNDED else QueryRoute.GENERAL,
            reason = if (previousAnswerWasGrounded) {
                "followup_inherited_grounded"
            } else {
                "followup_inherited_general"
            },
            maxSimilarity = maxSimilarity ?: Float.NEGATIVE_INFINITY,
        )
    }
}
