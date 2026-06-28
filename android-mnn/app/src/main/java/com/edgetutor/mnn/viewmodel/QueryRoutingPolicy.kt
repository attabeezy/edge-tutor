package com.edgetutor.mnn.viewmodel

enum class QueryRoute { GROUNDED, GENERAL }

data class QueryRouteDecision(
    val route: QueryRoute,
    val reason: String,
    val maxSimilarity: Float,
    val secondSimilarity: Float,
    val meanTop5Similarity: Float,
)

/**
 * Routes a query from semantic retrieval confidence, never lexical overlap.
 *
 * The default threshold is intentionally isolated here because cosine-score
 * distributions depend on the embedding model and document corpus. Device
 * validation reports must be used to calibrate it for Arctic Embed XS.
 */
object QueryRoutingPolicy {
    const val REQUIRED_SIMILARITY_COUNT = 5
    const val MIN_MEAN_TOP5_SIMILARITY = 0.63165f

    fun decide(similarities: List<Float>): QueryRouteDecision {
        val sorted = similarities.sortedDescending()
        if (sorted.size < REQUIRED_SIMILARITY_COUNT) {
            return QueryRouteDecision(
                QueryRoute.GENERAL,
                "insufficient_similarity_distribution",
                sorted.getOrNull(0) ?: Float.NEGATIVE_INFINITY,
                sorted.getOrNull(1) ?: Float.NEGATIVE_INFINITY,
                Float.NEGATIVE_INFINITY,
            )
        }
        val meanTop5 = sorted.take(REQUIRED_SIMILARITY_COUNT).average().toFloat()
        val grounded = meanTop5 >= MIN_MEAN_TOP5_SIMILARITY
        return QueryRouteDecision(
            if (grounded) QueryRoute.GROUNDED else QueryRoute.GENERAL,
            if (grounded) "mean_top5_threshold_met" else "mean_top5_threshold_not_met",
            sorted[0],
            sorted[1],
            meanTop5,
        )
    }
}
