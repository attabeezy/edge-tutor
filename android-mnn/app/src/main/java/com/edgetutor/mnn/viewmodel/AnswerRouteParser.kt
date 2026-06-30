package com.edgetutor.mnn.viewmodel

enum class QueryRoute { TEXTBOOK, GENERAL, FALLBACK_GENERAL }

data class RouteParseResult(
    val visibleText: String,
    val routeResolvedNow: Boolean,
)

/**
 * Removes the model's route marker while tokens stream.
 *
 * Initial chunks are buffered because native token boundaries can split a
 * marker anywhere. Any malformed or missing marker fails closed to a visibly
 * labelled general answer with no textbook attribution.
 */
class AnswerRouteParser {
    companion object {
        const val TEXTBOOK_MARKER = "[TEXTBOOK]"
        const val GENERAL_MARKER = "[GENERAL]"
        const val GENERAL_LABEL =
            "Model knowledge (not from textbook; may be inaccurate): "

        private val MARKERS = listOf(TEXTBOOK_MARKER, GENERAL_MARKER)
        private const val MAX_PREFIX_CHARS = 32
    }

    private val pending = StringBuilder()
    var route: QueryRoute? = null
        private set
    var markerValid: Boolean = false
        private set

    fun consume(chunk: String): RouteParseResult {
        if (chunk.isEmpty()) return RouteParseResult("", false)
        if (route != null) return RouteParseResult(chunk, false)
        pending.append(chunk)
        return resolveIfPossible(final = false)
    }

    fun finish(): RouteParseResult =
        if (route == null) resolveIfPossible(final = true)
        else RouteParseResult("", false)

    private fun resolveIfPossible(final: Boolean): RouteParseResult {
        val raw = pending.toString()
        val leadingWhitespace = raw.length - raw.trimStart().length
        val candidate = raw.substring(leadingWhitespace)
        val complete = MARKERS.firstOrNull { candidate.startsWith(it) }
        if (complete != null) {
            route = if (complete == TEXTBOOK_MARKER) QueryRoute.TEXTBOOK else QueryRoute.GENERAL
            markerValid = true
            val answer = candidate.removePrefix(complete).trimStart()
            pending.clear()
            val visible = if (route == QueryRoute.GENERAL) GENERAL_LABEL + answer else answer
            return RouteParseResult(visible, true)
        }

        val couldBecomeMarker = MARKERS.any { it.startsWith(candidate) }
        if (!final && couldBecomeMarker && candidate.length <= MAX_PREFIX_CHARS) {
            return RouteParseResult("", false)
        }

        route = QueryRoute.FALLBACK_GENERAL
        markerValid = false
        pending.clear()
        return RouteParseResult(GENERAL_LABEL + candidate, true)
    }
}
