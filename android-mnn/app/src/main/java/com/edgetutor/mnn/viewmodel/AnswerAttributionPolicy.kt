package com.edgetutor.mnn.viewmodel

object AnswerAttributionPolicy {
    const val GENERAL_ANSWER_PREFIX = "The textbook does not cover this. General answer:"

    fun isGeneralKnowledgeAnswer(text: String): Boolean =
        text.trimStart().startsWith(GENERAL_ANSWER_PREFIX, ignoreCase = true)
}
