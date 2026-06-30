package com.edgetutor.mnn.viewmodel

object AnswerAttributionPolicy {
    fun isGeneralKnowledgeAnswer(text: String): Boolean =
        text.trimStart().startsWith(AnswerRouteParser.GENERAL_LABEL, ignoreCase = true)
}
