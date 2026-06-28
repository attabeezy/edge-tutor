package com.edgetutor.mnn.llm

import com.google.gson.JsonParser

object MnnThinkingPolicy {
    const val disabledConfigJson =
        """{"jinja":{"context":{"enable_thinking":false}}}"""

    fun isDisabled(effectiveConfigJson: String): Boolean =
        runCatching {
            !JsonParser.parseString(effectiveConfigJson)
                .asJsonObject
                .getAsJsonObject("jinja")
                .getAsJsonObject("context")
                .get("enable_thinking")
                .asBoolean
        }.getOrDefault(false)

    fun isHiddenOnlyAnswer(visibleAnswer: String, capturedThinking: String): Boolean =
        visibleAnswer.isBlank() && capturedThinking.isNotBlank()
}
