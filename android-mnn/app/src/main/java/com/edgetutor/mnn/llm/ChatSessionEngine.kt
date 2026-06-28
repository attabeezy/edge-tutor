package com.edgetutor.mnn.llm

enum class ChatRole { SYSTEM, USER, ASSISTANT }

sealed interface ChatContentPart {
    data class Text(val value: String) : ChatContentPart
    data class Image(val localPath: String) : ChatContentPart
}

data class ChatRoleMessage(val role: ChatRole, val content: List<ChatContentPart>)
data class GenerationChunk(val answerDelta: String = "", val thinkingDelta: String = "")
enum class CompletionReason { EOP, CANCELLED, MAX_TOKENS, ERROR }

data class ChatGenerationResult(
    val answer: String,
    val thinking: String,
    val reason: CompletionReason,
    val metrics: GenerationMetrics,
)

interface ChatSessionEngine : AutoCloseable {
    suspend fun load()
    suspend fun generate(
        messages: List<ChatRoleMessage>,
        thinkingEnabled: Boolean,
        onChunk: (GenerationChunk) -> Boolean,
    ): ChatGenerationResult
    fun cancel()
    suspend fun reset()
}
