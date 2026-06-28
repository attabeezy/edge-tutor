package com.edgetutor.mnn.data.db

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "messages",
    foreignKeys = [
        ForeignKey(
            entity = DocumentEntity::class,
            parentColumns = ["id"],
            childColumns = ["documentId"],
            onDelete = ForeignKey.CASCADE,
        ),
        ForeignKey(
            entity = ChatSessionEntity::class,
            parentColumns = ["id"],
            childColumns = ["sessionId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("documentId"), Index("sessionId")],
)
data class MessageEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val documentId: Long,
    val sessionId: Long,
    val role: String,
    val text: String,
    val thinking: String? = null,
    val imagePath: String? = null,
    val sourcesJson: String = "[]",
    val timestamp: Long = System.currentTimeMillis(),
    val completionState: String = "complete",
    val thinkingEnabled: Boolean = false,
    val promptTokens: Long = 0,
    val answerTokens: Long = 0,
    val thinkingTokens: Long = 0,
    val prefillUs: Long = 0,
    val decodeUs: Long = 0,
    val ttftMs: Long = -1,
    val thinkingDurationMs: Long = 0,
)
