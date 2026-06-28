package com.edgetutor.mnn.data.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface ChatSessionDao {

    @Query(
        """SELECT s.id AS id, s.documentId AS documentId, s.title AS title,
                  s.updatedAt AS updatedAt, d.displayName AS documentName
           FROM chat_sessions s
           LEFT JOIN documents d ON s.documentId = d.id
           ORDER BY s.updatedAt DESC"""
    )
    fun observeAll(): Flow<List<SessionListItem>>

    @Insert
    suspend fun insert(session: ChatSessionEntity): Long

    @Query("SELECT * FROM chat_sessions WHERE id = :id")
    suspend fun getById(id: Long): ChatSessionEntity?

    @Query("SELECT * FROM chat_sessions WHERE documentId = :documentId ORDER BY updatedAt DESC LIMIT 1")
    suspend fun latestForDocument(documentId: Long): ChatSessionEntity?

    @Query("UPDATE chat_sessions SET title = :title, updatedAt = :updatedAt WHERE id = :id")
    suspend fun updateMeta(id: Long, title: String, updatedAt: Long)

    @Query("UPDATE chat_sessions SET updatedAt = :updatedAt WHERE id = :id")
    suspend fun touch(id: Long, updatedAt: Long)

    @Query("SELECT COUNT(*) FROM messages WHERE sessionId = :sessionId")
    suspend fun messageCount(sessionId: Long): Int

    @Query("DELETE FROM chat_sessions WHERE id = :id")
    suspend fun delete(id: Long)
}
