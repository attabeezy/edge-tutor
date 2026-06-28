package com.edgetutor.mnn.data.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface MessageDao {
    @Query("SELECT * FROM messages WHERE documentId = :documentId ORDER BY timestamp, id")
    fun observeForDocument(documentId: Long): Flow<List<MessageEntity>>

    @Query("SELECT * FROM messages WHERE documentId = :documentId ORDER BY timestamp, id")
    suspend fun getForDocument(documentId: Long): List<MessageEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(message: MessageEntity): Long

    @Update
    suspend fun update(message: MessageEntity)

    @Query("DELETE FROM messages WHERE documentId = :documentId")
    suspend fun deleteForDocument(documentId: Long)
}
