package com.edgetutor.data.db

import androidx.room.Entity
import androidx.room.PrimaryKey

enum class IngestionStatus { PENDING, RUNNING, DONE, ERROR }

@Entity(tableName = "documents")
data class DocumentEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val displayName: String,
    val uriString: String,
    val pageCount: Int = 0,
    val chunkCount: Int = 0,
    val status: IngestionStatus = IngestionStatus.PENDING,
    val errorMessage: String? = null,
    val addedAt: Long = System.currentTimeMillis(),
    /** True when PdfExtractor detected avg words/page < 100 — likely a scanned image PDF. */
    val isLikelyScanned: Boolean = false,
)
