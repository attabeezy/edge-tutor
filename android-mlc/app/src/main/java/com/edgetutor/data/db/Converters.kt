package com.edgetutor.data.db

import androidx.room.TypeConverter

class Converters {
    @TypeConverter fun fromStatus(s: IngestionStatus): String = s.name
    @TypeConverter fun toStatus(s: String): IngestionStatus = IngestionStatus.valueOf(s)
}
