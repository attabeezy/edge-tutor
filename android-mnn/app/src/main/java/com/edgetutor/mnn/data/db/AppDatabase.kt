package com.edgetutor.mnn.data.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(entities = [DocumentEntity::class, MessageEntity::class], version = 3, exportSchema = false)
@TypeConverters(Converters::class)
abstract class AppDatabase : RoomDatabase() {

    abstract fun documentDao(): DocumentDao
    abstract fun messageDao(): MessageDao

    companion object {
        @Volatile private var INSTANCE: AppDatabase? = null

        fun get(context: Context): AppDatabase = INSTANCE ?: synchronized(this) {
            INSTANCE ?: Room.databaseBuilder(
                context.applicationContext,
                AppDatabase::class.java,
                "edgetutor_mnn.db",   // separate DB from the ltk variant
            )
                .addMigrations(MIGRATION_2_3)
                .build().also { INSTANCE = it }
        }

        val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    """CREATE TABLE IF NOT EXISTS `messages` (
                        `id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        `documentId` INTEGER NOT NULL,
                        `role` TEXT NOT NULL,
                        `text` TEXT NOT NULL,
                        `thinking` TEXT,
                        `imagePath` TEXT,
                        `sourcesJson` TEXT NOT NULL,
                        `timestamp` INTEGER NOT NULL,
                        `completionState` TEXT NOT NULL,
                        `thinkingEnabled` INTEGER NOT NULL,
                        `promptTokens` INTEGER NOT NULL,
                        `answerTokens` INTEGER NOT NULL,
                        `thinkingTokens` INTEGER NOT NULL,
                        `prefillUs` INTEGER NOT NULL,
                        `decodeUs` INTEGER NOT NULL,
                        `ttftMs` INTEGER NOT NULL,
                        `thinkingDurationMs` INTEGER NOT NULL,
                        FOREIGN KEY(`documentId`) REFERENCES `documents`(`id`) ON UPDATE NO ACTION ON DELETE CASCADE
                    )""".trimIndent()
                )
                db.execSQL("CREATE INDEX IF NOT EXISTS `index_messages_documentId` ON `messages` (`documentId`)")
            }
        }
    }
}
