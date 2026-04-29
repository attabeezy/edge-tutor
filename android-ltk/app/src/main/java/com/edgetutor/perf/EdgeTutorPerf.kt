package com.edgetutor.perf

import android.app.ActivityManager
import android.content.Context
import android.util.Log

object EdgeTutorPerf {

    private const val TAG = "EdgeTutorPerf"

    fun log(event: String, vararg fields: Pair<String, Any?>) {
        val payload = linkedMapOf<String, Any?>("event" to event)
        fields.forEach { (key, value) -> payload[key] = value }
        Log.d(TAG, payload.entries.joinToString(" ") { (key, value) -> "$key=${formatValue(value)}" })
    }

    fun snapshot(context: Context, event: String, vararg fields: Pair<String, Any?>) {
        val runtime = Runtime.getRuntime()
        val heapUsedMb = (runtime.totalMemory() - runtime.freeMemory()) / MB
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val info = ActivityManager.MemoryInfo().also(am::getMemoryInfo)
        log(
            event,
            *fields,
            "avail_mem_mb" to (info.availMem / MB),
            "heap_used_mb" to heapUsedMb,
        )
    }

    fun elapsedMs(startNs: Long): Long = (System.nanoTime() - startNs) / 1_000_000

    inline fun <T> trace(event: String, vararg fields: Pair<String, Any?>, block: () -> T): T {
        val startNs = System.nanoTime()
        return try {
            block()
        } finally {
            val durationMs = (System.nanoTime() - startNs) / 1_000_000
            log(event, *fields, "duration_ms" to durationMs)
        }
    }

    suspend inline fun <T> traceSuspend(
        event: String,
        vararg fields: Pair<String, Any?>,
        crossinline block: suspend () -> T,
    ): T {
        val startNs = System.nanoTime()
        return try {
            block()
        } finally {
            val durationMs = (System.nanoTime() - startNs) / 1_000_000
            log(event, *fields, "duration_ms" to durationMs)
        }
    }

    private fun formatValue(value: Any?): String = when (value) {
        null -> "null"
        is String -> value.replace("\\s+".toRegex(), "_")
        else -> value.toString()
    }

    private const val MB = 1024L * 1024L
}
