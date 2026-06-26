package com.edgetutor.mnn.perf

import android.app.ActivityManager
import android.content.Context
import android.util.Log

/**
 * Lightweight structured performance logger.
 *
 * Identical contract to android-ltk's EdgeTutorPerf — same log tag, same
 * key/value format — so existing Logcat filters work unchanged:
 *
 *   adb logcat EdgeTutorPerf:D MnnEngine:D AndroidRuntime:E *:S
 */
object EdgeTutorPerf {

    private const val TAG = "EdgeTutorPerf"

    /** Log a structured performance event with arbitrary key/value pairs. */
    fun log(event: String, vararg pairs: Pair<String, Any?>) {
        val sb = StringBuilder(event)
        for ((k, v) in pairs) {
            sb.append(" $k=$v")
        }
        Log.d(TAG, sb.toString())
    }

    /** Log a memory snapshot for [event] using the system ActivityManager. */
    fun snapshot(context: Context, event: String, vararg pairs: Pair<String, Any?>) {
        val am   = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val info = ActivityManager.MemoryInfo()
        am.getMemoryInfo(info)
        val availMb  = info.availMem  / (1024 * 1024)
        val totalMb  = info.totalMem  / (1024 * 1024)
        val lowMemory = info.lowMemory
        log(event, *pairs, "avail_mem_mb" to availMb, "total_mem_mb" to totalMb, "low_memory" to lowMemory)
    }

    /**
     * Inline trace helper: logs [event] with [pairs] plus elapsed_ms,
     * then returns the block result.
     */
    inline fun <T> trace(event: String, vararg pairs: Pair<String, Any?>, block: () -> T): T {
        val startNs = System.nanoTime()
        val result  = block()
        log(event, *pairs, "elapsed_ms" to elapsedMs(startNs))
        return result
    }

    /**
     * Suspend trace helper — same as [trace] but for suspend functions.
     */
    suspend inline fun <T> traceSuspend(
        event: String,
        vararg pairs: Pair<String, Any?>,
        crossinline block: suspend () -> T,
    ): T {
        val startNs = System.nanoTime()
        val result  = block()
        log(event, *pairs, "elapsed_ms" to elapsedMs(startNs))
        return result
    }

    /** Returns milliseconds elapsed since [startNs] (a [System.nanoTime] value). */
    fun elapsedMs(startNs: Long): Long = (System.nanoTime() - startNs) / 1_000_000
}
