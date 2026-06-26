package com.edgetutor.mnn.perf

import android.content.Context
import com.edgetutor.mnn.llm.MnnEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

data class MnnBenchmarkConfig(
    val promptChars: List<Int> = listOf(128, 512),
    val repeatCount: Int = 3,
    val promptSeed: String = "Explain calculus in one concise paragraph. ",
)

data class MnnBenchmarkResult(
    val promptChars: Int,
    val repeatIndex: Int,
    val nativePromptTokens: Long,
    val nativeDecodeTokens: Long,
    val nativePrefillUs: Long,
    val nativeDecodeUs: Long,
    val visibleTtftMs: Long,
    val totalMs: Long,
    val outputChars: Int,
)

class MnnBenchmark(private val context: Context) {
    suspend fun run(config: MnnBenchmarkConfig = MnnBenchmarkConfig()): List<MnnBenchmarkResult> =
        withContext(Dispatchers.IO) {
            val engine = MnnEngine(context)
            try {
                engine.copyModelIfNeeded()
                engine.initNativeModel()
                engine.warmUp()
                val results = mutableListOf<MnnBenchmarkResult>()
                for (chars in config.promptChars) {
                    val prompt = buildPrompt(config.promptSeed, chars)
                    repeat(config.repeatCount) { index ->
                        var firstVisibleMs = -1L
                        val startNs = System.nanoTime()
                        val output = engine.generate(prompt) { token ->
                            if (firstVisibleMs < 0 && token.isNotBlank()) {
                                firstVisibleMs = EdgeTutorPerf.elapsedMs(startNs)
                            }
                        }
                        results += MnnBenchmarkResult(
                            promptChars = chars,
                            repeatIndex = index + 1,
                            nativePromptTokens = -1L,
                            nativeDecodeTokens = -1L,
                            nativePrefillUs = -1L,
                            nativeDecodeUs = -1L,
                            visibleTtftMs = firstVisibleMs,
                            totalMs = EdgeTutorPerf.elapsedMs(startNs),
                            outputChars = output.length,
                        )
                    }
                }
                results
            } finally {
                engine.close()
            }
        }

    suspend fun writeReports(
        outputDir: File,
        config: MnnBenchmarkConfig = MnnBenchmarkConfig(),
    ): List<MnnBenchmarkResult> {
        val results = run(config)
        withContext(Dispatchers.IO) {
            outputDir.mkdirs()
            File(outputDir, "android-mnn-benchmark.csv").writeText(results.toCsv())
            File(outputDir, "android-mnn-benchmark.md").writeText(results.toMarkdown())
        }
        return results
    }

    private fun buildPrompt(seed: String, targetChars: Int): String {
        val sb = StringBuilder(targetChars)
        while (sb.length < targetChars) sb.append(seed)
        return sb.take(targetChars).toString()
    }
}

fun List<MnnBenchmarkResult>.toCsv(): String {
    val rows = this
    return buildString {
        appendLine("prompt_chars,repeat,visible_ttft_ms,total_ms,output_chars")
        rows.forEach { r ->
            appendLine("${r.promptChars},${r.repeatIndex},${r.visibleTtftMs},${r.totalMs},${r.outputChars}")
        }
    }
}

fun List<MnnBenchmarkResult>.toMarkdown(): String {
    val rows = this
    return buildString {
        appendLine("# Android MNN Benchmark")
        appendLine()
        appendLine("| Prompt chars | Repeat | Visible TTFT ms | Total ms | Output chars |")
        appendLine("|---:|---:|---:|---:|---:|")
        rows.forEach { r ->
            appendLine("| ${r.promptChars} | ${r.repeatIndex} | ${r.visibleTtftMs} | ${r.totalMs} | ${r.outputChars} |")
        }
    }
}
