package com.edgetutor.mnn.perf

import com.edgetutor.mnn.viewmodel.AnswerAttributionPolicy
import com.edgetutor.mnn.viewmodel.QueryRoute
import com.edgetutor.mnn.viewmodel.QueryRoutingPolicy
import java.io.File

enum class ValidationCategory { GROUNDED, FOLLOW_UP, UNSUPPORTED_ACADEMIC, NON_ACADEMIC }

data class ValidationCase(
    val id: String,
    val category: ValidationCategory,
    val question: String,
    val startsNewConversation: Boolean = true,
    val expectedEvidence: String,
    val setupQuestion: String? = null,
)

data class ValidationResult(
    val caseId: String,
    val category: ValidationCategory,
    val policyId: String,
    val question: String,
    val answer: String,
    val sources: List<String>,
    val promptChars: Int,
    val promptTokens: Long,
    val prefillUs: Long,
    val decodeUs: Long,
    val visibleTtftMs: Long,
    val totalMs: Long,
    val availableMemoryMb: Long,
    val route: QueryRoute = QueryRoute.GROUNDED,
    val routeReason: String = "",
    val maxSimilarity: Float = 0f,
    val secondSimilarity: Float = 0f,
    val meanTop5Similarity: Float = 0f,
    val error: String = "",
) {
    val isGeneralAnswer: Boolean
        get() = AnswerAttributionPolicy.isGeneralKnowledgeAnswer(answer)
}

object EdgeTutorValidationSuite {
    val cases = listOf(
        ValidationCase("g1", ValidationCategory.GROUNDED, "What is calculus?", expectedEvidence = "Definition of calculus"),
        ValidationCase("g2", ValidationCategory.GROUNDED, "What is a differential?", expectedEvidence = "Differential or derivative definition"),
        ValidationCase("g3", ValidationCategory.GROUNDED, "Explain integration in simple terms.", expectedEvidence = "Integration explanation"),
        ValidationCase("g4", ValidationCategory.GROUNDED, "Give a small worked example of differentiation.", expectedEvidence = "Worked derivative"),
        ValidationCase("f1", ValidationCategory.FOLLOW_UP, "Show me an example of that.", false, "Example of differentiation", "What is differentiation?"),
        ValidationCase("f2", ValidationCategory.FOLLOW_UP, "Can you explain it more simply?", false, "Simplified power rule", "Explain the power rule."),
        ValidationCase("f3", ValidationCategory.FOLLOW_UP, "How is it reversed?", false, "Integration reverses differentiation", "What is differentiation?"),
        ValidationCase("f4", ValidationCategory.FOLLOW_UP, "Give another example.", false, "Second relevant calculus example", "Give an example of integration."),
        ValidationCase("ua1", ValidationCategory.UNSUPPORTED_ACADEMIC, "What causes a solar eclipse?", expectedEvidence = "General-answer label"),
        ValidationCase("ua2", ValidationCategory.UNSUPPORTED_ACADEMIC, "Explain photosynthesis.", expectedEvidence = "General-answer label"),
        ValidationCase("ua3", ValidationCategory.UNSUPPORTED_ACADEMIC, "Who wrote Things Fall Apart?", expectedEvidence = "General-answer label"),
        ValidationCase("ua4", ValidationCategory.UNSUPPORTED_ACADEMIC, "What is the capital of Japan?", expectedEvidence = "General-answer label"),
        ValidationCase("na1", ValidationCategory.NON_ACADEMIC, "How do I bake bread?", expectedEvidence = "General-answer label"),
        ValidationCase("na2", ValidationCategory.NON_ACADEMIC, "Write a short birthday greeting.", expectedEvidence = "General-answer label"),
        ValidationCase("na3", ValidationCategory.NON_ACADEMIC, "What should I pack for a picnic?", expectedEvidence = "General-answer label"),
        ValidationCase("na4", ValidationCategory.NON_ACADEMIC, "Tell me a clean joke.", expectedEvidence = "General-answer label"),
    )

    fun writeReports(outputDir: File, results: List<ValidationResult>): Pair<File, File> {
        outputDir.mkdirs()
        val csv = File(outputDir, "query-validation.csv")
        val markdown = File(outputDir, "query-validation.md")
        csv.writeText(results.toCsv())
        markdown.writeText(results.toMarkdown())
        return csv to markdown
    }
}

fun List<ValidationResult>.toCsv(): String = buildString {
    appendLine("case_id,category,policy,route,route_reason,max_similarity,second_similarity,mean_top5_similarity,mean_top5_threshold,question,answer,sources,prompt_chars,prompt_tokens,prefill_us,decode_us,visible_ttft_ms,total_ms,available_memory_mb,is_general,error,correctness_0_2,grounding_0_2,relevance_0_2,clarity_0_2")
    for (r in this@toCsv) {
        appendLine(listOf(
            r.caseId, r.category, r.policyId, r.route, r.routeReason, r.maxSimilarity,
            r.secondSimilarity, r.meanTop5Similarity,
            QueryRoutingPolicy.MIN_MEAN_TOP5_SIMILARITY, r.question, r.answer,
            r.sources.joinToString(" | "), r.promptChars, r.promptTokens, r.prefillUs,
            r.decodeUs, r.visibleTtftMs, r.totalMs, r.availableMemoryMb,
            r.isGeneralAnswer, r.error, "", "", "", "",
        ).joinToString(",") { csvCell(it.toString()) })
    }
}

fun List<ValidationResult>.toMarkdown(): String = buildString {
    appendLine("# EdgeTutor Query Validation")
    appendLine()
    appendLine("Complete the four 0-2 rubric columns after reviewing each answer.")
    appendLine()
    appendLine("| Case | Category | Policy | Route | Top-1 | Top-2 | Mean top-5 | TTFT ms | Total ms | General | Sources | C | G | R | Cl |")
    appendLine("|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for (r in this@toMarkdown) {
        appendLine("| ${r.caseId} | ${r.category} | ${r.policyId} | ${r.route} | ${"%.4f".format(r.maxSimilarity)} | ${"%.4f".format(r.secondSimilarity)} | ${"%.4f".format(r.meanTop5Similarity)} | ${r.visibleTtftMs} | ${r.totalMs} | ${r.isGeneralAnswer} | ${r.sources.size} |  |  |  |  |")
        appendLine()
        appendLine("**Question:** ${r.question}")
        appendLine()
        appendLine("**Answer:** ${r.answer.ifBlank { r.error }}")
        appendLine()
    }
}

private fun csvCell(value: String): String = "\"${value.replace("\"", "\"\"")}\""
