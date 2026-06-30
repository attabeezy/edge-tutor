package com.edgetutor.mnn.perf

import com.edgetutor.mnn.viewmodel.AnswerAttributionPolicy
import com.edgetutor.mnn.viewmodel.QueryRoute
import java.io.File

enum class ValidationCategory {
    GROUNDED,
    FOLLOW_UP,
    UNSUPPORTED_ACADEMIC,
    NON_ACADEMIC,
    TUTOR_DIAGNOSTIC,
    TUTOR_HINT,
    TUTOR_CORRECTION,
    TUTOR_EXPLANATION,
    TUTOR_UNDERSTANDING_CHECK,
    TUTOR_COMPLETE_ANSWER,
}

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
    val route: QueryRoute = QueryRoute.TEXTBOOK,
    val routeReason: String = "",
    val routeMarkerValid: Boolean = false,
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

        ValidationCase("td-math", ValidationCategory.TUTOR_DIAGNOSTIC, "I think 0.35 is greater than 0.8 because 35 is greater than 8. Help me find my mistake without giving the answer.", expectedEvidence = "Diagnoses place-value misconception and asks one question"),
        ValidationCase("td-science", ValidationCategory.TUTOR_DIAGNOSTIC, "I think heavier objects always fall faster because gravity pulls harder on them. Help me examine that idea without giving the conclusion.", expectedEvidence = "Diagnoses force-versus-acceleration misconception"),
        ValidationCase("td-english", ValidationCategory.TUTOR_DIAGNOSTIC, "I think every group of words with a verb is a complete sentence. Help me test that idea.", expectedEvidence = "Diagnoses complete-thought misconception"),
        ValidationCase("td-social", ValidationCategory.TUTOR_DIAGNOSTIC, "I think a primary source is always more accurate than a secondary source. Help me examine that claim.", expectedEvidence = "Diagnoses reliability-versus-source-type misconception"),

        ValidationCase("th-math", ValidationCategory.TUTOR_HINT, "Give me one hint, not the answer: solve 4x + 7 = 31.", expectedEvidence = "Inverse-operation hint and one question"),
        ValidationCase("th-science", ValidationCategory.TUTOR_HINT, "Give me one hint, not the full answer: why does the Moon appear to have phases?", expectedEvidence = "Guides attention to illuminated portion"),
        ValidationCase("th-english", ValidationCategory.TUTOR_HINT, "Give me one hint: a character keeps practicing after repeated failures and finally succeeds. What could the theme be?", expectedEvidence = "Guides from topic to general message"),
        ValidationCase("th-social", ValidationCategory.TUTOR_HINT, "Give me one hint, not the answer: what usually happens to price when supply falls but demand stays the same?", expectedEvidence = "Guides comparison of buyers and available goods"),

        ValidationCase("tc-math", ValidationCategory.TUTOR_CORRECTION, "I solved 2x + 6 = 18 by dividing 18 by 2 first. Correct my first step without finishing the problem.", expectedEvidence = "Corrects operation order without final x"),
        ValidationCase("tc-science", ValidationCategory.TUTOR_CORRECTION, "For a 20 g object with volume 5 cubic centimeters, I calculated density as 5 divided by 20. Correct me without giving the final number.", expectedEvidence = "Corrects density formula without final value"),
        ValidationCase("tc-english", ValidationCategory.TUTOR_CORRECTION, "I called 'Because the storm ended.' a complete sentence because it has a subject and verb. Correct my reasoning without rewriting it for me.", expectedEvidence = "Explains dependent-clause problem"),
        ValidationCase("tc-social", ValidationCategory.TUTOR_CORRECTION, "I said wages earned are the opportunity cost of working instead of attending a game. Correct my reasoning without giving the final answer.", expectedEvidence = "Redirects to best forgone alternative"),

        ValidationCase("te-math", ValidationCategory.TUTOR_EXPLANATION, "Explain equivalent fractions in simple terms, then check my understanding.", expectedEvidence = "Simple explanation and one check"),
        ValidationCase("te-science", ValidationCategory.TUTOR_EXPLANATION, "Explain photosynthesis for a beginner, then check my understanding.", expectedEvidence = "Level-appropriate explanation and one check"),
        ValidationCase("te-english", ValidationCategory.TUTOR_EXPLANATION, "Explain metaphors for a beginner, then check my understanding.", expectedEvidence = "Level-appropriate explanation and one check"),
        ValidationCase("te-social", ValidationCategory.TUTOR_EXPLANATION, "Explain checks and balances for a beginner, then check my understanding.", expectedEvidence = "Level-appropriate explanation and one check"),

        ValidationCase("tu-math", ValidationCategory.TUTOR_UNDERSTANDING_CHECK, "I think multiplying a fraction's numerator and denominator by the same nonzero number keeps its value. Check my understanding with one question.", expectedEvidence = "Confirms idea and asks application question"),
        ValidationCase("tu-science", ValidationCategory.TUTOR_UNDERSTANDING_CHECK, "I think acceleration means any object is moving fast. Check my understanding with one question.", expectedEvidence = "Distinguishes speed from velocity change"),
        ValidationCase("tu-english", ValidationCategory.TUTOR_UNDERSTANDING_CHECK, "I think a theme is just a one-word topic. Check my understanding with one question.", expectedEvidence = "Distinguishes theme from topic"),
        ValidationCase("tu-social", ValidationCategory.TUTOR_UNDERSTANDING_CHECK, "I think federalism and separation of powers mean the same thing. Check my understanding with one question.", expectedEvidence = "Distinguishes levels from branches"),

        ValidationCase("ta-math", ValidationCategory.TUTOR_COMPLETE_ANSWER, "I tried subtracting 7 from both sides of 4x + 7 = 31 and got 4x = 24. Show me the complete solution now.", expectedEvidence = "Complete solution x = 6 and one check"),
        ValidationCase("ta-science", ValidationCategory.TUTOR_COMPLETE_ANSWER, "I tried explaining seasons using Earth's distance from the Sun, but I am stuck. Give me the complete explanation now.", expectedEvidence = "Complete axial-tilt explanation and one check"),
        ValidationCase("ta-english", ValidationCategory.TUTOR_COMPLETE_ANSWER, "I drafted 'School lunches are a topic' as my thesis. Show me a complete improved thesis and explain the change.", expectedEvidence = "Arguable thesis with explanation and one check"),
        ValidationCase("ta-social", ValidationCategory.TUTOR_COMPLETE_ANSWER, "I compared two accounts of an event but do not know how to handle their disagreement. Give me a complete method now.", expectedEvidence = "Corroboration method and one check"),
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
    appendLine("case_id,category,policy,answer_route,route_reason,route_marker_valid,max_similarity,second_similarity,mean_top5_similarity,question,answer,sources,prompt_chars,prompt_tokens,prefill_us,decode_us,visible_ttft_ms,total_ms,available_memory_mb,is_general,error,correctness_0_2,grounding_0_2,relevance_0_2,clarity_0_2,tutor_move_0_2,answer_restraint_0_2,adaptation_0_2,corrective_feedback_0_2")
    for (r in this@toCsv) {
        appendLine(listOf(
            r.caseId, r.category, r.policyId, r.route, r.routeReason,
            r.routeMarkerValid, r.maxSimilarity, r.secondSimilarity,
            r.meanTop5Similarity, r.question, r.answer,
            r.sources.joinToString(" | "), r.promptChars, r.promptTokens, r.prefillUs,
            r.decodeUs, r.visibleTtftMs, r.totalMs, r.availableMemoryMb,
            r.isGeneralAnswer, r.error, "", "", "", "", "", "", "", "",
        ).joinToString(",") { csvCell(it.toString()) })
    }
}

fun List<ValidationResult>.toMarkdown(): String = buildString {
    appendLine("# EdgeTutor Query Validation")
    appendLine()
    appendLine("Complete the eight 0-2 rubric columns after reviewing each answer.")
    appendLine()
    appendLine("| Case | Category | Policy | Route | Marker valid | Top-1 | Top-2 | Mean top-5 | TTFT ms | Total ms | General | Sources | C | G | R | Cl | TM | AR | A | CF |")
    appendLine("|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (r in this@toMarkdown) {
        appendLine("| ${r.caseId} | ${r.category} | ${r.policyId} | ${r.route} | ${r.routeMarkerValid} | ${"%.4f".format(r.maxSimilarity)} | ${"%.4f".format(r.secondSimilarity)} | ${"%.4f".format(r.meanTop5Similarity)} | ${r.visibleTtftMs} | ${r.totalMs} | ${r.isGeneralAnswer} | ${r.sources.size} |  |  |  |  |  |  |  |  |")
        appendLine()
        appendLine("**Question:** ${r.question}")
        appendLine()
        appendLine("**Answer:** ${r.answer.ifBlank { r.error }}")
        appendLine()
    }
}

private fun csvCell(value: String): String = "\"${value.replace("\"", "\"\"")}\""
