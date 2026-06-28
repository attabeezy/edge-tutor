package com.edgetutor.mnn.viewmodel

data class PromptBudgetPolicy(
    val id: String,
    val maxKeptChunks: Int,
    val maxCharsPerChunk: Int,
) {
    init {
        require(id.isNotBlank())
        require(maxKeptChunks > 0)
        require(maxCharsPerChunk > 0)
    }

    companion object {
        val BASELINE = PromptBudgetPolicy("2x800", 2, 800)
        val REDUCED = PromptBudgetPolicy("2x500", 2, 500)
        val SINGLE_FULL = PromptBudgetPolicy("1x800", 1, 800)
        val SINGLE_REDUCED = PromptBudgetPolicy("1x500", 1, 500)

        val BENCHMARK_POLICIES = listOf(BASELINE, REDUCED, SINGLE_FULL, SINGLE_REDUCED)

        // Selected production default. Change only after a device report satisfies
        // the latency and quality gates documented in PROJECT.md.
        val DEFAULT = SINGLE_FULL
    }
}
