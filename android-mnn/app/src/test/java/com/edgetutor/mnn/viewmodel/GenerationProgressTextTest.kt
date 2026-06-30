package com.edgetutor.mnn.viewmodel

import org.junit.Assert.assertEquals
import org.junit.Test

class GenerationProgressTextTest {
    @Test
    fun searchingLabelHidesShortElapsedTime() {
        assertEquals(
            "Searching your textbook…",
            GenerationProgressText.format(
                GenerationProgress(GenerationPhase.SEARCHING_TEXTBOOK, elapsedSeconds = 2),
            ),
        )
    }

    @Test
    fun searchingLabelShowsElapsedTimeAtThreeSeconds() {
        assertEquals(
            "Searching your textbook… · 3s",
            GenerationProgressText.format(
                GenerationProgress(GenerationPhase.SEARCHING_TEXTBOOK, elapsedSeconds = 3),
            ),
        )
    }

    @Test
    fun nativePrefillUsesReadingContextLabel() {
        assertEquals(
            "Reading selected passages on device… · 15s",
            GenerationProgressText.format(
                GenerationProgress(GenerationPhase.READING_CONTEXT, elapsedSeconds = 15),
            ),
        )
    }
}
