package com.edgetutor.mnn.llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.io.path.createTempDirectory

class MnnModelManagerTest {
    @Test
    fun `validateModelDir reports missing directory`() {
        val dir = createTempDirectory().toFile().resolve("missing")

        val state = MnnModelManager.validateModelDir(dir)

        assertEquals(ModelReadinessKind.MISSING, state.kind)
        assertEquals(MnnModelManager.REQUIRED_FILES, state.missingFiles)
    }

    @Test
    fun `validateModelDir reports incomplete model`() {
        val dir = createTempDirectory().toFile()
        File(dir, "config.json").writeText("{}")

        val state = MnnModelManager.validateModelDir(dir)

        assertEquals(ModelReadinessKind.INCOMPLETE, state.kind)
        assertTrue("llm.mnn.weight" in state.missingFiles)
    }

    @Test
    fun `validateModelDir reports ready when required files exist`() {
        val dir = createTempDirectory().toFile()
        MnnModelManager.REQUIRED_FILES.forEach { File(dir, it).writeText("x") }

        val state = MnnModelManager.validateModelDir(dir)

        assertEquals(ModelReadinessKind.READY, state.kind)
    }
}
