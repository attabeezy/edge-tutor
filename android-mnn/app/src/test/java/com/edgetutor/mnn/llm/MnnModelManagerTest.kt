package com.edgetutor.mnn.llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
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

    @Test
    fun `installModelFiles stages and activates a complete bundled model`() {
        val root = createTempDirectory().toFile()
        val target = File(root, MnnModelManager.MODEL_DIR_NAME)
        File(target, "config.json").apply {
            parentFile?.mkdirs()
            writeText("incomplete")
        }
        val progress = mutableListOf<ModelReadinessState>()

        val state = MnnModelManager.installModelFiles(
            targetDir = target,
            openSource = { name -> ByteArrayInputStream("bundled:$name".toByteArray()) },
            onProgress = progress::add,
        )

        assertEquals(ModelReadinessKind.READY, state.kind)
        assertEquals("bundled:config.json", File(target, "config.json").readText())
        assertEquals(MnnModelManager.REQUIRED_FILES.size, progress.count {
            it.kind == ModelReadinessKind.IMPORTING
        })
        assertEquals(ModelReadinessKind.READY, progress.last().kind)
        assertTrue(!File(root, "${MnnModelManager.MODEL_DIR_NAME}.installing").exists())
    }

    @Test
    fun `installModelFiles skips an existing ready model`() {
        val target = createTempDirectory().toFile()
        MnnModelManager.REQUIRED_FILES.forEach { File(target, it).writeText("existing") }
        var sourceOpened = false

        val state = MnnModelManager.installModelFiles(
            targetDir = target,
            openSource = {
                sourceOpened = true
                ByteArrayInputStream(byteArrayOf())
            },
        )

        assertEquals(ModelReadinessKind.READY, state.kind)
        assertTrue(!sourceOpened)
        assertEquals("existing", File(target, "config.json").readText())
    }

    @Test
    fun `installModelFiles preserves an incomplete model when a bundled asset is missing`() {
        val root = createTempDirectory().toFile()
        val target = File(root, MnnModelManager.MODEL_DIR_NAME)
        File(target, "config.json").apply {
            parentFile?.mkdirs()
            writeText("keep-me")
        }

        val state = MnnModelManager.installModelFiles(
            targetDir = target,
            openSource = { name ->
                if (name == "llm.mnn.weight") error("missing bundled asset")
                ByteArrayInputStream(name.toByteArray())
            },
        )

        assertEquals(ModelReadinessKind.ERROR, state.kind)
        assertTrue(state.message.orEmpty().contains("missing bundled asset"))
        assertEquals("keep-me", File(target, "config.json").readText())
        assertTrue(!File(root, "${MnnModelManager.MODEL_DIR_NAME}.installing").exists())
    }
}
