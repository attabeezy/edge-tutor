package com.edgetutor.mnn.llm

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

enum class ModelReadinessKind {
    MISSING,
    IMPORTING,
    READY,
    INCOMPLETE,
    ERROR,
}

data class ModelReadinessState(
    val kind: ModelReadinessKind,
    val modelDir: String? = null,
    val missingFiles: List<String> = emptyList(),
    val message: String? = null,
    val progressCurrent: Int = 0,
    val progressTotal: Int = 0,
) {
    val isReady: Boolean get() = kind == ModelReadinessKind.READY

    companion object {
        val Missing = ModelReadinessState(ModelReadinessKind.MISSING)
        val Importing = ModelReadinessState(ModelReadinessKind.IMPORTING)
    }
}

object MnnModelManager {
    const val MODEL_DIR_NAME = "mnn_model"
    const val CONFIG_FILE = "config.json"

    val REQUIRED_FILES = listOf(
        "config.json",
        "llm_config.json",
        "tokenizer.txt",
        "llm.mnn",
        "llm.mnn.json",
        "llm.mnn.weight",
        "visual.mnn",
        "visual.mnn.weight",
    )

    fun internalModelDir(context: Context): File =
        File(context.filesDir, MODEL_DIR_NAME)

    fun appExternalModelDir(context: Context): File? =
        context.getExternalFilesDir(null)?.let { File(it, MODEL_DIR_NAME) }

    fun resolveReadyModelDir(context: Context): File? {
        val internal = internalModelDir(context)
        if (validateModelDir(internal).isReady) return internal
        val external = appExternalModelDir(context)
        if (external != null && validateModelDir(external).isReady) return external
        return null
    }

    fun validate(context: Context): ModelReadinessState {
        val internal = internalModelDir(context)
        val internalState = validateModelDir(internal)
        if (internalState.isReady) return internalState

        val external = appExternalModelDir(context)
        if (external != null) {
            val externalState = validateModelDir(external)
            if (externalState.isReady) return externalState
        }

        return if (internal.exists() || external?.exists() == true) {
            internalState.copy(
                kind = ModelReadinessKind.INCOMPLETE,
                message = "Model files are incomplete.",
            )
        } else {
            ModelReadinessState.Missing
        }
    }

    fun validateModelDir(modelDir: File): ModelReadinessState {
        if (!modelDir.exists()) {
            return ModelReadinessState(
                kind = ModelReadinessKind.MISSING,
                modelDir = modelDir.absolutePath,
                missingFiles = REQUIRED_FILES,
            )
        }
        val missing = REQUIRED_FILES.filterNot { File(modelDir, it).isFile }
        return if (missing.isEmpty()) {
            ModelReadinessState(
                kind = ModelReadinessKind.READY,
                modelDir = modelDir.absolutePath,
            )
        } else {
            ModelReadinessState(
                kind = ModelReadinessKind.INCOMPLETE,
                modelDir = modelDir.absolutePath,
                missingFiles = missing,
                message = "Missing ${missing.size} required model file(s).",
            )
        }
    }

    suspend fun importFromTreeUri(
        context: Context,
        treeUri: Uri,
        onProgress: (ModelReadinessState) -> Unit,
    ): ModelReadinessState = withContext(Dispatchers.IO) {
        try {
            onProgress(ModelReadinessState.Importing)
            val root = requireNotNull(
                androidx.documentfile.provider.DocumentFile.fromTreeUri(context, treeUri),
            ) { "Cannot open selected model folder." }
            val targetDir = internalModelDir(context)
            targetDir.mkdirs()

            REQUIRED_FILES.forEachIndexed { index, name ->
                val source = root.findFile(name)
                    ?: return@withContext ModelReadinessState(
                        kind = ModelReadinessKind.INCOMPLETE,
                        modelDir = targetDir.absolutePath,
                        missingFiles = REQUIRED_FILES.drop(index),
                        message = "Selected folder is missing $name.",
                    )
                val target = File(targetDir, name)
                onProgress(
                    ModelReadinessState(
                        kind = ModelReadinessKind.IMPORTING,
                        modelDir = targetDir.absolutePath,
                        message = "Copying $name",
                        progressCurrent = index,
                        progressTotal = REQUIRED_FILES.size,
                    ),
                )
                context.contentResolver.openInputStream(source.uri)?.use { input ->
                    target.outputStream().use { output -> input.copyTo(output) }
                } ?: return@withContext ModelReadinessState(
                    kind = ModelReadinessKind.ERROR,
                    modelDir = targetDir.absolutePath,
                    message = "Cannot read $name.",
                )
            }

            val state = validateModelDir(targetDir)
            onProgress(state)
            state
        } catch (e: Exception) {
            ModelReadinessState(
                kind = ModelReadinessKind.ERROR,
                message = e.message ?: e.javaClass.simpleName,
            )
        }
    }
}
