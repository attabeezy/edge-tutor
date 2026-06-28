package com.edgetutor.mnn.attachments

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileOutputStream
import java.util.UUID
import kotlin.math.max

class ImageAttachmentStore(private val context: Context) {
    companion object {
        const val MAX_SOURCE_BYTES = 10L * 1024L * 1024L
        const val MAX_LONG_EDGE = 1024
        private val ACCEPTED_TYPES = setOf("image/jpeg", "image/png", "image/webp")
    }

    private val root = File(context.filesDir, "attachments").apply { mkdirs() }

    fun newCameraFile(): File =
        File(root, "camera").apply { mkdirs() }
            .resolve("capture-${UUID.randomUUID()}.jpg")

    fun normalize(uri: Uri, documentId: Long): File {
        val mime = context.contentResolver.getType(uri)
        require(mime == null || mime in ACCEPTED_TYPES) { "Unsupported image type: ${mime ?: "unknown"}" }

        val source = File.createTempFile("source-", ".image", context.cacheDir)
        try {
            context.contentResolver.openInputStream(uri).use { input ->
                requireNotNull(input) { "Image could not be opened." }
                FileOutputStream(source).use { output ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    var total = 0L
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        total += read
                        require(total <= MAX_SOURCE_BYTES) { "Image exceeds the 10 MB limit." }
                        output.write(buffer, 0, read)
                    }
                }
            }
            return normalizeFile(source, documentId)
        } finally {
            source.delete()
        }
    }

    fun normalizeFile(source: File, documentId: Long): File {
        require(source.exists() && source.length() in 1..MAX_SOURCE_BYTES) {
            if (source.length() > MAX_SOURCE_BYTES) "Image exceeds the 10 MB limit." else "Image is missing or empty."
        }
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(source.absolutePath, bounds)
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "Image is corrupt or unsupported." }

        var sample = 1
        while (max(bounds.outWidth / sample, bounds.outHeight / sample) > MAX_LONG_EDGE * 2) sample *= 2
        val decoded = BitmapFactory.decodeFile(
            source.absolutePath,
            BitmapFactory.Options().apply { inSampleSize = sample },
        ) ?: throw IllegalArgumentException("Image is corrupt or unsupported.")

        val orientation = runCatching {
            ExifInterface(source).getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL,
            )
        }.getOrDefault(ExifInterface.ORIENTATION_NORMAL)
        val matrix = Matrix().apply {
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> postRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> postRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> postRotate(270f)
                ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> postScale(-1f, 1f)
                ExifInterface.ORIENTATION_FLIP_VERTICAL -> postScale(1f, -1f)
            }
        }
        val oriented = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
        if (oriented !== decoded) decoded.recycle()
        val scale = minOf(1f, MAX_LONG_EDGE.toFloat() / max(oriented.width, oriented.height))
        val normalized = if (scale < 1f) {
            Bitmap.createScaledBitmap(oriented, (oriented.width * scale).toInt(), (oriented.height * scale).toInt(), true)
                .also { if (it !== oriented) oriented.recycle() }
        } else oriented

        val directory = File(root, documentId.toString()).apply { mkdirs() }
        val target = File(directory, "${UUID.randomUUID()}.jpg")
        try {
            FileOutputStream(target).use {
                check(normalized.compress(Bitmap.CompressFormat.JPEG, 90, it)) { "Could not encode image." }
            }
        } finally {
            normalized.recycle()
        }
        return target
    }

    fun deleteForDocument(documentId: Long) {
        File(root, documentId.toString()).deleteRecursively()
    }
}
