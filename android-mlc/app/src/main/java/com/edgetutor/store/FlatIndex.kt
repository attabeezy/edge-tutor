package com.edgetutor.store

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Brute-force cosine similarity search over an in-memory flat vector store.
 *
 * For MVP scale (~500–1000 chunks per textbook) this is fast (< 5 ms per query
 * on any modern ARM CPU) and requires no JNI build or NDK.
 *
 * Replace with FAISS JNI or Spotify Voyager if multi-document libraries or
 * cross-document search become a requirement.
 *
 * NOT thread-safe for writes; reads ([search]) are safe after ingestion completes.
 *
 * Supports two modes:
 * 1. Bulk mode: accumulate entries in RAM, then [save] once.
 * 2. Streaming mode: [startAppend], [append] each entry, [finishAppend].
 *    Streaming mode uses constant RAM during ingestion (spills to disk incrementally).
 */
class FlatIndex {

    class Entry(val id: Long, val text: String, val vector: FloatArray)

    private val entries = mutableListOf<Entry>()

    /** Streaming append state */
    private var appendChannel: RandomAccessFile? = null
    private var appendChannelStream: java.io.FileOutputStream? = null
    private var appendEntryCount: Int = 0
    private var appendDims: Int = 0
    private var appendFile: File? = null

    val size: Int get() = entries.size

    fun add(id: Long, text: String, vector: FloatArray) {
        entries.add(Entry(id, text, vector))
    }

    /**
     * Return the [k] entries whose vectors are most similar to [query].
     * [query] need not be pre-normalised; normalisation is applied here.
     */
    fun search(query: FloatArray, k: Int): List<Entry> {
        if (entries.isEmpty()) return emptyList()
        val qn = l2Norm(query)
        return entries
            .map { e -> e to dot(qn, e.vector) }
            .sortedByDescending { (_, sim) -> sim }
            .take(k)
            .map { (entry, _) -> entry }
    }

    // ---------------------------------------------------------------------------
    // Bulk persistence — write all entries at once
    // ---------------------------------------------------------------------------

    fun save(file: File) {
        val dims = if (entries.isEmpty()) 0 else entries[0].vector.size
        val floatBufBytes = dims * Float.SIZE_BYTES

        RandomAccessFile(file, "rw").use { raf ->
            raf.setLength(0)
            val ch = raf.channel

            val header = ByteBuffer.allocateDirect(12).order(ByteOrder.nativeOrder())
            header.putInt(FILE_VERSION).putInt(dims).putInt(entries.size).flip()
            ch.write(header)

            val floatBuf = ByteBuffer.allocateDirect(floatBufBytes).order(ByteOrder.nativeOrder())

            for (e in entries) {
                val textBytes = e.text.toByteArray(Charsets.UTF_8)
                val fixedBuf = ByteBuffer.allocateDirect(12).order(ByteOrder.nativeOrder())
                fixedBuf.putLong(e.id).putInt(textBytes.size).flip()
                ch.write(fixedBuf)
                ch.write(ByteBuffer.wrap(textBytes))
                floatBuf.clear()
                floatBuf.asFloatBuffer().put(e.vector)
                floatBuf.limit(floatBufBytes)
                ch.write(floatBuf)
            }
        }
    }

    fun load(file: File) {
        entries.clear()
        RandomAccessFile(file, "r").use { raf ->
            val ch = raf.channel

            val header = ByteBuffer.allocateDirect(12).order(ByteOrder.nativeOrder())
            ch.read(header); header.flip()
            val version = header.int
            check(version == FILE_VERSION) { "Unsupported index version $version" }
            val dims  = header.int
            val count = header.int

            val floatBufBytes = dims * Float.SIZE_BYTES
            val floatBuf = ByteBuffer.allocateDirect(floatBufBytes).order(ByteOrder.nativeOrder())

            repeat(count) {
                val fixedBuf = ByteBuffer.allocateDirect(12).order(ByteOrder.nativeOrder())
                ch.read(fixedBuf); fixedBuf.flip()
                val id      = fixedBuf.long
                val textLen = fixedBuf.int

                val textBuf = ByteBuffer.allocate(textLen)
                ch.read(textBuf)
                val text = String(textBuf.array(), Charsets.UTF_8)

                floatBuf.clear()
                ch.read(floatBuf)
                floatBuf.flip()
                val vector = FloatArray(dims)
                floatBuf.asFloatBuffer().get(vector)

                entries.add(Entry(id, text, vector))
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Streaming append — constant memory usage during ingestion
    //
    // Usage:
    //   index.startAppend(file, dims)
    //   for (chunk in chunks) {
    //       index.append(Entry(id, chunk.text, embedding))
    //   }
    //   index.finishAppend()
    //
    // The file header reserves space for dims and count; finishAppend() updates
    // the entry count after all data is written.
    // ---------------------------------------------------------------------------

    /**
     * Begin streaming append mode. Writes a placeholder header; call [finishAppend]
     * after all entries are appended to finalize the entry count.
     */
    fun startAppend(file: File, dims: Int) {
        if (appendChannel != null) error("Append already in progress")
        appendFile = file
        appendDims = dims
        appendEntryCount = 0

        val raf = RandomAccessFile(file, "rw")
        appendChannel = raf
        raf.setLength(0)

        // Write placeholder header (count=0 will be updated in finishAppend)
        val header = ByteBuffer.allocateDirect(12).order(ByteOrder.nativeOrder())
        header.putInt(FILE_VERSION).putInt(dims).putInt(0).flip()
        raf.channel.write(header)

        appendChannelStream = java.io.FileOutputStream(file, true)
    }

    /**
     * Append a single entry to the index file. Writes are buffered by the OS.
     * Thread-unsafe: call from a single thread during ingestion.
     */
    fun append(entry: Entry) {
        val stream = appendChannelStream ?: error("startAppend not called")
        val dims = appendDims
        val floatBufBytes = dims * Float.SIZE_BYTES

        val textBytes = entry.text.toByteArray(Charsets.UTF_8)
        val fixedBuf = ByteBuffer.allocate(12).order(ByteOrder.nativeOrder())
        fixedBuf.putLong(entry.id).putInt(textBytes.size).flip()
        stream.channel.write(fixedBuf)
        stream.channel.write(ByteBuffer.wrap(textBytes))

        val floatBuf = ByteBuffer.allocateDirect(floatBufBytes).order(ByteOrder.nativeOrder())
        floatBuf.asFloatBuffer().put(entry.vector)
        floatBuf.limit(floatBufBytes)
        stream.channel.write(floatBuf)

        appendEntryCount++
    }

    /**
     * Finalize the streaming append: update the entry count in the file header.
     * After this call, the index can be loaded via [load].
     */
    fun finishAppend() {
        val raf = appendChannel ?: error("startAppend not called")
        try {
            // Update entry count in header (offset 8, after version + dims).
            // Must use a native-endian ByteBuffer — raf.writeInt() is big-endian
            // but load() reads the header with ByteOrder.nativeOrder() (little-endian
            // on ARM), which would silently corrupt the count field.
            val countBuf = ByteBuffer.allocate(4).order(ByteOrder.nativeOrder())
            countBuf.putInt(appendEntryCount).flip()
            raf.channel.position(8)
            raf.channel.write(countBuf)
        } finally {
            raf.close()
            appendChannelStream?.close()
            appendChannel = null
            appendChannelStream = null
            appendFile = null
        }
    }

    // ---------------------------------------------------------------------------
    // Math helpers
    // ---------------------------------------------------------------------------

    private fun l2Norm(v: FloatArray): FloatArray {
        var norm = 0f
        for (x in v) norm += x * x
        norm = sqrt(norm)
        return if (norm < 1e-9f) v.copyOf() else FloatArray(v.size) { i -> v[i] / norm }
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }

    companion object {
        private const val FILE_VERSION = 1
    }
}
