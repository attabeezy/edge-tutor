package com.edgetutor.store

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.File
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
 */
class FlatIndex {

    class Entry(val id: Long, val text: String, val vector: FloatArray)

    private val entries = mutableListOf<Entry>()

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
    // Persistence  —  JSON via Gson (sufficient for ~1000 chunks × 384 floats ≈ 6 MB)
    // ---------------------------------------------------------------------------

    private data class Row(val id: Long, val text: String, val v: List<Float>)

    fun save(file: File) {
        val rows = entries.map { Row(it.id, it.text, it.vector.toList()) }
        file.writeText(Gson().toJson(rows))
    }

    fun load(file: File) {
        val type = object : TypeToken<List<Row>>() {}.type
        val rows: List<Row> = Gson().fromJson(file.readText(), type)
        entries.clear()
        rows.forEach { r -> entries.add(Entry(r.id, r.text, r.v.toFloatArray())) }
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

    /** Dot product — assumes [b] is already L2-normalised (Embedder output is). */
    private fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }
}
