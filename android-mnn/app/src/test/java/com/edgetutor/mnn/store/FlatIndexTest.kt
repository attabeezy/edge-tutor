package com.edgetutor.mnn.store

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

class FlatIndexTest {
    @Test
    fun `save and load preserves searchable entries`() {
        val file = File.createTempFile("flat-index", ".idx")
        val index = FlatIndex()
        index.add(1, "calculus", floatArrayOf(1f, 0f))
        index.add(2, "history", floatArrayOf(0f, 1f))
        index.save(file)

        val loaded = FlatIndex()
        loaded.load(file)
        val result = loaded.search(floatArrayOf(1f, 0f), k = 1)

        assertEquals(1L, result.single().id)
        assertEquals("calculus", result.single().text)
    }

    @Test
    fun `streaming append writes loadable index`() {
        val file = File.createTempFile("flat-index-stream", ".idx")
        val index = FlatIndex()
        index.startAppend(file, dims = 2)
        index.append(FlatIndex.Entry(1, "first", floatArrayOf(1f, 0f)))
        index.append(FlatIndex.Entry(2, "second", floatArrayOf(0f, 1f)))
        index.finishAppend()

        val loaded = FlatIndex()
        loaded.load(file)

        assertEquals(2, loaded.size)
        assertEquals(2L, loaded.search(floatArrayOf(0f, 1f), k = 1).single().id)
    }
}
