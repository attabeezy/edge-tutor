package com.edgetutor.mnn.ui

import androidx.annotation.Nullable
import io.noties.markwon.ext.latex.JLatexMathNode
import io.noties.markwon.inlineparser.InlineProcessor
import org.commonmark.node.Node
import java.util.regex.Pattern

class LatexInlineProcessor : InlineProcessor() {
    override fun specialCharacter(): Char = '$'

    @Nullable
    override fun parse(): Node? {
        val match = match(RE) ?: return null
        val node = JLatexMathNode()
        node.latex(match.substring(1, match.length - 1))
        return node
    }

    private companion object {
        private val RE = Pattern.compile("(?<!\\$)\\$(?!\\$)([\\s\\S]+?)(?<!\\$)\\$(?!\\$)")
    }
}
