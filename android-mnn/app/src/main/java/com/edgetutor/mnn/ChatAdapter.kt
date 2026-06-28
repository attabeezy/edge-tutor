package com.edgetutor.mnn

import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.view.isVisible
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.edgetutor.mnn.databinding.ItemMessageAssistantBinding
import com.edgetutor.mnn.databinding.ItemMessageUserBinding
import com.edgetutor.mnn.ui.LatexInlineProcessor
import com.edgetutor.mnn.viewmodel.ChatMessage
import com.edgetutor.mnn.viewmodel.Role
import io.noties.markwon.Markwon
import io.noties.markwon.ext.latex.JLatexMathPlugin
import io.noties.markwon.ext.tables.TablePlugin
import io.noties.markwon.inlineparser.MarkwonInlineParserPlugin
import java.io.File

class ChatAdapter : ListAdapter<ChatMessage, RecyclerView.ViewHolder>(Diff) {
    override fun getItemViewType(position: Int) =
        if (getItem(position).role == Role.USER) USER else ASSISTANT

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        return if (viewType == USER) {
            UserHolder(ItemMessageUserBinding.inflate(inflater, parent, false))
        } else {
            AssistantHolder(ItemMessageAssistantBinding.inflate(inflater, parent, false))
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        when (holder) {
            is UserHolder -> holder.bind(getItem(position))
            is AssistantHolder -> holder.bind(getItem(position))
        }
    }

    class UserHolder(private val binding: ItemMessageUserBinding) : RecyclerView.ViewHolder(binding.root) {
        fun bind(item: ChatMessage) = with(binding) {
            message.text = item.text
            image.isVisible = item.imagePath != null
            item.imagePath?.let {
                image.setImageURI(Uri.fromFile(File(it)))
                image.setOnClickListener { _ -> FullScreenImageDialog.show(image.context, it) }
            }
        }
    }

    class AssistantHolder(private val binding: ItemMessageAssistantBinding) : RecyclerView.ViewHolder(binding.root) {
        private val markwon = Markwon.builder(binding.message.context)
            .usePlugin(MarkwonInlineParserPlugin.create { it.addInlineProcessor(LatexInlineProcessor()) })
            .usePlugin(TablePlugin.create(binding.message.context))
            .usePlugin(JLatexMathPlugin.create(binding.message.textSize, binding.message.textSize) {
                it.inlinesEnabled(true)
            })
            .build()

        fun bind(item: ChatMessage) = with(binding) {
            if (item.text.isEmpty()) {
                message.text = if (item.completionState == "streaming") "Thinking…" else ""
            } else {
                markwon.setMarkdown(message, markdownForRender(item.text, item.completionState == "streaming"))
            }
            thinkingGroup.isVisible = !item.thinking.isNullOrBlank()
            thinking.text = item.thinking
            thinkingHeader.setOnClickListener {
                thinking.visibility = if (thinking.isVisible) View.GONE else View.VISIBLE
                thinkingHeader.text = if (thinking.isVisible) "Thought process  ▾" else "Thought process  ▸"
            }
            sources.isVisible = item.sources.isNotEmpty()
            sources.text = item.sources.joinToString("\n\n") { "Textbook source\n$it" }
            metrics.isVisible = item.metricsText != null
            metrics.text = item.metricsText
        }

        private fun markdownForRender(raw: String, streaming: Boolean): String {
            var text = raw
                .replace("\\(", "$")
                .replace("\\)", "$")
                .replace("\\[", "$$")
                .replace("\\]", "$$")
            // Recover common model output that omitted Markdown math delimiters.
            text = Regex("""(?<![$\\])\\frac\{[^{}\n]+\}\{[^{}\n]+\}""")
                .replace(text) { "$${it.value}$" }
            text = Regex("""(?<![$\\])\\(?:nabla|int|sum)\b(?:\s+\\(?:cdot|mathbf\{[^}\n]+\})|\s+[A-Za-z0-9_^{}]+){0,4}""")
                .replace(text) { "$${it.value.trim()}$" }
            if (streaming) {
                val displayCount = Regex("(?<!\\\\)\\$\\$").findAll(text).count()
                if (displayCount % 2 != 0) text += "$$"
                val withoutDisplay = text.replace("$$", "")
                val inlineCount = Regex("(?<!\\\\)\\$").findAll(withoutDisplay).count()
                if (inlineCount % 2 != 0) text += "$"
                val fenceCount = Regex("```").findAll(text).count()
                if (fenceCount % 2 != 0) text += "\n```"
            }
            return text
        }
    }

    private object Diff : DiffUtil.ItemCallback<ChatMessage>() {
        override fun areItemsTheSame(old: ChatMessage, new: ChatMessage) = old.id == new.id
        override fun areContentsTheSame(old: ChatMessage, new: ChatMessage) = old == new
    }

    private companion object {
        const val USER = 1
        const val ASSISTANT = 2
    }
}
