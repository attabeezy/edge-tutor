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
import com.edgetutor.mnn.viewmodel.GenerationProgressText
import com.edgetutor.mnn.viewmodel.Role
import io.noties.markwon.Markwon
import io.noties.markwon.ext.latex.JLatexMathPlugin
import io.noties.markwon.ext.tables.TablePlugin
import io.noties.markwon.inlineparser.MarkwonInlineParserPlugin
import java.io.File

class ChatAdapter : ListAdapter<ChatMessage, RecyclerView.ViewHolder>(Diff) {
    /** Message ids whose retrieved-chunks (sources) box is expanded. Collapsed by default. */
    private val expandedSources = mutableSetOf<Long>()

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

    inner class AssistantHolder(private val binding: ItemMessageAssistantBinding) : RecyclerView.ViewHolder(binding.root) {
        private val markwon = Markwon.builder(binding.message.context)
            .usePlugin(MarkwonInlineParserPlugin.create { it.addInlineProcessor(LatexInlineProcessor()) })
            .usePlugin(TablePlugin.create(binding.message.context))
            .usePlugin(JLatexMathPlugin.create(binding.message.textSize, binding.message.textSize) {
                it.inlinesEnabled(true)
            })
            .build()

        fun bind(item: ChatMessage) = with(binding) {
            val terminalStatus = when (item.completionState) {
                "stopped" -> "Generation stopped"
                "error" -> "Couldn’t generate response"
                else -> null
            }
            generationStatusGroup.isVisible = item.generationProgress != null || terminalStatus != null
            generationSpinner.isVisible = item.generationProgress != null
            generationStatus.text = item.generationProgress?.let(GenerationProgressText::format)
                ?: terminalStatus.orEmpty()
            message.isVisible = item.text.isNotEmpty()
            if (item.text.isNotEmpty()) {
                markwon.setMarkdown(message, markdownForRender(item.text, item.completionState == "streaming"))
            } else {
                message.text = ""
            }
            thinkingGroup.isVisible = !item.thinking.isNullOrBlank()
            thinking.text = item.thinking
            thinkingHeader.setOnClickListener {
                thinking.visibility = if (thinking.isVisible) View.GONE else View.VISIBLE
                thinkingHeader.text = if (thinking.isVisible) "Thinking Process  ▾" else "Thinking Process  ▸"
            }
            val hasSources = item.sources.isNotEmpty()
            sourcesHeader.isVisible = hasSources
            sources.text = item.sources.joinToString("\n\n") { "Textbook source\n$it" }
            val expanded = item.id in expandedSources
            sources.isVisible = hasSources && expanded
            sourcesHeader.text = if (expanded) "Hide sources  ▾" else "Show sources  ▸"
            sourcesHeader.setOnClickListener {
                if (item.id in expandedSources) expandedSources.remove(item.id)
                else expandedSources.add(item.id)
                notifyItemChanged(bindingAdapterPosition)
            }
            metrics.isVisible = item.metricsText != null
            metrics.text = item.metricsText
        }

        private fun markdownForRender(raw: String, streaming: Boolean): String {
            // Normalize TeX-style delimiters, then keep only genuine formulas as math
            // so prose the model wrongly wrapped in $...$ renders as plain text.
            var text = raw
                .replace("\\(", "$")
                .replace("\\)", "$")
                .replace("\\[", "$$")
                .replace("\\]", "$$")
            text = sanitizeMath(text)

            // While streaming, only balance open delimiters so partially received
            // math renders in place (cue from MNN Chat's preprocessStreamingMarkdown).
            // The heavier recovery of un-delimited LaTeX is deferred to the final
            // render to avoid the raw-then-rendered flicker as tokens complete.
            if (streaming) return balanceOpenDelimiters(text)

            // Final render: recover common model output that omitted $...$ delimiters.
            return text
                .let { Regex("""(?<![$\\])\\frac\{[^{}\n]+\}\{[^{}\n]+\}""").replace(it) { m -> "$${m.value}$" } }
                .let {
                    Regex("""(?<![$\\])\\(?:nabla|int|sum)\b(?:\s+\\(?:cdot|mathbf\{[^}\n]+\})|\s+[A-Za-z0-9_^{}]+){0,4}""")
                        .replace(it) { m -> "$${m.value.trim()}$" }
                }
        }

        /**
         * EdgeTutor's small model frequently wraps ordinary prose in $...$ / $$...$$.
         * In math mode LaTeX collapses spaces, so such prose renders as a spaceless
         * serif run. MNN Chat avoids this by relying on clean model output; since ours
         * is noisier, we render a segment as LaTeX only when it actually looks like a
         * formula and otherwise strip it back to plain text.
         */
        private fun sanitizeMath(text: String): String {
            var out = DISPLAY_MATH.replace(text) { m ->
                if (looksLikeProse(m.groupValues[1], display = true)) stripToProse(m.groupValues[1]) else m.value
            }
            out = INLINE_MATH.replace(out) { m ->
                if (looksLikeProse(m.groupValues[1], display = false)) stripToProse(m.groupValues[1]) else m.value
            }
            return out
        }

        /** A math segment is treated as prose if it carries real words rather than a formula. */
        private fun looksLikeProse(inner: String, display: Boolean): Boolean {
            if (inner.contains("\\text{")) return true
            val proseOnly = inner.replace(Regex("""\\[A-Za-z]+"""), " ").trim()
            val words = Regex("""[A-Za-z]{3,}""").findAll(proseOnly).count()
            val letters = proseOnly.count { it.isLetter() }
            return if (display) words >= 4 || letters > 80 || proseOnly.contains('\n')
                   else words >= 2 || letters > 24
        }

        private fun stripToProse(inner: String): String =
            Regex("""\\text\{([^{}]*)\}""").replace(inner) { it.groupValues[1] }
                .replace(Regex("""\\[A-Za-z]+\{[^{}]*\}"""), "")  // \hphantom{#}, \mathbf{x}, ...
                .replace(Regex("""\\[A-Za-z]+"""), "")            // bare \alpha, \circ, ...
                .replace(Regex("""\\(?![A-Za-z])"""), "")         // lone backslashes
                .replace(Regex(""" {2,}"""), " ")
                .trim()

        private fun balanceOpenDelimiters(raw: String): String {
            var text = raw
            val displayCount = Regex("(?<!\\\\)\\$\\$").findAll(text).count()
            if (displayCount % 2 != 0) text += "$$"
            val withoutDisplay = text.replace("$$", "")
            val inlineCount = Regex("(?<!\\\\)\\$").findAll(withoutDisplay).count()
            if (inlineCount % 2 != 0) text += "$"
            val fenceCount = Regex("```").findAll(text).count()
            if (fenceCount % 2 != 0) text += "\n```"
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

        // $$...$$ display math, and single $...$ inline math (not part of $$).
        val DISPLAY_MATH = Regex("\\\$\\\$(.+?)\\\$\\\$", RegexOption.DOT_MATCHES_ALL)
        val INLINE_MATH = Regex("(?<!\\\$)\\\$(?!\\\$)(.+?)(?<!\\\$)\\\$(?!\\\$)", RegexOption.DOT_MATCHES_ALL)
    }
}
