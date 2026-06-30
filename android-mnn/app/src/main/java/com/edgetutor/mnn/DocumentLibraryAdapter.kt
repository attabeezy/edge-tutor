package com.edgetutor.mnn

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.core.view.isVisible
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.edgetutor.mnn.data.db.DocumentEntity
import com.edgetutor.mnn.data.db.IngestionStatus
import com.edgetutor.mnn.databinding.ItemLibraryDocumentBinding

data class LibraryDocumentItem(
    val document: DocumentEntity,
    val statusText: String,
    val selected: Boolean,
)

class DocumentLibraryAdapter(
    private val onOpen: (DocumentEntity) -> Unit,
    private val onDelete: (DocumentEntity) -> Unit,
) : ListAdapter<LibraryDocumentItem, DocumentLibraryAdapter.Holder>(Diff) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): Holder =
        Holder(
            ItemLibraryDocumentBinding.inflate(
                LayoutInflater.from(parent.context),
                parent,
                false,
            ),
        )

    override fun onBindViewHolder(holder: Holder, position: Int) =
        holder.bind(getItem(position))

    inner class Holder(
        private val binding: ItemLibraryDocumentBinding,
    ) : RecyclerView.ViewHolder(binding.root) {
        fun bind(item: LibraryDocumentItem) = with(binding) {
            documentName.text = item.document.displayName
            documentStatus.text = item.statusText
            selectedLabel.isVisible = item.selected
            documentProgress.isVisible = item.document.status == IngestionStatus.RUNNING
            root.strokeWidth = if (item.selected) 3 else 1
            root.alpha = if (item.document.status == IngestionStatus.DONE) 1f else 0.78f
            root.setOnClickListener {
                if (item.document.status == IngestionStatus.DONE) onOpen(item.document)
            }
            deleteDocument.setOnClickListener { onDelete(item.document) }
        }
    }

    private object Diff : DiffUtil.ItemCallback<LibraryDocumentItem>() {
        override fun areItemsTheSame(old: LibraryDocumentItem, new: LibraryDocumentItem) =
            old.document.id == new.document.id

        override fun areContentsTheSame(old: LibraryDocumentItem, new: LibraryDocumentItem) =
            old == new
    }
}
