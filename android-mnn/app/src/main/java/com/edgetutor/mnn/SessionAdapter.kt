package com.edgetutor.mnn

import android.text.format.DateUtils
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.edgetutor.mnn.data.db.SessionListItem
import com.edgetutor.mnn.databinding.ItemSessionBinding

class SessionAdapter(
    private val onOpen: (SessionListItem) -> Unit,
    private val onDelete: (SessionListItem) -> Unit,
) : ListAdapter<SessionListItem, SessionAdapter.Holder>(Diff) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): Holder =
        Holder(ItemSessionBinding.inflate(LayoutInflater.from(parent.context), parent, false))

    override fun onBindViewHolder(holder: Holder, position: Int) = holder.bind(getItem(position))

    inner class Holder(private val binding: ItemSessionBinding) : RecyclerView.ViewHolder(binding.root) {
        fun bind(item: SessionListItem) = with(binding) {
            sessionTitle.text = item.title
            val time = DateUtils.getRelativeTimeSpanString(
                item.updatedAt, System.currentTimeMillis(), DateUtils.MINUTE_IN_MILLIS,
            )
            sessionSubtitle.text = listOfNotNull(item.documentName, time).joinToString(" · ")
            root.setOnClickListener { onOpen(item) }
            sessionDelete.setOnClickListener { onDelete(item) }
        }
    }

    private object Diff : DiffUtil.ItemCallback<SessionListItem>() {
        override fun areItemsTheSame(old: SessionListItem, new: SessionListItem) = old.id == new.id
        override fun areContentsTheSame(old: SessionListItem, new: SessionListItem) = old == new
    }
}
