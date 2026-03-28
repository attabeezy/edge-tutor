package com.edgetutor

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.edgetutor.data.db.DocumentEntity
import com.edgetutor.data.db.IngestionStatus
import com.edgetutor.ingestion.PdfExtractor
import com.edgetutor.viewmodel.ChatViewModel
import com.edgetutor.viewmodel.IngestViewModel
import com.edgetutor.viewmodel.Role

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // PdfBox requires one-time resource init
        PdfExtractor.init(this)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    EdgeTutorApp()
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Root composable — Phase 3 smoke-test UI
// Replace in Phase 4 with proper Library / Upload / Chat / Settings screens.
// ---------------------------------------------------------------------------

@Composable
fun EdgeTutorApp(
    ingestVm: IngestViewModel = viewModel(),
    chatVm:   ChatViewModel   = viewModel(),
) {
    val context    = LocalContext.current
    val documents  by ingestVm.documents.collectAsState()
    val messages   by chatVm.messages.collectAsState()
    val thinking   by chatVm.isThinking.collectAsState()
    val warmingUp  by chatVm.isWarmingUp.collectAsState()

    var question  by remember { mutableStateOf("") }
    var activeDoc by remember { mutableStateOf<DocumentEntity?>(null) }

    // File picker — restricted to PDFs
    val picker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri ?: return@rememberLauncherForActivityResult
        val name = uri.lastPathSegment ?: "document.pdf"
        ingestVm.ingest(uri, name)
    }

    Column(modifier = Modifier.fillMaxSize().padding(12.dp)) {

        // ── Document list ──────────────────────────────────────────────────
        Text("Documents", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(4.dp))

        LazyColumn(modifier = Modifier.weight(0.3f)) {
            items(documents) { doc ->
                DocumentRow(
                    doc       = doc,
                    isActive  = doc.id == activeDoc?.id,
                    onSelect  = {
                        activeDoc = doc
                        chatVm.loadDocument(doc)
                    },
                    onDelete  = { ingestVm.delete(doc) },
                )
            }
        }

        Button(onClick = { picker.launch(arrayOf("application/pdf", "text/plain")) }) {
            Text("+ Add document")
        }

        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

        // ── Chat ──────────────────────────────────────────────────────────
        Text(
            text  = when {
                activeDoc == null -> "Select a document above to start chatting"
                warmingUp         -> "Loading model for ${activeDoc!!.displayName}…"
                else              -> "Chat — ${activeDoc!!.displayName}"
            },
            style = MaterialTheme.typography.titleMedium,
        )
        Spacer(Modifier.height(4.dp))

        LazyColumn(modifier = Modifier.weight(0.6f)) {
            items(messages) { msg ->
                val prefix = if (msg.role == Role.USER) "You: " else "EdgeTutor: "
                Text(
                    text     = prefix + msg.text,
                    modifier = Modifier.padding(vertical = 2.dp),
                )
                if (msg.sources.isNotEmpty()) {
                    Text(
                        text  = "  [Source: ${msg.sources.first()}]",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.outline,
                    )
                }
            }
            if (thinking) {
                item { Text("EdgeTutor is thinking…") }
            }
        }

        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value         = question,
                onValueChange = { question = it },
                label         = { Text("Ask a question") },
                modifier      = Modifier.weight(1f),
                enabled       = activeDoc != null && !thinking && !warmingUp,
            )
            Spacer(Modifier.width(8.dp))
            Button(
                onClick  = { chatVm.ask(question); question = "" },
                enabled  = question.isNotBlank() && activeDoc != null && !thinking && !warmingUp,
            ) { Text("Ask") }
        }
    }
}

@Composable
private fun DocumentRow(
    doc:      DocumentEntity,
    isActive: Boolean,
    onSelect: () -> Unit,
    onDelete: () -> Unit,
) {
    val statusLabel = when (doc.status) {
        IngestionStatus.PENDING -> "pending…"
        IngestionStatus.RUNNING -> "indexing…"
        IngestionStatus.DONE    -> "${doc.chunkCount} chunks"
        IngestionStatus.ERROR   -> "error"
    }

    Row(
        modifier            = Modifier.fillMaxWidth().padding(vertical = 2.dp),
        verticalAlignment   = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text  = doc.displayName,
                style = if (isActive) MaterialTheme.typography.bodyMedium
                        else MaterialTheme.typography.bodySmall,
            )
            Text(
                text  = statusLabel,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.outline,
            )
        }
        if (doc.status == IngestionStatus.DONE) {
            TextButton(onClick = onSelect) { Text("Open") }
        }
        TextButton(onClick = onDelete) { Text("Delete") }
    }
}
