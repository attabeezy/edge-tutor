package com.edgetutor

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.edgetutor.data.db.DocumentEntity
import com.edgetutor.data.db.IngestionStatus
import com.edgetutor.ingestion.PdfExtractor
import com.edgetutor.viewmodel.ChatMessage
import com.edgetutor.viewmodel.ChatViewModel
import com.edgetutor.viewmodel.IngestViewModel
import com.edgetutor.viewmodel.Role
import kotlinx.coroutines.delay

private val AppGreen = Color(0xFF2F7D4B)
private val AppGreenSoft = Color(0xFFDCEFD9)
private val AppGreenWash = Color(0xFFF4FBF3)
private val AppTextSoft = Color(0xFF5B6C5E)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        PdfExtractor.init(this)
        setContent {
            MaterialTheme(
                colorScheme = lightColorScheme(
                    primary = AppGreen,
                    onPrimary = Color.White,
                    primaryContainer = AppGreenSoft,
                    onPrimaryContainer = Color(0xFF173B24),
                    secondary = Color(0xFF5A8F67),
                    onSecondary = Color.White,
                    surface = Color.White,
                    surfaceContainer = Color(0xFFF8FCF7),
                    surfaceContainerHigh = Color(0xFFF1F7EF),
                    surfaceVariant = AppGreenWash,
                    onSurface = Color(0xFF152018),
                    onSurfaceVariant = AppTextSoft,
                    outline = Color(0xFFD6E5D3),
                    errorContainer = Color(0xFFFFE8E4),
                    onErrorContainer = Color(0xFF7A2010),
                ),
            ) {
                Surface(modifier = Modifier.fillMaxSize()) {
                    EdgeTutorApp()
                }
            }
        }
    }
}

@Composable
fun EdgeTutorApp(
    ingestVm: IngestViewModel = viewModel(),
    chatVm: ChatViewModel = viewModel(),
) {
    val documents by ingestVm.documents.collectAsState()
    val progress by ingestVm.progress.collectAsState()
    val messages by chatVm.messages.collectAsState()
    val thinking by chatVm.isThinking.collectAsState()
    val lastThinkingDurationMs by chatVm.lastThinkingDurationMs.collectAsState()
    val warmingUp by chatVm.isWarmingUp.collectAsState()
    val errorMsg by chatVm.errorMessage.collectAsState()
    val activeDocId by chatVm.activeDocumentId.collectAsState()
    var question by remember { mutableStateOf("") }

    val currentDoc = documents.firstOrNull()
    val readyDoc = currentDoc?.takeIf { it.status == IngestionStatus.DONE }
    val chatLocked = currentDoc == null || currentDoc.status != IngestionStatus.DONE || warmingUp
    val canSend = question.isNotBlank() && !chatLocked && !thinking
    val canAdd = question.isBlank() && !thinking

    val statusText = currentDoc?.let { doc ->
        when {
            doc.status == IngestionStatus.RUNNING -> progress[doc.id]?.let { p ->
                if (p.total > 0) "${p.phase.lowercase()} ${p.current}/${p.total}"
                else "${p.phase.lowercase()} ${p.current}"
            } ?: "indexing"
            warmingUp -> "warming up"
            doc.status == IngestionStatus.ERROR -> doc.errorMessage ?: "something went wrong"
            else -> "ready"
        }
    }

    val picker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
    ) { uri: Uri? ->
        uri ?: return@rememberLauncherForActivityResult
        val name = uri.lastPathSegment ?: "document.pdf"
        ingestVm.ingest(uri, name)
    }

    LaunchedEffect(readyDoc?.id, activeDocId) {
        if (readyDoc != null && readyDoc.id != activeDocId) {
            chatVm.loadDocument(readyDoc)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.surface)
            .statusBarsPadding()
            .navigationBarsPadding()
            .padding(horizontal = 18.dp, vertical = 14.dp),
    ) {
        HeaderBlock(title = currentDoc?.displayName ?: "EdgeTutor")

        Spacer(Modifier.height(18.dp))

        if (currentDoc != null) {
            DocumentStrip(
                doc = currentDoc,
                statusText = statusText ?: "ready",
                showCancel = currentDoc.status == IngestionStatus.RUNNING,
                onCancel = { ingestVm.cancelIngest(currentDoc.id) },
                onDelete = { ingestVm.delete(currentDoc) },
            )
            Spacer(Modifier.height(12.dp))
        }

        errorMsg?.let { msg ->
            ErrorLine(
                message = msg,
                onDismiss = { chatVm.clearError() },
            )
            Spacer(Modifier.height(12.dp))
        }

        ChatFeed(
            modifier = Modifier.weight(1f),
            currentDoc = currentDoc,
            messages = messages,
            thinking = thinking,
            lastThinkingDurationMs = lastThinkingDurationMs,
            warmingUp = warmingUp,
        )

        Spacer(Modifier.height(12.dp))
        HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.6f))
        Spacer(Modifier.height(10.dp))

        ComposerBar(
            value = question,
            onValueChange = { question = it },
            placeholder = if (currentDoc == null) "add a document to start" else "ask about this document",
            sendEnabled = canSend,
            addEnabled = canAdd,
            inputEnabled = !chatLocked && !thinking,
            onAdd = { picker.launch(arrayOf("application/pdf", "text/plain")) },
            onSend = {
                chatVm.ask(question)
                question = ""
            },
        )
    }
}

@Composable
private fun HeaderBlock(title: String) {
    Text(
        text = title,
        style = MaterialTheme.typography.headlineSmall,
        fontWeight = FontWeight.SemiBold,
        maxLines = 1,
        overflow = TextOverflow.Ellipsis,
        modifier = Modifier.fillMaxWidth(),
    )
}

@Composable
private fun DocumentStrip(
    doc: DocumentEntity,
    statusText: String,
    showCancel: Boolean,
    onCancel: () -> Unit,
    onDelete: () -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .background(
                    color = when (doc.status) {
                        IngestionStatus.DONE -> AppGreen
                        IngestionStatus.RUNNING -> Color(0xFF86B97C)
                        IngestionStatus.ERROR -> Color(0xFFC45D47)
                        IngestionStatus.PENDING -> Color(0xFF97A79A)
                    },
                    shape = CircleShape,
                ),
        )
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = doc.displayName,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Medium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                text = statusText,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        if (showCancel) {
            TextButton(onClick = onCancel, contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp)) {
                Text("stop", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        IconGlyphButton(
            text = "x",
            onClick = onDelete,
            enabled = true,
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun ErrorLine(message: String, onDismiss: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                color = MaterialTheme.colorScheme.errorContainer,
                shape = RoundedCornerShape(16.dp),
            )
            .padding(horizontal = 12.dp, vertical = 10.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = message,
            color = MaterialTheme.colorScheme.onErrorContainer,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.weight(1f),
        )
        TextButton(onClick = onDismiss, contentPadding = PaddingValues(horizontal = 6.dp, vertical = 0.dp)) {
            Text("dismiss", color = MaterialTheme.colorScheme.onErrorContainer)
        }
    }
}

@Composable
private fun ChatFeed(
    modifier: Modifier = Modifier,
    currentDoc: DocumentEntity?,
    messages: List<ChatMessage>,
    thinking: Boolean,
    lastThinkingDurationMs: Long?,
    warmingUp: Boolean,
) {
    if (currentDoc == null) {
        EmptyState(modifier = modifier.fillMaxWidth())
        return
    }

    LazyColumn(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        contentPadding = PaddingValues(vertical = 8.dp),
    ) {
        items(messages) { message ->
            MessageRow(msg = message)
        }
        if (thinking) {
            item {
                ThinkingNote()
            }
        } else if (lastThinkingDurationMs != null) {
            item {
                StatusNote("thought for ${formatElapsedDuration(lastThinkingDurationMs)}")
            }
        }
        if (warmingUp) {
            item {
                StatusNote("warming up...")
            }
        }
    }
}

@Composable
private fun EmptyState(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(horizontal = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "No document loaded",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Medium,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            text = "Use + to add one document. Once it is ready, use > to send a question.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
    }
}

@Composable
private fun MessageRow(msg: ChatMessage) {
    val isUser = msg.role == Role.USER

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = if (isUser) Alignment.End else Alignment.Start,
    ) {
        if (isUser) {
            Surface(
                color = MaterialTheme.colorScheme.primaryContainer,
                contentColor = MaterialTheme.colorScheme.onPrimaryContainer,
                shape = RoundedCornerShape(20.dp),
                modifier = Modifier.widthIn(max = 320.dp),
            ) {
                Text(
                    text = msg.text,
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp),
                )
            }
        } else {
            Column(
                modifier = Modifier.widthIn(max = 340.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = msg.text,
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                if (msg.sources.isNotEmpty()) {
                    Text(
                        text = msg.sources.first(),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 3,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
            }
        }
    }
}

@Composable
private fun StatusNote(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.bodyMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
}

@Composable
private fun ThinkingNote() {
    var dotCount by remember { mutableIntStateOf(1) }

    LaunchedEffect(Unit) {
        while (true) {
            delay(450)
            dotCount = if (dotCount == 3) 1 else dotCount + 1
        }
    }

    StatusNote("thinking${".".repeat(dotCount)}")
}

private fun formatElapsedDuration(durationMs: Long): String {
    val totalSeconds = (durationMs.coerceAtLeast(0L) / 1000L).toInt()
    val hours = totalSeconds / 3600
    val minutes = totalSeconds / 60
    val seconds = totalSeconds % 60
    return when {
        hours > 0 -> "${hours}h ${minutes % 60}m ${seconds}s"
        minutes > 0 -> "${minutes}m ${seconds}s"
        else -> "${seconds}s"
    }
}

@Composable
private fun ComposerBar(
    value: String,
    onValueChange: (String) -> Unit,
    placeholder: String,
    sendEnabled: Boolean,
    addEnabled: Boolean,
    inputEnabled: Boolean,
    onAdd: () -> Unit,
    onSend: () -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        IconGlyphButton(
            text = "+",
            onClick = onAdd,
            enabled = addEnabled,
            tint = MaterialTheme.colorScheme.primary,
        )
        TextField(
            value = value,
            onValueChange = onValueChange,
            modifier = Modifier.weight(1f),
            enabled = inputEnabled,
            textStyle = MaterialTheme.typography.bodyLarge,
            placeholder = {
                Text(
                    text = placeholder,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            },
            shape = RoundedCornerShape(24.dp),
            colors = TextFieldDefaults.colors(
                focusedContainerColor = MaterialTheme.colorScheme.surfaceVariant,
                unfocusedContainerColor = MaterialTheme.colorScheme.surfaceVariant,
                disabledContainerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f),
                focusedIndicatorColor = Color.Transparent,
                unfocusedIndicatorColor = Color.Transparent,
                disabledIndicatorColor = Color.Transparent,
                focusedTextColor = MaterialTheme.colorScheme.onSurface,
                unfocusedTextColor = MaterialTheme.colorScheme.onSurface,
                disabledTextColor = MaterialTheme.colorScheme.onSurfaceVariant,
                cursorColor = MaterialTheme.colorScheme.primary,
            ),
            singleLine = false,
            maxLines = 4,
        )
        IconGlyphButton(
            text = ">",
            onClick = onSend,
            enabled = sendEnabled,
            tint = MaterialTheme.colorScheme.primary,
        )
    }
}

@Composable
private fun IconGlyphButton(
    text: String,
    onClick: () -> Unit,
    enabled: Boolean,
    tint: Color,
) {
    IconButton(
        onClick = onClick,
        enabled = enabled,
        modifier = Modifier
            .background(
                color = if (enabled) MaterialTheme.colorScheme.surfaceVariant else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f),
                shape = CircleShape,
            )
            .size(42.dp),
    ) {
        Text(
            text = text,
            color = if (enabled) tint else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.55f),
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Medium,
        )
    }
}
