package com.edgetutor.mnn

import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
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
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Icon
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
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.lifecycle.viewmodel.compose.viewModel
import com.edgetutor.mnn.data.db.DocumentEntity
import com.edgetutor.mnn.data.db.IngestionStatus
import com.edgetutor.mnn.ingestion.PdfExtractor
import com.edgetutor.mnn.llm.ModelReadinessKind
import com.edgetutor.mnn.llm.ModelReadinessState
import com.edgetutor.mnn.ui.LatexInlineProcessor
import com.edgetutor.mnn.viewmodel.ChatMessage
import com.edgetutor.mnn.viewmodel.ChatViewModel
import com.edgetutor.mnn.viewmodel.IngestViewModel
import com.edgetutor.mnn.viewmodel.Role
import com.edgetutor.mnn.viewmodel.ThinkingUiState
import io.noties.markwon.Markwon
import io.noties.markwon.ext.latex.JLatexMathPlugin
import io.noties.markwon.ext.tables.TablePlugin
import io.noties.markwon.inlineparser.MarkwonInlineParserPlugin
import kotlinx.coroutines.delay
import android.widget.TextView
import android.content.Context

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
    val thinkingUiState by chatVm.thinkingUiState.collectAsState()
    val warmingUp by chatVm.isWarmingUp.collectAsState()
    val errorMsg by chatVm.errorMessage.collectAsState()
    val activeDocId by chatVm.activeDocumentId.collectAsState()
    val modelReadiness by chatVm.modelReadiness.collectAsState()
    val context = LocalContext.current
    var question by remember { mutableStateOf("") }

    val currentDoc = documents.firstOrNull()
    val readyDoc = currentDoc?.takeIf { it.status == IngestionStatus.DONE }
    val modelReady = modelReadiness.isReady
    val chatLocked = !modelReady || currentDoc == null || currentDoc.status != IngestionStatus.DONE || warmingUp
    val canSend = question.isNotBlank() && !chatLocked && !thinking
    val canAdd = modelReady && question.isBlank() && !thinking

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
        val name = displayNameForUri(context, uri)
        ingestVm.ingest(uri, name)
    }

    val modelPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree(),
    ) { uri: Uri? ->
        uri ?: return@rememberLauncherForActivityResult
        chatVm.importModel(uri)
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

        if (!modelReady) {
            ModelReadinessStrip(
                state = modelReadiness,
                onImport = { modelPicker.launch(null) },
            )
            Spacer(Modifier.height(12.dp))
        }

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
            thinkingUiState = thinkingUiState,
            warmingUp = warmingUp,
        )

        Spacer(Modifier.height(12.dp))
        HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.6f))
        Spacer(Modifier.height(10.dp))

        ComposerBar(
            value = question,
            onValueChange = { question = it },
            placeholder = when {
                !modelReady -> "import the MNN model to start"
                currentDoc == null -> "add a document to start"
                else -> "ask about this document"
            },
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
        IconActionButton(
            icon = Icons.Filled.Close,
            contentDescription = "Delete document",
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
    thinkingUiState: ThinkingUiState,
    warmingUp: Boolean,
) {
    if (currentDoc == null) {
        EmptyState(modifier = modifier.fillMaxWidth())
        return
    }

    val listState = rememberLazyListState()
    LaunchedEffect(messages.size, messages.lastOrNull()?.text, thinkingUiState, warmingUp) {
        val lastIndex = messages.size +
            (if (thinkingUiState !is ThinkingUiState.Idle) 1 else 0) +
            (if (warmingUp) 1 else 0) - 1
        if (lastIndex >= 0) {
            listState.animateScrollToItem(lastIndex)
        }
    }

    LazyColumn(
        modifier = modifier.fillMaxWidth(),
        state = listState,
        verticalArrangement = Arrangement.spacedBy(14.dp),
        contentPadding = PaddingValues(vertical = 8.dp),
    ) {
        items(messages) { message ->
            MessageRow(msg = message)
        }
        when (val ts = thinkingUiState) {
            is ThinkingUiState.Active -> item { ThinkingNote() }
            is ThinkingUiState.Done   -> item { StatusNote("thought for ${formatElapsedDuration(ts.durationMs)}") }
            ThinkingUiState.Idle      -> { /* nothing */ }
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
                AssistantMarkdownText(
                    text = msg.text,
                )
                if (msg.sources.isNotEmpty()) {
                    SourceCard(text = msg.sources.first())
                }
            }
        }
    }
}

@Composable
private fun SourceCard(text: String) {
    Surface(
        color = MaterialTheme.colorScheme.surfaceContainer,
        contentColor = MaterialTheme.colorScheme.onSurfaceVariant,
        shape = RoundedCornerShape(8.dp),
    ) {
        Column(modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp)) {
            Text(
                text = "Source",
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.primary,
            )
            Spacer(Modifier.height(2.dp))
            Text(
                text = text,
                style = MaterialTheme.typography.bodySmall,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

private fun displayNameForUri(context: Context, uri: Uri): String =
    context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
        ?.use { cursor ->
            if (cursor.moveToFirst()) {
                val idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (idx >= 0) cursor.getString(idx) else null
            } else {
                null
            }
        }
        ?: uri.lastPathSegment?.substringAfterLast('/')?.substringAfterLast(':')
        ?: "document"

@Composable
private fun ModelReadinessStrip(
    state: ModelReadinessState,
    onImport: () -> Unit,
) {
    val message = when (state.kind) {
        ModelReadinessKind.MISSING -> "MNN model not imported"
        ModelReadinessKind.IMPORTING -> state.message ?: "importing model"
        ModelReadinessKind.INCOMPLETE -> state.message ?: "model files are incomplete"
        ModelReadinessKind.ERROR -> state.message ?: "model import failed"
        ModelReadinessKind.READY -> "model ready"
    }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surfaceContainerHigh, RoundedCornerShape(16.dp))
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = "Model setup",
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Medium,
            )
            Text(
                text = if (state.kind == ModelReadinessKind.IMPORTING && state.progressTotal > 0) {
                    "$message ${state.progressCurrent}/${state.progressTotal}"
                } else {
                    message
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        TextButton(
            onClick = onImport,
            enabled = state.kind != ModelReadinessKind.IMPORTING,
            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp),
        ) {
            Text("import")
        }
    }
}

@Composable
private fun AssistantMarkdownText(text: String) {
    val context = LocalContext.current
    val textColor = MaterialTheme.colorScheme.onSurface.toArgb()
    val markwon = remember(context) {
        Markwon.builder(context)
            .usePlugin(MarkwonInlineParserPlugin.create { builder ->
                builder.addInlineProcessor(LatexInlineProcessor())
            })
            .usePlugin(TablePlugin.create(context))
            .usePlugin(JLatexMathPlugin.create(16f, 16f) { builder ->
                builder.inlinesEnabled(true)
            })
            .build()
    }

    AndroidView(
        factory = { viewContext ->
            TextView(viewContext).apply {
                setTextColor(textColor)
                textSize = 16f
                setLineSpacing(0f, 1.12f)
            }
        },
        update = { view ->
            view.setTextColor(textColor)
            val renderText = preprocessStreamingMarkdown(text)
            if (renderText.isBlank()) {
                view.text = ""
            } else {
                markwon.setMarkdown(view, renderText)
            }
        },
    )
}

private fun preprocessStreamingMarkdown(text: String): String {
    var inInlineMath = false
    var inBlockMath = false
    var i = 0
    while (i < text.length) {
        if (i < text.length - 1 && text[i] == '$' && text[i + 1] == '$') {
            if (!inInlineMath) inBlockMath = !inBlockMath
            i += 2
        } else if (text[i] == '$') {
            if (!inBlockMath) inInlineMath = !inInlineMath
            i += 1
        } else {
            i += 1
        }
    }

    return when {
        inBlockMath -> "$text\n\$\$"
        inInlineMath -> text + "$"
        else -> text
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
    val startMs = remember { System.currentTimeMillis() }
    var elapsedSecs by remember { mutableIntStateOf(0) }

    LaunchedEffect(Unit) {
        while (true) {
            delay(450)
            dotCount = if (dotCount == 3) 1 else dotCount + 1
            elapsedSecs = ((System.currentTimeMillis() - startMs) / 1000).toInt()
        }
    }

    StatusNote("thinking${".".repeat(dotCount)} (${elapsedSecs}s)")
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
        IconActionButton(
            icon = Icons.Filled.Add,
            contentDescription = "Add document",
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
        IconActionButton(
            icon = Icons.AutoMirrored.Filled.Send,
            contentDescription = "Send question",
            onClick = onSend,
            enabled = sendEnabled,
            tint = MaterialTheme.colorScheme.primary,
        )
    }
}

@Composable
private fun IconActionButton(
    icon: ImageVector,
    contentDescription: String,
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
        Icon(
            imageVector = icon,
            contentDescription = contentDescription,
            tint = if (enabled) tint else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.55f),
            modifier = Modifier.size(21.dp),
        )
    }
}
