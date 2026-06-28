package com.edgetutor.mnn

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.view.isVisible
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import com.edgetutor.mnn.attachments.ImageAttachmentStore
import com.edgetutor.mnn.data.db.IngestionStatus
import com.edgetutor.mnn.databinding.ActivityMainBinding
import com.edgetutor.mnn.ingestion.PdfExtractor
import com.edgetutor.mnn.viewmodel.ChatViewModel
import com.edgetutor.mnn.viewmodel.IngestViewModel
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val ingestVm: IngestViewModel by viewModels()
    private val chatVm: ChatViewModel by viewModels()
    private lateinit var adapter: ChatAdapter
    private lateinit var attachmentStore: ImageAttachmentStore
    private var pendingCameraFile: File? = null

    private val documentPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        ingestVm.ingest(uri, displayName(uri))
    }
    private val modelPicker = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        uri?.let(chatVm::importModel)
    }
    private val photoPicker = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let(::normalizeAttachment)
    }
    private val camera = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        pendingCameraFile?.takeIf { ok }?.let { normalizeAttachment(Uri.fromFile(it)) }
        pendingCameraFile = null
    }
    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) launchCamera() else chatVm.reportError("Camera permission is required to take a photo.")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        PdfExtractor.init(this)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        attachmentStore = ImageAttachmentStore(this)
        adapter = ChatAdapter()
        binding.chatList.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        binding.chatList.adapter = adapter
        binding.chatList.itemAnimator = null
        bindActions()
        collectState()
        handleDebugIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleDebugIntent(intent)
    }

    private fun handleDebugIntent(intent: Intent?) {
        if (!BuildConfig.DEBUG) return
        when (intent?.action) {
            ACTION_RUN_VALIDATION -> chatVm.runValidationSuite()
            ACTION_RUN_PROMPT_BENCHMARK -> chatVm.runPromptPolicyBenchmark()
        }
    }

    private fun bindActions() = with(binding) {
        importModel.setOnClickListener { modelPicker.launch(null) }
        addDocument.setOnClickListener { documentPicker.launch(arrayOf("application/pdf", "text/plain")) }
        send.setOnClickListener {
            if (chatVm.isGenerating.value) chatVm.stopGeneration()
            else {
                val text = composer.text?.toString().orEmpty()
                if (text.isNotBlank() || chatVm.pendingImagePath.value != null) {
                    chatVm.ask(text)
                    composer.text?.clear()
                }
            }
        }
        attach.setOnClickListener {
            photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        cameraButton.setOnClickListener {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) launchCamera()
            else cameraPermission.launch(Manifest.permission.CAMERA)
        }
        removeAttachment.setOnClickListener { chatVm.setPendingImage(null) }
        returnToBottom.setOnClickListener { scrollToBottom() }
        chatList.addOnScrollListener(object : androidx.recyclerview.widget.RecyclerView.OnScrollListener() {
            override fun onScrolled(recyclerView: androidx.recyclerview.widget.RecyclerView, dx: Int, dy: Int) {
                val lm = recyclerView.layoutManager as LinearLayoutManager
                returnToBottom.isVisible = lm.findLastVisibleItemPosition() < adapter.itemCount - 2
            }
        })
    }

    companion object {
        const val ACTION_RUN_VALIDATION = "com.edgetutor.mnn.action.RUN_VALIDATION"
        const val ACTION_RUN_PROMPT_BENCHMARK = "com.edgetutor.mnn.action.RUN_PROMPT_BENCHMARK"
    }

    private fun collectState() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch {
                    combine(ingestVm.documents, ingestVm.progress, chatVm.modelReadiness) { docs, progress, model ->
                        Triple(docs.firstOrNull(), progress, model)
                    }.collect { (doc, progress, model) ->
                        binding.importModel.isVisible = !model.isReady
                        binding.modelStatus.text = if (model.isReady) "Model ready" else model.message ?: "Import Qwen3.5 model"
                        binding.documentName.text = doc?.displayName ?: "No textbook selected"
                        binding.documentStatus.text = when (doc?.status) {
                            IngestionStatus.RUNNING -> progress[doc.id]?.let { "${it.phase} ${it.current}/${it.total}" } ?: "Indexing"
                            IngestionStatus.DONE -> if (doc.isLikelyScanned) "Ready · scanned PDF warning" else "Ready"
                            IngestionStatus.ERROR -> doc.errorMessage ?: "Import failed"
                            else -> "Add a PDF or text file"
                        }
                        binding.addDocument.isEnabled = model.isReady && !chatVm.isGenerating.value
                        if (doc?.status == IngestionStatus.DONE && doc.id != chatVm.activeDocumentId.value) chatVm.loadDocument(doc)
                    }
                }
                launch {
                    chatVm.messages.collect {
                        val nearBottom = (binding.chatList.layoutManager as LinearLayoutManager)
                            .findLastVisibleItemPosition() >= adapter.itemCount - 2
                        adapter.submitList(it)
                        if (nearBottom) binding.chatList.post(::scrollToBottom)
                    }
                }
                launch {
                    combine(chatVm.isGenerating, chatVm.isWarmingUp, chatVm.pendingImagePath) {
                            generating, warming, image -> arrayOf(generating, warming, image)
                    }.collect { state ->
                        val generating = state[0] as Boolean
                        val warming = state[1] as Boolean
                        val image = state[2] as String?
                        binding.send.setImageResource(if (generating) R.drawable.ic_stop else R.drawable.ic_send)
                        binding.send.contentDescription = if (generating) "Stop generation" else "Send message"
                        binding.prefill.isVisible = warming || (generating && adapter.currentList.lastOrNull()?.text.isNullOrBlank())
                        binding.attach.isEnabled = !generating
                        binding.cameraButton.isEnabled = !generating
                        binding.attachmentPreview.isVisible = image != null
                        if (image != null) binding.attachmentImage.setImageURI(Uri.fromFile(File(image)))
                    }
                }
                launch {
                    chatVm.errorMessage.collect {
                        binding.error.isVisible = it != null
                        binding.error.text = it
                    }
                }
            }
        }
    }

    private fun normalizeAttachment(uri: Uri) {
        val docId = chatVm.activeDocumentId.value
        if (docId == null) {
            chatVm.reportError("Add a textbook before attaching an image.")
            return
        }
        lifecycleScope.launch(kotlinx.coroutines.Dispatchers.IO) {
            runCatching { attachmentStore.normalize(uri, docId).absolutePath }
                .onSuccess(chatVm::setPendingImage)
                .onFailure { chatVm.reportError(it.message ?: "Image could not be processed.") }
        }
    }

    private fun launchCamera() {
        val file = attachmentStore.newCameraFile()
        pendingCameraFile = file
        camera.launch(FileProvider.getUriForFile(this, "$packageName.files", file))
    }

    private fun scrollToBottom() {
        if (adapter.itemCount > 0) binding.chatList.scrollToPosition(adapter.itemCount - 1)
    }

    private fun displayName(uri: Uri): String {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
            if (it.moveToFirst()) return it.getString(0)
        }
        return uri.lastPathSegment ?: "Textbook"
    }
}
