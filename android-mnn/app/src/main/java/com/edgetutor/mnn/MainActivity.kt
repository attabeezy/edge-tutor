package com.edgetutor.mnn

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.Menu
import android.view.MenuItem
import android.view.View
import androidx.activity.addCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.view.GravityCompat
import androidx.core.view.isVisible
import com.edgetutor.mnn.data.db.SessionListItem
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import com.edgetutor.mnn.attachments.ImageAttachmentStore
import com.edgetutor.mnn.data.db.IngestionStatus
import com.edgetutor.mnn.data.db.DocumentEntity
import com.edgetutor.mnn.databinding.ActivityMainBinding
import com.edgetutor.mnn.ingestion.PdfExtractor
import com.edgetutor.mnn.viewmodel.ChatViewModel
import com.edgetutor.mnn.viewmodel.IngestionProgress
import com.edgetutor.mnn.viewmodel.IngestViewModel
import com.edgetutor.mnn.llm.ModelReadinessState
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val ingestVm: IngestViewModel by viewModels()
    private val chatVm: ChatViewModel by viewModels()
    private lateinit var adapter: ChatAdapter
    private lateinit var sessionAdapter: SessionAdapter
    private lateinit var documentAdapter: DocumentLibraryAdapter
    private lateinit var attachmentStore: ImageAttachmentStore
    private var pendingCameraFile: File? = null
    private var modelReady = false
    private var generating = false
    private var currentScreen = AppScreen.CHAT
    private var currentDocumentTitle = "EdgeTutor"
    private var currentDocumentStatus = ""

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
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayShowTitleEnabled(false)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.setHomeAsUpIndicator(R.drawable.ic_menu)
        attachmentStore = ImageAttachmentStore(this)
        adapter = ChatAdapter()
        binding.chatList.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        binding.chatList.adapter = adapter
        binding.chatList.itemAnimator = null
        setupLibrary()
        setupHistoryDrawer()
        bindActions()
        showScreen(AppScreen.CHAT)
        collectState()
        handleDebugIntent(intent)
    }

    private fun setupHistoryDrawer() {
        sessionAdapter = SessionAdapter(
            onOpen = {
                chatVm.openSession(it)
                binding.drawerLayout.closeDrawer(GravityCompat.START)
            },
            onDelete = ::confirmDeleteSession,
        )
        binding.sessionList.layoutManager = LinearLayoutManager(this)
        binding.sessionList.adapter = sessionAdapter
        binding.newChat.setOnClickListener {
            chatVm.newSession()
            binding.drawerLayout.closeDrawer(GravityCompat.START)
        }
        onBackPressedDispatcher.addCallback(this) {
            if (binding.drawerLayout.isDrawerOpen(GravityCompat.START)) {
                binding.drawerLayout.closeDrawer(GravityCompat.START)
            } else if (currentScreen != AppScreen.CHAT) {
                showScreen(AppScreen.CHAT)
            } else {
                isEnabled = false
                onBackPressedDispatcher.onBackPressed()
            }
        }
    }

    private fun setupLibrary() {
        documentAdapter = DocumentLibraryAdapter(
            onOpen = { doc ->
                ingestVm.selectDocument(doc.id)
                showScreen(AppScreen.CHAT)
            },
            onDelete = ::confirmDeleteDocument,
        )
        binding.libraryDocumentList.layoutManager = LinearLayoutManager(this)
        binding.libraryDocumentList.adapter = documentAdapter
        binding.libraryDocumentList.itemAnimator = null
    }

    private fun confirmDeleteDocument(doc: DocumentEntity) {
        MaterialAlertDialogBuilder(this)
            .setTitle("Delete textbook?")
            .setMessage("This removes \"${doc.displayName}\", its RAG index, chats, and attachments.")
            .setPositiveButton("Delete") { _, _ -> ingestVm.delete(doc) }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun confirmDeleteSession(item: SessionListItem) {
        MaterialAlertDialogBuilder(this)
            .setTitle("Delete chat?")
            .setMessage("This will permanently remove \"${item.title}\".")
            .setPositiveButton("Delete") { _, _ -> chatVm.deleteSession(item) }
            .setNegativeButton("Cancel", null)
            .show()
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
        navChat.setOnClickListener { showScreen(AppScreen.CHAT) }
        navLibrary.setOnClickListener { showScreen(AppScreen.LIBRARY) }
        navSettings.setOnClickListener { showScreen(AppScreen.SETTINGS) }
        libraryAddTextbook.setOnClickListener {
            documentPicker.launch(arrayOf("application/pdf", "text/plain"))
        }
        settingsImportModel.setOnClickListener { modelPicker.launch(null) }
        settingsRunValidation.setOnClickListener { chatVm.runValidationSuite() }
        settingsRunBenchmark.setOnClickListener { chatVm.runPromptPolicyBenchmark() }
        settingsRunValidation.isVisible = BuildConfig.DEBUG
        settingsRunBenchmark.isVisible = BuildConfig.DEBUG
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
        btPlus.setOnClickListener { layoutMoreMenu.isVisible = !layoutMoreMenu.isVisible }
        btnToggleThinking.setOnClickListener { chatVm.toggleThinking() }
        attach.setOnClickListener {
            layoutMoreMenu.isVisible = false
            photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        cameraButton.setOnClickListener {
            layoutMoreMenu.isVisible = false
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

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onPrepareOptionsMenu(menu: Menu): Boolean {
        menu.findItem(R.id.action_import_model)?.isVisible = false
        menu.findItem(R.id.action_choose_textbook)?.isVisible = false
        return super.onPrepareOptionsMenu(menu)
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean = when (item.itemId) {
        android.R.id.home -> { binding.drawerLayout.openDrawer(GravityCompat.START); true }
        R.id.action_import_model -> { modelPicker.launch(null); true }
        R.id.action_choose_textbook -> {
            documentPicker.launch(arrayOf("application/pdf", "text/plain")); true
        }
        else -> super.onOptionsItemSelected(item)
    }

    companion object {
        const val ACTION_RUN_VALIDATION = "com.edgetutor.mnn.action.RUN_VALIDATION"
        const val ACTION_RUN_PROMPT_BENCHMARK = "com.edgetutor.mnn.action.RUN_PROMPT_BENCHMARK"
    }

    private enum class AppScreen { CHAT, LIBRARY, SETTINGS }

    private data class LibraryState(
        val documents: List<DocumentEntity>,
        val progress: Map<Long, IngestionProgress>,
        val model: ModelReadinessState,
        val selectedDocumentId: Long?,
    )

    private fun showScreen(screen: AppScreen) = with(binding) {
        currentScreen = screen
        val chatVisible = screen == AppScreen.CHAT
        chatList.isVisible = chatVisible
        inputCard.isVisible = chatVisible
        if (!chatVisible) {
            returnToBottom.isVisible = false
            prefill.isVisible = false
            layoutMoreMenu.isVisible = false
        }
        libraryScreen.isVisible = screen == AppScreen.LIBRARY
        settingsScreen.isVisible = screen == AppScreen.SETTINGS
        navChat.isSelected = chatVisible
        navLibrary.isSelected = screen == AppScreen.LIBRARY
        navSettings.isSelected = screen == AppScreen.SETTINGS
        navChat.alpha = if (chatVisible) 1f else 0.55f
        navLibrary.alpha = if (screen == AppScreen.LIBRARY) 1f else 0.55f
        navSettings.alpha = if (screen == AppScreen.SETTINGS) 1f else 0.55f
        supportActionBar?.setDisplayHomeAsUpEnabled(chatVisible)
        toolbarTitle.text = when (screen) {
            AppScreen.CHAT -> currentDocumentTitle
            AppScreen.LIBRARY -> "Library"
            AppScreen.SETTINGS -> "Settings"
        }
        toolbarStatus.text = if (screen == AppScreen.CHAT) currentDocumentStatus else ""
        invalidateOptionsMenu()
    }

    private fun collectState() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch {
                    combine(
                        ingestVm.documents,
                        ingestVm.progress,
                        chatVm.modelReadiness,
                        ingestVm.selectedDocumentId,
                    ) { docs, progress, model, selectedId ->
                        LibraryState(docs, progress, model, selectedId)
                    }.collect { state ->
                        val docs = state.documents
                        val progress = state.progress
                        val model = state.model
                        val doc = docs.firstOrNull { it.id == state.selectedDocumentId }
                            ?: docs.firstOrNull { it.status == IngestionStatus.DONE }
                            ?: docs.firstOrNull()
                        if (doc != null && state.selectedDocumentId != doc.id) {
                            ingestVm.selectDocument(doc.id)
                        }
                        modelReady = model.isReady
                        val status = when {
                            !model.isReady -> model.message ?: "Import Qwen3.5 model"
                            doc == null -> "Add a PDF or text file"
                            doc.status == IngestionStatus.RUNNING ->
                                progress[doc.id]?.let { "${it.phase} ${it.current}/${it.total}" } ?: "Indexing"
                            doc.status == IngestionStatus.DONE ->
                                if (doc.isLikelyScanned) "Ready · scanned PDF warning" else "Ready"
                            doc.status == IngestionStatus.ERROR -> doc.errorMessage ?: "Import failed"
                            else -> "Add a PDF or text file"
                        }
                        currentDocumentTitle = doc?.displayName ?: "EdgeTutor"
                        currentDocumentStatus = status
                        if (currentScreen == AppScreen.CHAT) {
                            binding.toolbarTitle.text = doc?.displayName ?: "EdgeTutor"
                            binding.toolbarStatus.text = status
                        }
                        binding.libraryDocumentName.text = doc?.displayName ?: "No textbook added"
                        binding.libraryDocumentStatus.text = when {
                            doc == null -> "Choose a PDF or text file to begin."
                            doc.status == IngestionStatus.RUNNING -> status
                            doc.status == IngestionStatus.DONE ->
                                "${doc.pageCount} pages · ${doc.chunkCount} RAG chunks · index ready"
                            doc.status == IngestionStatus.ERROR -> status
                            else -> status
                        }
                        binding.libraryProgress.isVisible = doc?.status == IngestionStatus.RUNNING
                        binding.libraryEmpty.isVisible = docs.isEmpty()
                        documentAdapter.submitList(
                            docs.map { item ->
                                val itemStatus = when (item.status) {
                                    IngestionStatus.RUNNING ->
                                        progress[item.id]?.let { "${it.phase} ${it.current}/${it.total}" }
                                            ?: "Building RAG index"
                                    IngestionStatus.DONE ->
                                        "${item.pageCount} pages · ${item.chunkCount} chunks · ready"
                                    IngestionStatus.ERROR -> item.errorMessage ?: "Indexing failed"
                                    IngestionStatus.PENDING -> "Waiting to index"
                                }
                                LibraryDocumentItem(item, itemStatus, item.id == doc?.id)
                            },
                        )
                        binding.libraryAddTextbook.isEnabled = model.isReady &&
                            doc?.status != IngestionStatus.RUNNING && !generating
                        binding.settingsModelStatus.text =
                            if (model.isReady) "Qwen model ready" else model.message ?: "Qwen model not imported"
                        binding.settingsImportModel.isVisible = !model.isReady
                        invalidateOptionsMenu()
                        if (doc?.status == IngestionStatus.DONE && doc.id != chatVm.activeDocumentId.value) {
                            chatVm.loadDocument(doc)
                        } else if (
                            chatVm.activeDocumentId.value != null &&
                            docs.none { it.id == chatVm.activeDocumentId.value }
                        ) {
                            chatVm.clearDocument()
                        }
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
                            generating, warming, image -> Triple(generating, warming, image)
                    }.collect { state ->
                        val (isGenerating, warming, image) = state
                        generating = isGenerating
                        invalidateOptionsMenu()
                        binding.send.setImageResource(if (isGenerating) R.drawable.ic_stop else R.drawable.ic_send)
                        binding.send.contentDescription = if (isGenerating) "Stop generation" else "Send message"
                        binding.prefill.isVisible = warming || (isGenerating && adapter.currentList.lastOrNull()?.text.isNullOrBlank())
                        binding.attach.isEnabled = !isGenerating
                        binding.cameraButton.isEnabled = !isGenerating
                        binding.btnToggleThinking.isEnabled = !isGenerating
                        binding.libraryAddTextbook.isEnabled = modelReady && !isGenerating &&
                            ingestVm.progress.value.isEmpty()
                        val testsIdle = chatVm.validationStatus.value?.startsWith("running") != true
                        binding.settingsRunValidation.isEnabled = testsIdle && modelReady && !isGenerating
                        binding.settingsRunBenchmark.isEnabled = testsIdle && modelReady && !isGenerating
                        binding.attachmentPreview.isVisible = image != null
                        if (image != null) binding.attachmentImage.setImageURI(Uri.fromFile(File(image)))
                    }
                }
                launch {
                    chatVm.thinkingEnabled.collect { binding.btnToggleThinking.isSelected = it }
                }
                launch {
                    chatVm.validationStatus.collect {
                        binding.settingsValidationStatus.text = it ?: "No test running"
                        val running = it?.startsWith("running") == true
                        binding.settingsRunValidation.isEnabled = !running && modelReady && !generating
                        binding.settingsRunBenchmark.isEnabled = !running && modelReady && !generating
                    }
                }
                launch {
                    chatVm.errorMessage.collect {
                        binding.error.isVisible = it != null
                        binding.error.text = it
                    }
                }
                launch {
                    chatVm.sessions.collect {
                        sessionAdapter.submitList(it)
                        binding.noHistory.isVisible = it.isEmpty()
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
