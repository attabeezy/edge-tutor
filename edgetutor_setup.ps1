# =============================================================================
# EdgeTutor — Dev Environment Setup (Windows / PowerShell 7+)
# =============================================================================
# Run the phase you need. Each phase INCLUDES the one before it.
#
# Usage (from the edge-tutor project folder):
#   .\edgetutor_setup.ps1 phase1   # ingestion pipeline only
#   .\edgetutor_setup.ps1 phase2   # + full RAG (Ollama + Qwen2.5)
#   .\edgetutor_setup.ps1 phase3   # + Android Studio scaffold + JDK check
#   .\edgetutor_setup.ps1 all      # everything
#
# Prerequisites (already on your machine):
#   - Python 3.10+  (python.exe on PATH)
#   - Ollama        (ollama.exe on PATH)
#   - PowerShell 7+ (you have 7.5.4)
# =============================================================================

param(
    [Parameter(Position=0)]
    [ValidateSet("phase1","phase2","phase3","all","help","")]
    [string]$Phase = "help"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
function Info    { param($m) Write-Host "[info]  $m" -ForegroundColor Cyan    }
function Ok      { param($m) Write-Host "[ok]    $m" -ForegroundColor Green   }
function Warn    { param($m) Write-Host "[warn]  $m" -ForegroundColor Yellow  }
function Die     { param($m) Write-Host "[error] $m" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------------------
# Helpers: write files, creating parent directories as needed
# ---------------------------------------------------------------------------
function Write-ProjectFile {
    param([string]$Path, [string]$Content)
    $dir = Split-Path $Path -Parent
    if ($dir -and !(Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    # Use UTF-8 without BOM — Python is sensitive to BOM in source files
    [System.IO.File]::WriteAllText($Path, $Content, [System.Text.UTF8Encoding]::new($false))
}

# Use this for Python source files — skips writing if the file already exists
# so that manual improvements made after initial setup are never overwritten.
function Write-SourceFile {
    param([string]$Path, [string]$Content)
    if (Test-Path $Path) {
        Warn "Skipping $Path (already exists — delete it manually to regenerate)"
        return
    }
    Write-ProjectFile $Path $Content
}

# ===========================================================================
# PHASE 1 — Ingestion pipeline (Python)
# Needs: Python 3.10+
# Gives you: PDF parsing, chunking, embedding (all-MiniLM-L6-v2), FAISS index
# ===========================================================================
function Phase1 {
    Info "=== Phase 1: Ingestion pipeline ==="

    # Check Python
    try { $pyver = python --version 2>&1; Ok "Found $pyver" }
    catch { Die "python not found on PATH. Install from https://python.org and re-run." }

    # Project scaffold — lives next to the script
    $ProjectDir = $PSScriptRoot
    $dirs = @(
        "$ProjectDir\data\raw",
        "$ProjectDir\data\index",
        "$ProjectDir\src\ingestion",
        "$ProjectDir\src\rag",
        "$ProjectDir\src\utils",
        "$ProjectDir\tests",
        "$ProjectDir\notebooks"
    )
    foreach ($d in $dirs) {
        if (!(Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
    }
    Info "Project directory: $ProjectDir"

    # Virtual environment
    $venvActivate = "$ProjectDir\.venv\Scripts\Activate.ps1"
    $venvPython   = "$ProjectDir\.venv\Scripts\python.exe"

    if ((Test-Path "$ProjectDir\.venv") -and !(Test-Path $venvActivate)) {
        Warn ".venv is broken (previous failed run) — removing and recreating"
        Remove-Item -Recurse -Force "$ProjectDir\.venv"
    }
    if (!(Test-Path "$ProjectDir\.venv")) {
        Info "Creating virtual environment..."
        python -m venv "$ProjectDir\.venv"
        Ok "Created .venv"
    } else {
        Ok ".venv already exists and is valid"
    }

    # Activate venv for this session
    & $venvActivate

    # Upgrade pip
    & $venvPython -m pip install --upgrade pip --quiet

    # Phase 1 dependencies
    Info "Installing Phase 1 packages (pypdf, sentence-transformers, faiss-cpu, numpy)..."
    & $venvPython -m pip install --quiet `
        pypdf==4.3.1 `
        "pdfminer.six==20221105" `
        sentence-transformers==3.3.1 `
        "faiss-cpu==1.9.0.post1" `
        numpy `
        tqdm==4.66.4 `
        pytest==8.2.2
    if ($LASTEXITCODE -ne 0) { Die "pip install failed — see errors above" }
    Ok "Phase 1 packages installed"

    # Write requirements
    & $venvPython -m pip freeze | Out-File "$ProjectDir\requirements.txt" -Encoding utf8
    Ok "requirements.txt written"

    # ------------------------------------------------------------------
    # Source files
    # ------------------------------------------------------------------
    Write-SourceFile "$ProjectDir\src\__init__.py" ""
    Write-SourceFile "$ProjectDir\src\ingestion\__init__.py" '"""EdgeTutor ingestion pipeline — Phase 1."""'

    Write-SourceFile "$ProjectDir\src\ingestion\pipeline.py" @'
"""
Parse -> clean -> chunk -> embed -> FAISS index.
Entry point: ingest(pdf_path, index_dir)
"""
import re
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ------------------------------------------------------------------
# Config (tune in Week 1-2 testing)
# ------------------------------------------------------------------
CHUNK_TOKENS            = 400   # target chunk size  (spec: 256-512)
OVERLAP_TOKENS          = 50    # overlap between chunks
APPROX_CHARS_PER_TOKEN  = 4     # rough heuristic for splitting

EMBED_MODEL = "all-MiniLM-L6-v2"   # 22M params, ~80 MB, 384-dim


# ------------------------------------------------------------------
# Step 1: Parse
# ------------------------------------------------------------------
def parse_pdf(path: str) -> List[Tuple[int, str]]:
    """Return list of (page_number, page_text) tuples."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


# ------------------------------------------------------------------
# Step 2: Clean
# ------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Strip common PDF noise: headers, page numbers, excess whitespace."""
    text = re.sub(r"\f", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


# ------------------------------------------------------------------
# Step 3: Chunk
# ------------------------------------------------------------------
def chunk_text(text: str, chunk_chars: int = None, overlap_chars: int = None):
    """
    Sliding window chunker that respects paragraph breaks.
    Returns list of chunk strings.
    """
    chunk_chars   = chunk_chars   or CHUNK_TOKENS   * APPROX_CHARS_PER_TOKEN
    overlap_chars = overlap_chars or OVERLAP_TOKENS * APPROX_CHARS_PER_TOKEN

    # Split on paragraph breaks first; fall back to sentence boundaries
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    # If a paragraph is longer than chunk_chars, break it into smaller pieces
    # so the sliding window below always has manageable units to work with
    units = []
    for para in paragraphs:
        if len(para) <= chunk_chars:
            units.append(para)
        else:
            # Try sentence splits first
            sentences = re.split(r"(?<=[.!?])\s+", para)
            bucket = ""
            for sent in sentences:
                if len(bucket) + len(sent) <= chunk_chars:
                    bucket += (" " if bucket else "") + sent
                else:
                    if bucket:
                        units.append(bucket)
                    # If a single sentence is still too long, hard-split it
                    while len(sent) > chunk_chars:
                        units.append(sent[:chunk_chars])
                        sent = sent[chunk_chars - overlap_chars:]
                    bucket = sent
            if bucket:
                units.append(bucket)

    # Sliding window over units with overlap
    chunks, current = [], ""
    for unit in units:
        if len(current) + len(unit) + 1 <= chunk_chars:
            current += (" " if current else "") + unit
        else:
            if current:
                chunks.append(current)
            overlap = current[-overlap_chars:] if current else ""
            current = (overlap + " " + unit).strip() if overlap else unit
    if current:
        chunks.append(current)
    return chunks


# ------------------------------------------------------------------
# Step 4: Embed
# ------------------------------------------------------------------
_model_cache = {}

def get_embed_model(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    if model_name not in _model_cache:
        print(f"Loading embedding model: {model_name} (first run downloads ~80 MB)")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_chunks(chunks: List[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    model = get_embed_model(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype("float32")


# ------------------------------------------------------------------
# Step 5: FAISS index
# ------------------------------------------------------------------
def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index(index, chunks: List[str], index_dir: str, doc_name: str):
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(Path(index_dir) / f"{doc_name}.faiss"))
    np.save(
        str(Path(index_dir) / f"{doc_name}_chunks.npy"),
        np.array(chunks, dtype=object),
    )


def load_index(index_dir: str, doc_name: str):
    index  = faiss.read_index(str(Path(index_dir) / f"{doc_name}.faiss"))
    chunks = np.load(
        str(Path(index_dir) / f"{doc_name}_chunks.npy"), allow_pickle=True
    ).tolist()
    return index, chunks


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def ingest(pdf_path: str, index_dir: str = "data/index") -> dict:
    """Full ingestion pipeline. Returns stats dict."""
    doc_name = Path(pdf_path).stem
    print(f"\n--- Ingesting: {pdf_path} ---")

    pages      = parse_pdf(pdf_path)
    full_text  = "\n\n".join(clean_text(text) for _, text in pages)
    chunks     = chunk_text(full_text)
    embeddings = embed_chunks(chunks)
    index      = build_index(embeddings)
    save_index(index, chunks, index_dir, doc_name)

    stats = {
        "doc":       doc_name,
        "pages":     len(pages),
        "chunks":    len(chunks),
        "embed_dim": embeddings.shape[1],
    }
    print(f"Done: {stats}")
    return stats


# ------------------------------------------------------------------
# Quick retrieval (used in Phase 1 validation)
# ------------------------------------------------------------------
def retrieve(query: str, index_dir: str, doc_name: str, top_k: int = 3):
    """Return top-k (chunk, distance) tuples for a query."""
    index, chunks = load_index(index_dir, doc_name)
    model = get_embed_model()
    q_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, idxs = index.search(q_vec, top_k)
    return [(chunks[i], float(distances[0][j])) for j, i in enumerate(idxs[0])]
'@

    Write-SourceFile "$ProjectDir\tests\__init__.py" ""

    Write-SourceFile "$ProjectDir\tests\test_ingestion.py" @'
"""
Phase 1 exit criterion: top-3 retrieval precision > 70% on a test set.
Run:  pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import chunk_text, clean_text


def test_clean_strips_lone_page_numbers():
    text = "Some content\n\n42\n\nMore content"
    result = clean_text(text)
    assert "42" not in result.split("\n")


def test_chunk_produces_overlap():
    text = " ".join(["word"] * 600)
    chunks = chunk_text(text, chunk_chars=200, overlap_chars=50)
    assert len(chunks) >= 2
    tail = chunks[0][-50:]
    assert tail in chunks[1]


def test_chunk_count_reasonable():
    text = " ".join(["word"] * 2000)
    chunks = chunk_text(text, chunk_chars=400, overlap_chars=50)
    assert 4 <= len(chunks) <= 12
'@

    Ok "Phase 1 source files written"
    Write-Host ""
    Info "--- Phase 1 done. Next steps: ---"
    Write-Host "  cd $ProjectDir"
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  pytest tests\ -v"
    Write-Host "  python -c `"from src.ingestion.pipeline import ingest; ingest('data/raw/yourfile.pdf')`""
    Write-Host ""
}

# ===========================================================================
# PHASE 2 — Full RAG pipeline (Python)
# Needs: Phase 1 + Ollama installed and running
# Gives you: question -> retrieve -> generate flow via Qwen2.5-0.5B
# ===========================================================================
function Phase2 {
    Phase1   # Phase 2 includes Phase 1

    Info "=== Phase 2: Full RAG pipeline ==="

    $ProjectDir   = $PSScriptRoot
    $venvPython   = "$ProjectDir\.venv\Scripts\python.exe"
    $venvActivate = "$ProjectDir\.venv\Scripts\Activate.ps1"
    & $venvActivate

    # Additional deps
    Info "Installing Phase 2 packages (ollama client, rich)..."
    & $venvPython -m pip install --quiet `
        ollama==0.3.3 `
        httpx==0.27.0 `
        rich==13.7.1
    if ($LASTEXITCODE -ne 0) { Die "pip install failed — see errors above" }
    Ok "Phase 2 packages installed"

    & $venvPython -m pip freeze | Out-File "$ProjectDir\requirements.txt" -Encoding utf8
    Ok "requirements.txt updated"

    # Pull model via Ollama
    Info "Pulling Qwen2.5-0.5B via Ollama (~400 MB — may take a few minutes)..."
    try {
        ollama pull qwen2.5:0.5b
        Ok "Model pulled"
    } catch {
        Warn "Ollama pull failed. Make sure Ollama is running (check system tray) and retry:"
        Write-Host "  ollama pull qwen2.5:0.5b"
    }

    # RAG source files
    Write-SourceFile "$ProjectDir\src\rag\__init__.py" '"""EdgeTutor RAG pipeline — Phase 2."""'

    Write-SourceFile "$ProjectDir\src\rag\query.py" @'
"""
RAG query pipeline.
Entry points:
  ask(question, doc_name)  -> streams answer to stdout, returns full text
  retrieve_chunks(question, doc_name, top_k) -> list of chunk strings
"""
import re
import time
import ollama

from src.ingestion.pipeline import retrieve, get_embed_model, EMBED_MODEL as DEFAULT_EMBED_MODEL

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
LLM_MODEL            = "qwen2.5:0.5b"
INDEX_DIR            = "data/index"
TOP_K                = 3
MAX_RELEVANT_DISTANCE = 1.4   # L2 threshold; queries above this aren't in the document
MIN_LEXICAL_OVERLAP  = 2      # content-word matches required between question and any chunk

SYSTEM_PROMPT = "Be concise."

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can i you he she it we they "
    "what how why when where who which this that these those of in on at "
    "to for with by from about into than or and but if not no so".split()
)


# ------------------------------------------------------------------
# Retrieve
# ------------------------------------------------------------------
def retrieve_chunks(question: str, doc_name: str, top_k: int = TOP_K, embed_model: str = DEFAULT_EMBED_MODEL, verbose: bool = False):
    """Return (chunks, min_distance) for a question."""
    if verbose:
        t0 = time.perf_counter()
        print(f"\n[embed]     encoding question...", flush=True)
    results = retrieve(question, INDEX_DIR, doc_name, top_k=top_k, model_name=embed_model)
    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"[retrieve]  got {len(results)} chunks  ({elapsed:.3f}s)")
        for i, (chunk, dist) in enumerate(results, 1):
            preview = chunk[:80].replace("\n", " ")
            print(f"[retrieve]  chunk {i} (dist={dist:.3f}): {preview!r}")
    chunks = [chunk for chunk, _dist in results]
    min_dist = min(dist for _chunk, dist in results)
    return chunks, min_dist


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _has_lexical_overlap(question: str, chunks: list[str]) -> bool:
    q_tokens = {w for w in re.findall(r"[a-z]+", question.lower()) if w not in _STOPWORDS}
    # Require fewer matches when the question itself has few content words
    required = min(MIN_LEXICAL_OVERLAP, max(1, len(q_tokens)))
    for chunk in chunks:
        chunk_tokens = set(re.findall(r"[a-z]+", chunk.lower()))
        if len(q_tokens & chunk_tokens) >= required:
            return True
    return False


_CONTINUATION = re.compile(
    r"^(continue|go on|keep going|more|next|and\??|ok|okay|yes|sure|please)\.?$",
    re.IGNORECASE,
)

def _is_followup(text: str) -> bool:
    """True for short inputs that continue the conversation rather than ask something new."""
    stripped = text.strip()
    return bool(_CONTINUATION.match(stripped)) or len(stripped.split()) <= 2


# ------------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------------
def _build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(
        f"[Passage {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    return (
        f"Context passages from the document:\n\n"
        f"{context}\n\n"
        f"Answer using ONLY the passages above.\n"
        f"Question: {question}"
    )


# ------------------------------------------------------------------
# Generate (streaming)
# ------------------------------------------------------------------
def ask(
    question: str,
    doc_name: str,
    history: list[dict] | None = None,
    stream: bool = True,
    embed_model: str = DEFAULT_EMBED_MODEL,
    verbose: bool = False,
    llm_model: str = LLM_MODEL,
) -> tuple[str, list[dict]]:
    """
    Retrieve relevant chunks and generate an answer via Ollama.
    Streams tokens to stdout if stream=True.

    history: list of {"role": ..., "content": ...} dicts from prior turns.
             Pass None or [] to start a fresh conversation.

    Returns (response_text, updated_history).
    """
    history = list(history or [])

    if _is_followup(question) and history:
        # Continuation: don't re-retrieve; just append the bare question
        history.append({"role": "user", "content": question})
    else:
        chunks, min_dist = retrieve_chunks(question, doc_name, embed_model=embed_model, verbose=verbose)
        out_of_scope = (
            min_dist > MAX_RELEVANT_DISTANCE
            or not _has_lexical_overlap(question, chunks)
        )
        if verbose:
            print(f"[gate]      lexical_ok={_has_lexical_overlap(question, chunks)}  min_dist={min_dist:.3f}  threshold={MAX_RELEVANT_DISTANCE}", flush=True)
        if out_of_scope:
            response = "Not covered in this document."
            if stream:
                print(response)
            history.append({"role": "user",      "content": question})
            history.append({"role": "assistant",  "content": response})
            return response, history
        prompt = _build_prompt(question, chunks)
        history.append({"role": "user", "content": prompt})

    if verbose:
        print(f"[llm]       sending to {llm_model}...", flush=True)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    response_text = ""
    stream_iter = ollama.chat(
        model=llm_model,
        messages=messages,
        stream=True,
        options={"temperature": 0.3, "num_predict": 450},
    )

    for chunk in stream_iter:
        token = (chunk.get("message", {}).get("content") if isinstance(chunk, dict) else chunk.message.content) or ""
        response_text += token
        if stream:
            print(token, end="", flush=True)

    if stream:
        print()  # newline after streamed output

    history.append({"role": "assistant", "content": response_text})
    return response_text, history
'@

    Write-SourceFile "$ProjectDir\src\rag\repl.py" @'
"""
Interactive REPL for testing the RAG pipeline.
Usage:
    python -m src.rag.repl <doc_name> [-e minilm|bge|arctic] [-m MODEL] [-v]
"""
import argparse
from src.rag.query import ask, LLM_MODEL

EMBED_MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "bge":    "TaylorAI/bge-micro-v2",
    "arctic": "Snowflake/snowflake-arctic-embed-xs",
}


def main():
    parser = argparse.ArgumentParser(description="EdgeTutor interactive REPL")
    parser.add_argument("doc_name", help="Document name (e.g. CalculusMadeEasy)")
    parser.add_argument(
        "-e", "--embedding",
        choices=["minilm", "bge", "arctic"],
        default="minilm",
        help="Embedding model alias. Default: minilm",
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help=f"Ollama model name (e.g. qwen2.5:0.5b). Default: {LLM_MODEL}",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Print retrieval debug info.",
    )
    args = parser.parse_args()

    embed_model = EMBED_MODELS[args.embedding]
    llm_model   = args.model or LLM_MODEL

    from src.ingestion.pipeline import get_embed_model
    get_embed_model(embed_model)

    print(f"EdgeTutor REPL | doc={args.doc_name} | llm={llm_model} | embed={embed_model}")
    print("Type your question and press Enter. 'new' to reset conversation. Ctrl-C to quit.\n")

    history = []

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue

        if question.lower() == "new":
            history = []
            print("(Conversation reset)\n")
            continue

        print("Tutor: ", end="", flush=True)
        _, history = ask(
            question, args.doc_name,
            history=history, embed_model=embed_model,
            verbose=args.verbose, llm_model=llm_model,
        )
        print()


if __name__ == "__main__":
    main()
'@

    Ok "Phase 2 RAG module written"
    Write-Host ""
    Info "--- Phase 2 done. Next steps: ---"
    Write-Host "  cd $ProjectDir"
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  # Ingest a PDF first:"
    Write-Host "  python -c `"from src.ingestion.pipeline import ingest; ingest('data/raw/yourfile.pdf')`""
    Write-Host "  # Then start the REPL:"
    Write-Host "  python -m src.rag.repl yourfile"
    Write-Host ""
}

# ===========================================================================
# PHASE 3 — Android toolchain
# Needs: Phase 2 + Android Studio installed on Windows
# Gives you: JDK 17 check + Gradle project skeleton ready to open
# ===========================================================================
function Phase3 {
    Phase2   # Phase 3 includes Phase 2

    Info "=== Phase 3: Android toolchain ==="

    # JDK 17 check
    try {
        $javaVer = java -version 2>&1 | Select-String "version"
        if ($javaVer -match '"(17|21|22|23)') {
            Ok "JDK $($Matches[1]) found"
        } else {
            Warn "JDK 17+ not found. Install from: https://adoptium.net"
            Warn "Choose: Temurin 17 (LTS) — required for Android Gradle plugin 8+"
        }
    } catch {
        Warn "java not on PATH. Install JDK 17 from https://adoptium.net"
    }

    # Android Studio reminder
    Warn "Android Studio must be installed separately if not already."
    Write-Host "  Download: https://developer.android.com/studio" -ForegroundColor White
    Write-Host ""

    # Android project scaffold
    $AndroidDir = Join-Path $PSScriptRoot "edgetutor-android"
    $androidDirs = @(
        "$AndroidDir\app\src\main\java\com\edgetutor",
        "$AndroidDir\app\src\main\assets",
        "$AndroidDir\app\src\main\res\layout",
        "$AndroidDir\app\libs"
    )
    foreach ($d in $androidDirs) {
        if (!(Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
    }

    Write-ProjectFile "$AndroidDir\settings.gradle.kts" @'
pluginManagement {
    repositories {
        google(); mavenCentral(); gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories { google(); mavenCentral() }
}
rootProject.name = "EdgeTutor"
include(":app")
'@

    Write-ProjectFile "$AndroidDir\build.gradle.kts" @'
plugins {
    id("com.android.application") version "8.5.0" apply false
    id("org.jetbrains.kotlin.android") version "1.9.24" apply false
    id("com.google.devtools.ksp") version "1.9.24-1.0.20" apply false
}
'@

    Write-ProjectFile "$AndroidDir\app\build.gradle.kts" @'
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.devtools.ksp")
}

android {
    namespace   = "com.edgetutor"
    compileSdk  = 35

    defaultConfig {
        applicationId = "com.edgetutor"
        minSdk        = 29          // Android 10 -- matches spec
        targetSdk     = 35
        versionCode   = 1
        versionName   = "0.1.0-mvp"
    }

    buildFeatures { compose = true }
    composeOptions { kotlinCompilerExtensionVersion = "1.5.14" }

    kotlinOptions { jvmTarget = "17" }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    lint { checkReleaseBuilds = false }
}

dependencies {
    // Jetpack Compose BOM
    val composeBom = platform("androidx.compose:compose-bom:2024.09.00")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.activity:activity-compose:1.9.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.0")

    // Room — metadata DB for document list
    val roomVersion = "2.6.1"
    implementation("androidx.room:room-runtime:$roomVersion")
    implementation("androidx.room:room-ktx:$roomVersion")
    ksp("androidx.room:room-compiler:$roomVersion")

    // ONNX Runtime Mobile — embedding model inference
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")

    // PDF parsing — on-device text extraction
    implementation("com.tom-roush:pdfbox-android:2.0.27.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")

    // Gson — FlatIndex JSON serialisation
    implementation("com.google.code.gson:gson:2.10.1")

    // Llamatik (llama.cpp wrapper for GGUF models — Qwen2.5-0.5B)
    // Verify latest version at https://github.com/ferranpons/Llamatik/releases
    implementation("com.llamatik:library-android:0.11.0")

    // MediaPipe LLM Inference (Google — Gemma 3 270M .task format)
    // Verify latest version at https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android
    implementation("com.google.mediapipe:tasks-genai:0.10.27")
}
'@

    Write-ProjectFile "$AndroidDir\app\src\main\AndroidManifest.xml" @'
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"
        android:maxSdkVersion="32"/>
    <uses-permission android:name="android.permission.READ_MEDIA_DOCUMENTS"/>

    <application
        android:label="EdgeTutor"
        android:theme="@style/Theme.AppCompat.DayNight.NoActionBar">
        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN"/>
                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
    </application>
</manifest>
'@

    Write-SourceFile "$AndroidDir\app\src\main\java\com\edgetutor\MainActivity.kt" @'
package com.edgetutor

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.tooling.preview.Preview

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface { EdgeTutorApp() }
            }
        }
    }
}

@Composable
fun EdgeTutorApp() {
    // TODO Phase 4: wire up Library, Upload, Chat, Settings screens
    Text("Edge-Tutor — scaffold ready")
}

@Preview(showBackground = true)
@Composable
fun DefaultPreview() = EdgeTutorApp()
'@

    Write-ProjectFile "$AndroidDir\README.md" @'
# EdgeTutor Android

## Phase 3 checklist

- [ ] Open this folder in Android Studio (File -> Open -> edgetutor-android)
- [ ] Let Gradle sync and accept SDK licence prompts
- [ ] Download model files (see table below) -> place in app/src/main/assets/
- [ ] Build and run smoke test on a physical device

## LLM engines

Two engines are available — pick one (or both for comparison):

| Engine         | Class             | Model file                          | Size    |
|----------------|-------------------|-------------------------------------|---------|
| Llamatik       | LlamaEngine       | qwen2.5-0.5b-instruct-q4_k_m.gguf  | ~350 MB |
| MediaPipe      | MediaPipeEngine   | gemma-3-270m-it-q4_k_m.task        | ~253 MB |

To switch engines, change one line in ChatViewModel.kt:
  private val llm: LlmEngine by lazy { MediaPipeEngine(app) }

## All model files needed in assets/

| File                                   | Source                                          | Size    |
|----------------------------------------|-------------------------------------------------|---------|
| qwen2.5-0.5b-instruct-q4_k_m.gguf     | HuggingFace: Qwen/Qwen2.5-0.5B-Instruct-GGUF   | ~350 MB |
| gemma-3-270m-it-q4_k_m.task           | HuggingFace: search "gemma-3-270m LiteRT"       | ~253 MB |
| minilm.onnx                            | Run: python scripts/export_onnx.py              | ~22 MB  |
| vocab.txt                              | Run: python scripts/export_onnx.py              | ~226 KB |

Do NOT commit model/ONNX files to git.
'@

    Ok "Android project scaffold written to $AndroidDir"
    Write-Host ""
    Info "--- Phase 3 done. Next steps: ---"
    Write-Host "  1. Open $AndroidDir in Android Studio"
    Write-Host "  2. Let Gradle sync (first sync downloads ~500 MB of tooling)"
    Write-Host "  3. Follow $AndroidDir\README.md for model file placement"
    Write-Host ""
}

# ===========================================================================
# ALL — Phases 1 + 2 + 3
# ===========================================================================
function All {
    Phase3   # includes 1 and 2

    Info "=== Phase 4 notes (UI + testing — iterative in Android Studio) ==="
    Write-Host "  Key Compose screens: Library, Upload, Chat, Settings"
    Write-Host "  See spec section 9 for UX constraints:"
    Write-Host "    - Streaming 'thinking' indicator"
    Write-Host "    - Source attribution per response"
    Write-Host "    - Background ingestion service"
    Write-Host ""
    Ok "Full environment set up. Good luck with Edge-Tutor!"
}

# ===========================================================================
# Dispatch
# ===========================================================================
switch ($Phase) {
    "phase1" { Phase1 }
    "phase2" { Phase2 }
    "phase3" { Phase3 }
    "all"    { All    }
    default  {
        Write-Host ""
        Write-Host "  Usage: .\edgetutor_setup.ps1 [phase1|phase2|phase3|all]" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  phase1  — Python ingestion pipeline only  (weeks 1-2)"
        Write-Host "  phase2  — + full RAG with Ollama + Qwen2.5 (weeks 3-4)"
        Write-Host "  phase3  — + Android Studio scaffold + JDK check (weeks 5-7)"
        Write-Host "  all     — everything (phases 1-4)"
        Write-Host ""
    }
}
