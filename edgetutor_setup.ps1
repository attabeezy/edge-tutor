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
# Helper: write a file, creating parent directories as needed
# ---------------------------------------------------------------------------
function Write-ProjectFile {
    param([string]$Path, [string]$Content)
    $dir = Split-Path $Path -Parent
    if ($dir -and !(Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    # Use UTF-8 without BOM — Python is sensitive to BOM in source files
    [System.IO.File]::WriteAllText($Path, $Content, [System.Text.UTF8Encoding]::new($false))
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
    & $venvPython -m pip freeze | Out-File "$ProjectDir\requirements-phase1.txt" -Encoding utf8
    Ok "requirements-phase1.txt written"

    # ------------------------------------------------------------------
    # Source files
    # ------------------------------------------------------------------
    Write-ProjectFile "$ProjectDir\src\__init__.py" ""
    Write-ProjectFile "$ProjectDir\src\ingestion\__init__.py" '"""EdgeTutor ingestion pipeline — Phase 1."""'

    Write-ProjectFile "$ProjectDir\src\ingestion\pipeline.py" @'
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

    Write-ProjectFile "$ProjectDir\tests\__init__.py" ""

    Write-ProjectFile "$ProjectDir\tests\test_ingestion.py" @'
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

    & $venvPython -m pip freeze | Out-File "$ProjectDir\requirements-phase2.txt" -Encoding utf8
    Ok "requirements-phase2.txt written"

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
    Write-ProjectFile "$ProjectDir\src\rag\__init__.py" '"""EdgeTutor RAG pipeline — Phase 2."""'

    Write-ProjectFile "$ProjectDir\src\rag\query.py" @'
"""
Retrieval-Augmented Generation — query pipeline.
Entry point: ask(question, doc_name, index_dir)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import ollama
from src.ingestion.pipeline import retrieve

# ------------------------------------------------------------------
# Config (tune during Week 3-4 testing)
# ------------------------------------------------------------------
OLLAMA_MODEL = "qwen2.5:0.5b"
TOP_K        = 3   # spec: 3-5

SYSTEM_PROMPT = """\
You are Edge-Tutor, an offline AI tutor for engineering students.
You only answer questions based on the context passages provided.
If the context does not contain enough information to answer, say so clearly.
Guide the student with hints and step-by-step reasoning rather than just
giving the final answer. Keep responses concise and focused."""


def build_prompt(question: str, chunks_with_scores: list) -> str:
    context_blocks = "\n---\n".join(c for c, _ in chunks_with_scores)
    return f"Context:\n{context_blocks}\n\nStudent question: {question}"


def ask(question: str, doc_name: str, index_dir: str = "data/index",
        stream: bool = True) -> str:
    """Full RAG pipeline: embed -> retrieve -> generate."""
    chunks_with_scores = retrieve(question, index_dir, doc_name, top_k=TOP_K)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt(question, chunks_with_scores)},
    ]

    if stream:
        response_text = ""
        print("\nEdge-Tutor: ", end="", flush=True)
        for chunk in ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True):
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            response_text += token
        print()
        return response_text
    else:
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        return response["message"]["content"]
'@

    Write-ProjectFile "$ProjectDir\src\rag\repl.py" @'
"""
Quick REPL for Phase 2 manual testing.
Usage: python -m src.rag.repl --doc <doc_name_without_extension>
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.query import ask

def main():
    parser = argparse.ArgumentParser(description="EdgeTutor RAG REPL")
    parser.add_argument("--doc",       required=True, help="Document name (no extension)")
    parser.add_argument("--index-dir", default="data/index")
    args = parser.parse_args()

    print(f"\nEdge-Tutor REPL -- document: {args.doc}")
    print("Type a question, or 'exit' to quit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit", "q"):
            break
        if q:
            ask(q, args.doc, args.index_dir)

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
    Write-Host "  python -m src.rag.repl --doc yourfile"
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
}
'@

    Write-ProjectFile "$AndroidDir\app\build.gradle.kts" @'
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
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

    // Room (document metadata)
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")

    // ONNX Runtime Mobile (embedding model)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")

    // PDF parsing (on-device)
    implementation("com.tom-roush:pdfbox-android:2.0.27.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")

    // Llamatik (llama.cpp wrapper) -- check latest coords:
    // https://github.com/llamatik/llamatik
    // implementation("io.github.llamatik:llamatik-android:<version>")
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

    Write-ProjectFile "$AndroidDir\app\src\main\java\com\edgetutor\MainActivity.kt" @'
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
- [ ] Install Llamatik dependency (check latest: https://github.com/llamatik/llamatik)
- [ ] Download Qwen2.5-0.5B Q4_K_M GGUF -> place in app/src/main/assets/
- [ ] Download all-MiniLM-L6-v2 ONNX    -> place in app/src/main/assets/

## Model files needed in assets/

| File                                    | Source                                        | Size     |
|-----------------------------------------|-----------------------------------------------|----------|
| qwen2.5-0.5b-instruct-q4_k_m.gguf      | HuggingFace: Qwen/Qwen2.5-0.5B-Instruct-GGUF | ~380 MB  |
| all-MiniLM-L6-v2.onnx                   | HuggingFace: sentence-transformers            | ~22 MB   |

## FAISS for Android
Lightweight alternative recommended for MVP:
  https://github.com/spotify/voyager  (Java-native, no JNI build required)
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
