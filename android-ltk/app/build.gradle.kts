plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.devtools.ksp")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace   = "com.edgetutor"
    compileSdk  = 36

    defaultConfig {
        applicationId = "com.edgetutor"
        minSdk        = 29          // Android 10 — matches spec
        targetSdk     = 35
        versionCode   = 1
        versionName   = "0.1.0-mvp"
    }

    buildFeatures { compose = true }

    kotlinOptions { jvmTarget = "17" }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    // Keep large asset files out of APK lint checks
    lint {
        checkReleaseBuilds = false
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

    // Room — metadata DB for document list
    val roomVersion = "2.7.0"
    implementation("androidx.room:room-runtime:$roomVersion")
    implementation("androidx.room:room-ktx:$roomVersion")
    ksp("androidx.room:room-compiler:$roomVersion")

    // ONNX Runtime Mobile — embedding model inference
    // 1.22.0+ ships 16 KB page-aligned .so files (required for Google Play / Android 15)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")

    // PDF parsing — on-device text extraction
    implementation("com.tom-roush:pdfbox-android:2.0.27.0")

    // Material Components — provides Theme.Material3.DayNight.NoActionBar for the Activity window
    implementation("com.google.android.material:material:1.12.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")

    // Gson — FlatIndex JSON serialisation
    implementation("com.google.code.gson:gson:2.10.1")

    // Llamatik (llama.cpp wrapper for GGUF models — Gemma 3 270M)
    // 0.18.0 includes KV-cache support and improved ARM NEON optimizations
    // Verify latest version at https://github.com/ferranpons/Llamatik/releases
    implementation("com.llamatik:library-android:0.18.0")

    // MediaPipe LLM Inference (Google — Gemma 3 270M .task format)
    // Verify latest version at https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android
    implementation("com.google.mediapipe:tasks-genai:0.10.27")
}
