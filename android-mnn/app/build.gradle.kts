plugins {
    id("com.android.application")
    id("com.google.devtools.ksp")
}

val requiredMnnModelFiles = listOf(
    "config.json",
    "llm_config.json",
    "tokenizer.txt",
    "llm.mnn",
    "llm.mnn.json",
    "llm.mnn.weight",
    "visual.mnn",
    "visual.mnn.weight",
)
val bundledMnnModelSource = rootProject.file("../models/Qwen3.5-0.8B-MNN")
val generatedBundledMnnAssets = layout.buildDirectory.dir("generated/assets/bundledMnnModel")
val prepareBundledMnnModel by tasks.registering(Sync::class) {
    group = "build setup"
    description = "Stages the local Qwen MNN model as generated Android assets."
    inputs.dir(bundledMnnModelSource)
    outputs.dir(generatedBundledMnnAssets)

    doFirst {
        val missing = requiredMnnModelFiles.filterNot { bundledMnnModelSource.resolve(it).isFile }
        if (missing.isNotEmpty()) {
            throw GradleException(
                "Cannot bundle Qwen3.5-0.8B-MNN. Missing from " +
                    "${bundledMnnModelSource.absolutePath}: ${missing.joinToString()}",
            )
        }
    }

    from(bundledMnnModelSource) {
        include(requiredMnnModelFiles)
    }
    into(generatedBundledMnnAssets.map { it.dir("mnn_model") })
}

android {
    namespace   = "com.edgetutor.mnn"
    compileSdk  = 36

    defaultConfig {
        applicationId = "com.edgetutor.mnn"
        minSdk        = 29          // Android 10
        targetSdk     = 35
        versionCode   = 1
        versionName   = "0.1.0-mnn"

        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                arguments("-DANDROID_STL=c++_shared")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildFeatures {
        viewBinding = true
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    // Prevent APK compression of MNN weight/binary assets so the runtime can
    // mmap them directly without decompression overhead.
    androidResources {
        noCompress += listOf("mnn", "weight", "bin", "txt", "json", "md")
    }

    // Pre-built libMNN.so lives in jniLibs — allow legacy (uncompressed)
    // packaging so the .so is not compressed inside the APK (required for dlopen).
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    lint {
        checkReleaseBuilds = false
    }

    sourceSets.named("main") {
        assets.srcDir(generatedBundledMnnAssets.get().asFile)
    }
}

// Generated asset directories do not automatically establish a task dependency
// when registered through the legacy sourceSets API.
tasks.configureEach {
    if (name.startsWith("merge") && name.endsWith("Assets")) {
        dependsOn(prepareBundledMnnModel)
    }
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.9.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.1")
    implementation("androidx.recyclerview:recyclerview:1.4.0")
    implementation("androidx.exifinterface:exifinterface:1.4.1")
    implementation("androidx.constraintlayout:constraintlayout:2.2.1")
    implementation("androidx.documentfile:documentfile:1.0.1")

    // Room — document metadata DB (same schema as android-ltk)
    val roomVersion = "2.7.0"
    implementation("androidx.room:room-runtime:$roomVersion")
    implementation("androidx.room:room-ktx:$roomVersion")
    ksp("androidx.room:room-compiler:$roomVersion")

    // ONNX Runtime Mobile — embedding model (arctic.onnx unchanged from ltk)
    // 1.22.0+ ships 16 KB page-aligned .so files (required for Android 15 / Google Play)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")

    // PDF parsing — on-device text extraction (unchanged from ltk)
    implementation("com.tom-roush:pdfbox-android:2.0.27.0")

    // Material Components — system chrome theming
    implementation("com.google.android.material:material:1.12.0")

    // Markdown/math rendering — mirrors the renderer used by MNN Chat.
    implementation("com.github.Juude.Markwon:core:v4.6.2-mnnchat.1")
    implementation("com.github.Juude.Markwon:inline-parser:v4.6.2-mnnchat.1")
    implementation("com.github.Juude.Markwon:ext-latex:v4.6.2-mnnchat.1")
    implementation("com.github.Juude.Markwon:ext-tables:v4.6.2-mnnchat.1")
    implementation("ru.noties:jlatexmath-android:0.2.0")
    implementation("ru.noties:jlatexmath-android-font-cyrillic:0.2.0")
    implementation("ru.noties:jlatexmath-android-font-greek:0.2.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")

    // Gson — FlatIndex JSON serialisation
    implementation("com.google.code.gson:gson:2.10.1")

    // NOTE: No Llamatik dependency here.
    // MNN-LLM inference is provided by the pre-built native library loaded via
    // MnnNativeBridge.  Place the compiled monolithic libMNN.so at:
    //   app/src/main/jniLibs/arm64-v8a/libMNN.so
    // before building.  See README.md for build instructions.

    testImplementation("junit:junit:4.13.2")
}
