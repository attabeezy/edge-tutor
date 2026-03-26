# EdgeTutor Android

## Phase 3 checklist

- [ ] Open this folder in Android Studio (File -> Open -> edgetutor-android)
- [ ] Let Gradle sync and accept SDK licence prompts
- [ ] Install Llamatik dependency (check latest: https://github.com/ferrapons/llamatik)
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