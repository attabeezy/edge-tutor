cmake_minimum_required(VERSION 3.22.1)
project("edgetutor_mnn" CXX)

# ── Paths ────────────────────────────────────────────────────────────────────
# CMakeLists.txt lives at android-mnn/app/CMakeLists.txt.
# MNN source tree is two levels up from the project root.
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(MNN_ROOT "${REPO_ROOT}/MNN")

# ── Imported pre-built libMNN.so ──────────────────────────────────────────────
add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES
    IMPORTED_LOCATION
        "${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libMNN.so")

# ── JNI bridge library ────────────────────────────────────────────────────────
add_library(edgetutor_mnn SHARED
    src/main/cpp/edgetutor_jni.cpp)

target_include_directories(edgetutor_mnn PRIVATE
    # MNN public C++ headers (Interpreter, Session, Expr, …)
    "${MNN_ROOT}/include"
    # MNN Transformer/LLM public header (llm/llm.hpp)
    "${MNN_ROOT}/transformers/llm/engine/include"
)

target_compile_options(edgetutor_mnn PRIVATE
    -O2
    -fvisibility=hidden
    # MNN is built without RTTI/exceptions; our bridge matches.
    -fno-rtti
    -fno-exceptions
)

# Match MNN's Android chat app: keep native libraries compatible with
# Android 15+ devices that require 16 KB page-size support.
target_link_options(edgetutor_mnn PRIVATE "-Wl,-z,max-page-size=16384")

target_link_libraries(edgetutor_mnn
    MNN
    android
    log)
