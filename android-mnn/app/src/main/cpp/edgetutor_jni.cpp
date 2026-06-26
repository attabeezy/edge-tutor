// Thin JNI bridge: com.edgetutor.mnn.llm.MnnNativeBridge -> MNN Llm C++ API
#include <jni.h>
#include <android/log.h>
#include <string>
#include <sstream>
#include <streambuf>
#include <atomic>
#include <unordered_map>

#include "llm/llm.hpp"

#define TAG "EdgeTutorJNI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

using namespace MNN::Transformer;

static void logLongMessage(android_LogPriority priority,
                           const char* prefix,
                           const std::string& message) {
    if (message.empty()) {
        __android_log_print(priority, TAG, "%s<empty>", prefix);
        return;
    }
    constexpr size_t kChunkSize = 3500;
    for (size_t offset = 0; offset < message.size(); offset += kChunkSize) {
        std::string chunk = message.substr(offset, kChunkSize);
        __android_log_print(priority, TAG, "%s%s", prefix, chunk.c_str());
    }
}

// ---------------------------------------------------------------------------
// Custom streambuf that calls Kotlin MnnProgressListener.onProgress per write
// ---------------------------------------------------------------------------
class KotlinCallbackBuf : public std::streambuf {
public:
    KotlinCallbackBuf(JNIEnv* env, jobject listener, jmethodID onProgress)
        : env_(env), listener_(listener), onProgress_(onProgress) {}

    bool isStopped() const { return stopped_; }

protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (stopped_ || !s || n <= 0) return n;
        std::string chunk(s, static_cast<size_t>(n));
        jstring jtoken = env_->NewStringUTF(chunk.c_str());
        jboolean stop = env_->CallBooleanMethod(listener_, onProgress_, jtoken);
        env_->DeleteLocalRef(jtoken);
        if (stop) stopped_ = true;
        return n;
    }

    int overflow(int c) override {
        if (c != EOF) {
            char ch = static_cast<char>(c);
            xsputn(&ch, 1);
        }
        return c;
    }

private:
    JNIEnv*   env_;
    jobject   listener_;
    jmethodID onProgress_;
    bool      stopped_ = false;
};

// ---------------------------------------------------------------------------
// Helper: build a java.util.HashMap<String, Long> from a flat key/value list
// ---------------------------------------------------------------------------
static jobject buildMetricsMap(JNIEnv* env,
                                int64_t prompt_len, int64_t decode_len,
                                int64_t prefill_us, int64_t decode_us) {
    jclass    hmClass  = env->FindClass("java/util/HashMap");
    jmethodID hmInit   = env->GetMethodID(hmClass, "<init>", "()V");
    jmethodID hmPut    = env->GetMethodID(hmClass, "put",
                             "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jclass    lngClass = env->FindClass("java/lang/Long");
    jmethodID lngInit  = env->GetMethodID(lngClass, "<init>", "(J)V");

    jobject map = env->NewObject(hmClass, hmInit);
    auto put = [&](const char* key, int64_t val) {
        jstring k = env->NewStringUTF(key);
        jobject v = env->NewObject(lngClass, lngInit, val);
        env->CallObjectMethod(map, hmPut, k, v);
        env->DeleteLocalRef(k);
        env->DeleteLocalRef(v);
    };
    put("prompt_len",   prompt_len);
    put("decode_len",   decode_len);
    put("prefill_time", prefill_us);
    put("decode_time",  decode_us);
    return map;
}

// ---------------------------------------------------------------------------
// JNI methods — names match com.edgetutor.mnn.llm.MnnNativeBridge (object)
// ---------------------------------------------------------------------------
extern "C" {

// MnnNativeBridge.initSession(modelDir: String, configJson: String): Long
JNIEXPORT jlong JNICALL
Java_com_edgetutor_mnn_llm_MnnNativeBridge_initSession(JNIEnv* env, jclass /*cls*/,
                                                        jstring jModelDir,
                                                        jstring jConfigJson) {
    const char* modelDir  = env->GetStringUTFChars(jModelDir,  nullptr);
    const char* configJson = env->GetStringUTFChars(jConfigJson, nullptr);

    std::string configPath = std::string(modelDir) + "/config.json";
    LOGD("initSession: loading from %s", configPath.c_str());

    Llm* llm = Llm::createLLM(configPath);
    env->ReleaseStringUTFChars(jModelDir,  modelDir);

    if (!llm) {
        LOGE("initSession: Llm::createLLM returned null");
        env->ReleaseStringUTFChars(jConfigJson, configJson);
        return 0L;
    }

    // Apply caller-supplied config overrides (e.g., enable_thinking=false)
    if (configJson && configJson[0] != '\0' &&
        !(configJson[0] == '{' && configJson[1] == '}')) {
        llm->set_config(std::string(configJson));
    }
    env->ReleaseStringUTFChars(jConfigJson, configJson);

    const std::string effectiveConfig = llm->dump_config();
    LOGD("initSession: effective config flags is_visual=%d has_deepstack=%d",
         effectiveConfig.find("\"is_visual\":true") != std::string::npos,
         effectiveConfig.find("\"has_deepstack\":true") != std::string::npos);

    if (!llm->load()) {
        LOGE("initSession: llm->load() failed");
        logLongMessage(ANDROID_LOG_ERROR, "initSession: MNN load log: ", llm->getLog());
        Llm::destroy(llm);
        return 0L;
    }

    LOGD("initSession: success, ptr=%p", llm);
    return reinterpret_cast<jlong>(llm);
}

// MnnNativeBridge.submitPrompt(...): Map<String, Long>
JNIEXPORT jobject JNICALL
Java_com_edgetutor_mnn_llm_MnnNativeBridge_submitPrompt(JNIEnv* env, jclass /*cls*/,
                                                          jlong sessionPtr,
                                                          jstring jPrompt,
                                                          jboolean keepHistory,
                                                          jobject progressListener) {
    auto* llm = reinterpret_cast<Llm*>(sessionPtr);
    if (!llm) {
        LOGE("submitPrompt: null session pointer");
        return buildMetricsMap(env, 0, 0, 0, 0);
    }

    const char* prompt = env->GetStringUTFChars(jPrompt, nullptr);
    std::string promptStr(prompt);
    env->ReleaseStringUTFChars(jPrompt, prompt);

    // Reset KV cache for single-turn mode (keepHistory=false)
    if (!keepHistory) {
        llm->reset();
    }

    // Resolve the onProgress(String?):Boolean method on the listener
    jmethodID onProgress = nullptr;
    if (progressListener) {
        jclass lClass = env->GetObjectClass(progressListener);
        onProgress = env->GetMethodID(lClass, "onProgress", "(Ljava/lang/String;)Z");
        if (!onProgress) {
            LOGE("submitPrompt: MnnProgressListener.onProgress method not found");
        }
    }

    // Tokenise the pre-formatted prompt (avoids double chat-template application)
    std::vector<int> inputIds = llm->tokenizer_encode(promptStr);

    // Build the streaming ostream backed by our Kotlin callback
    KotlinCallbackBuf cbBuf(env, progressListener, onProgress);
    std::ostream os(&cbBuf);

    // Run generation; MNN writes decoded tokens to os as they're produced.
    // max_new_tokens=-1 lets MNN use its own default (usually limited by context).
    // Pass end_with=nullptr since Kotlin handles stop-sequence scanning.
    constexpr int MAX_NEW_TOKENS = 600;
    llm->response(inputIds, &os, nullptr, MAX_NEW_TOKENS);

    // Signal EOP to the Kotlin side (null token = generation complete)
    if (progressListener && onProgress) {
        env->CallBooleanMethod(progressListener, onProgress, nullptr);
    }

    // Collect timing metrics
    const LlmContext* ctx = llm->getContext();
    int64_t prompt_len  = ctx ? ctx->prompt_len  : 0;
    int64_t decode_len  = ctx ? ctx->gen_seq_len : 0;
    int64_t prefill_us  = ctx ? ctx->prefill_us  : 0;
    int64_t decode_us   = ctx ? ctx->decode_us   : 0;

    return buildMetricsMap(env, prompt_len, decode_len, prefill_us, decode_us);
}

// MnnNativeBridge.releaseSession(sessionPtr: Long)
JNIEXPORT void JNICALL
Java_com_edgetutor_mnn_llm_MnnNativeBridge_releaseSession(JNIEnv* /*env*/, jclass /*cls*/,
                                                           jlong sessionPtr) {
    auto* llm = reinterpret_cast<Llm*>(sessionPtr);
    if (llm) {
        LOGD("releaseSession: destroying ptr=%p", llm);
        Llm::destroy(llm);
    }
}

} // extern "C"
