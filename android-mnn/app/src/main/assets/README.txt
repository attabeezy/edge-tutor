# Place MNN model files here before building.
#
# Expected directory layout:
#   mnn_model/
#     config.json
#     llm.mnn         (main weight file)
#     embedding.mnn   (optional)
#     ... other .mnn shards as provided by the HuggingFace repo ...
#
# Download options:
#   Option A — Hugging Face CLI:
#     pip install huggingface_hub
#     huggingface-cli download taobao-mnn/Qwen2.5-0.5B-Instruct-MNN \
#       --local-dir android-mnn/app/src/main/assets/mnn_model
#
#   Option B — adb push (already downloaded):
#     adb push <local-model-dir>/ /data/data/com.edgetutor.mnn/files/mnn_model/
#
# Do NOT commit .mnn files to git (they are git-ignored).
