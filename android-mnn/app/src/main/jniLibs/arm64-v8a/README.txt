# libMNN.so — MNN monolithic shared library (LLM baked in)
#
# STATUS: Already present.
# Source: MNN/project/android/build_64/libMNN.so
# Built with: MNN_BUILD_LLM=ON, MNN_ARM82=ON, MNN_BUILD_SHARED_LIBS=ON
# ABI: arm64-v8a
#
# If you need to rebuild from the MNN source tree (MNN/ in the repo root):
#
#   cd MNN/project/android
#   mkdir build_64 && cd build_64
#   cmake .. \
#     -DCMAKE_TOOLCHAIN_FILE=$NDK_HOME/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a \
#     -DANDROID_NATIVE_API_LEVEL=android-21 \
#     -DMNN_BUILD_LLM=ON \
#     -DMNN_ARM82=ON \
#     -DMNN_BUILD_SHARED_LIBS=ON \
#     -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc) MNN
#   cp libMNN.so \
#       ../../../android-mnn/app/src/main/jniLibs/arm64-v8a/libMNN.so
#
# Do NOT commit libMNN.so to git (it is git-ignored).
