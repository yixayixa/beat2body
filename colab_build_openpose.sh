#!/bin/bash
# Build the official CMU OpenPose (C++, CUDA) on Google Colab.
# Based on the community Colab recipe (github.com/rwlopez98/OpenPose-Colab-Notebook).
# Result: /content/openpose/build/examples/openpose/openpose.bin with the BODY_25 model.
set -e
cd /content
BIN=/content/openpose/build/examples/openpose/openpose.bin
if [ -x "$BIN" ]; then echo "OpenPose already built: $BIN"; exit 0; fi

echo "== 1/5 tools and system libraries"
# OpenPose / Caffe need an older CMake (newer versions reject their minimum-version settings)
wget -q https://cmake.org/files/v3.17/cmake-3.17.2-Linux-x86_64.tar.gz
tar xfz cmake-3.17.2-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local
apt-get -qq update > /dev/null
apt-get -qq install -y libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev \
  libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev liblmdb-dev \
  opencl-headers ocl-icd-opencl-dev libviennacl-dev libboost-all-dev libopencv-dev > /dev/null

echo "== 2/5 source code"
rm -rf openpose
git clone -q --depth 1 https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
# pin the Caffe version that is known to build
sed -i 's/execute_process(COMMAND git checkout --recursive master WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/execute_process(COMMAND git checkout f019d0dfe86f49d1140961f8c7dec22130c83154 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/g' openpose/CMakeLists.txt
(cd openpose && git submodule update --init --recursive --remote > /dev/null)
(cd openpose/3rdparty/caffe && git checkout -q f019d0dfe86f49d1140961f8c7dec22130c83154)

# Colab ships CUDA 13, which needs C++17 and removed a few old device fields
sed -i 's/-std=c++11/-std=c++17/g; s/-std=c++14/-std=c++17/g' openpose/CMakeLists.txt
sed -i 's/    list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC)/    list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC -std=c++17)/' \
  openpose/3rdparty/caffe/cmake/Cuda.cmake
sed -i '/prop.clockRate;/d; /prop.deviceOverlap/d; /prop.kernelExecTimeoutEnabled/d; /"Concurrent copy and execution: "/d; /"Kernel execution timeout:      "/d' \
  openpose/3rdparty/caffe/src/caffe/common.cpp

echo "== 3/5 BODY_25 model (the original CMU download server is offline)"
MODEL=openpose/models/pose/body_25/pose_iter_584000.caffemodel
wget -q -O "$MODEL" https://huggingface.co/camenduru/openpose/resolve/main/models/pose/body_25/pose_iter_584000.caffemodel || true
if [ ! -s "$MODEL" ] || [ "$(stat -c %s "$MODEL")" -lt 100000000 ]; then
  echo "   Hugging Face mirror failed, using the community Google Drive copy"
  pip -q install gdown
  gdown -q 1_z6tj09sfHw833tJfzmpRuLSxdzWMQ8Y -O models.zip
  unzip -q -o models.zip -d openpose
fi
ls -la "$MODEL"

echo "== 4/5 configure (CUDA, no cuDNN)"
mkdir -p openpose/build && cd openpose/build
cmake .. -DUSE_CUDNN=OFF -DBUILD_PYTHON=OFF \
  -DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_BODY_COCO_MODEL=OFF -DDOWNLOAD_BODY_MPI_MODEL=OFF \
  -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF > /content/openpose_cmake.log 2>&1 \
  || { tail -40 /content/openpose_cmake.log; exit 1; }

echo "== 5/5 compile (about 20-40 minutes)"
make -j"$(nproc)" > /content/openpose_make.log 2>&1 || { tail -60 /content/openpose_make.log; exit 1; }
ls -la "$BIN"
echo "OpenPose build finished"
