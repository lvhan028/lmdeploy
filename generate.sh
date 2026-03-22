#!/bin/bash
WORKSPACE_PATH=$(dirname "$(readlink -f "$0")")

builder="-G Ninja"

if [ "$1" == "make" ]; then
    builder=""
fi

cmake ${builder} .. \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_INSTALL_PREFIX=${WORKSPACE_PATH}/install \
    -DBUILD_PY_FFI=ON \
    -DBUILD_MULTI_GPU=ON \
    -DUSE_NVTX=ON \
    -DLMDEPLOY_SPLIT_DEBUG_INFO=ON \
    -DLMDEPLOY_STRIP_BINARIES=ON \
    -DLMDEPLOY_ENABLE_CUDA_LINE_INFO=OFF \
    -DFETCHCONTENT_QUIET=OFF
