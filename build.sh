#!/bin/bash
set -e
mkdir -p build

echo "[1/3] Building stage1..."
g++ -O3 -std=c++17 -I src -I /ucrt64/include/eigen3 -I /ucrt64/include \
    src/main_stage1.cpp \
    src/stage0/ldlt.cpp src/stage0/ldlt_rank1.cpp \
    src/stage0/dogleg.cpp src/stage0/matrix_utils.cpp \
    src/stage1/euroc_reader.cpp src/stage1/imu_preintegration.cpp \
    src/stage1/reprojection_factor.cpp src/stage1/factor_graph.cpp \
    src/stage1/schur.cpp \
    -L /ucrt64/lib -lyaml-cpp \
    -o build/stage1.exe

if [ -f src/main_stage2.cpp ]; then
echo "[2/3] Building stage2..."
g++ -O3 -std=c++17 -I src -I /ucrt64/include/eigen3 \
    src/main_stage2.cpp \
    src/stage0/ldlt.cpp src/stage0/ldlt_rank1.cpp \
    src/stage0/dogleg.cpp src/stage0/matrix_utils.cpp \
    src/stage1/euroc_reader.cpp src/stage1/imu_preintegration.cpp \
    src/stage1/reprojection_factor.cpp src/stage1/factor_graph.cpp \
    src/stage1/schur.cpp \
    src/stage2/sparse_utils.cpp src/stage2/marginalization.cpp \
    -L /ucrt64/lib -lyaml-cpp \
    -o build/stage2.exe
fi

if [ -f src/main_stage3.cpp ]; then
echo "[3/3] Building stage3..."
g++ -O3 -std=c++17 -I src -I /ucrt64/include/eigen3 \
    src/main_stage3.cpp \
    src/stage0/ldlt.cpp src/stage0/ldlt_rank1.cpp \
    src/stage0/dogleg.cpp src/stage0/matrix_utils.cpp \
    src/stage1/euroc_reader.cpp src/stage1/imu_preintegration.cpp \
    src/stage1/reprojection_factor.cpp src/stage1/factor_graph.cpp \
    src/stage1/schur.cpp \
    src/stage3/optimizer.cpp src/stage3/ate_evaluator.cpp \
    -L /ucrt64/lib -lyaml-cpp \
    -o build/stage3.exe
fi

echo "Done. Binaries in build/"
