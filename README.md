# DeltaTrust-VIO

VSLAM backend accelerator targeting Xilinx Kria KV260.

## Stages
- [x] Stage 0: Golden model (LDL^T + GGMS rank-1 + Dogleg)
- [x] Stage 1: Real factor graph on EuRoC
- [ ] Stage 2: Sparse + Marginalization
- [ ] Stage 3: Evaluation (ATE)
- [ ] Stage 4: FPGA/HLS

## Build
g++ -O3 -std=c++17 -I src -I /ucrt64/include/eigen3 -I /ucrt64/include \
    src/main_stage1.cpp src/stage0/*.cpp src/stage1/*.cpp \
    -L /ucrt64/lib -lyaml-cpp -o stage1.exe

## Run
./stage1.exe data/euroc/V1_01_easy
