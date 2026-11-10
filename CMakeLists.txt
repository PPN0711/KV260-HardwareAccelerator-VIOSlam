cmake_minimum_required(VERSION 3.15)
project(DeltaTrustVIO CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-O3)

# Dùng package hệ thống (cài qua pacman)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)

add_library(stage0 STATIC
    src/stage0/ldlt.cpp
    src/stage0/ldlt_rank1.cpp
    src/stage0/dogleg.cpp
    src/stage0/matrix_utils.cpp)
target_include_directories(stage0 PUBLIC src)
target_link_libraries(stage0 PUBLIC Eigen3::Eigen)

add_executable(stage1
    src/stage1/euroc_reader.cpp
    src/stage1/imu_preintegration.cpp
    src/stage1/reprojection_factor.cpp
    src/stage1/factor_graph.cpp
    src/stage1/schur.cpp
    src/main_stage1.cpp)
target_link_libraries(stage1 PRIVATE stage0 Eigen3::Eigen yaml-cpp)
