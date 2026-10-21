# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env ASCEND_SLOG_PRINT_TO_STDOUT=0
            ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_QUERY_RESULT
)
if(NOT TORCH_QUERY_RESULT EQUAL 0 OR TORCH_CMAKE_PREFIX_PATH STREQUAL "")
    message(FATAL_ERROR "BUILD_TORCH_ADAPTER requires an importable PyTorch installation")
endif()
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env ASCEND_SLOG_PRINT_TO_STDOUT=0
            ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_NPU_QUERY_RESULT
)
if(NOT TORCH_NPU_QUERY_RESULT EQUAL 0 OR TORCH_NPU_PATH STREQUAL "")
    message(FATAL_ERROR "BUILD_TORCH_ADAPTER requires an importable torch_npu installation")
endif()

set(TORCH_NPU_INCLUDE_DIR "${TORCH_NPU_PATH}/include")
set(TORCH_NPU_LIBRARY_DIR "${TORCH_NPU_PATH}/lib")

add_library(ops_sparse_torch SHARED spmv_npu.cpp)
set_source_files_properties(spmv_npu.cpp PROPERTIES LANGUAGE CXX)

target_include_directories(ops_sparse_torch PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/sparse/common
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/op_common
)

target_link_directories(ops_sparse_torch PRIVATE
    ${TORCH_NPU_LIBRARY_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(ops_sparse_torch PRIVATE
    ${TORCH_LIBRARIES}
    torch_npu
    ops_sparse
    Python3::Python
)

target_compile_features(ops_sparse_torch PRIVATE cxx_std_17)
target_compile_options(ops_sparse_torch PRIVATE ${TORCH_CXX_FLAGS})
set_target_properties(ops_sparse_torch PROPERTIES
    OUTPUT_NAME ops_sparse_torch
    BUILD_RPATH "${TORCH_NPU_LIBRARY_DIR};${ASCEND_CANN_PACKAGE_PATH}/lib64"
    INSTALL_RPATH "${TORCH_NPU_LIBRARY_DIR};${ASCEND_CANN_PACKAGE_PATH}/lib64"
)

install(TARGETS ops_sparse_torch LIBRARY DESTINATION lib64)
