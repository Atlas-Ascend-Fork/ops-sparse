#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -e

LOG_HEAD() {
    local msg=${1}
    local ts
    ts=$(date +%Y%m%d-%H%M%S)
    echo "[INFO] ${ts} ${msg}"
}

LOG_ERROR() {
    local msg=${1}
    local ts
    ts=$(date +%Y%m%d-%H%M%S)
    echo "[ERROR] ${ts} ${msg}"
}

LOG_DO() {
    local cmd="$*"
    local ts
    ts=$(date +%Y%m%d-%H%M%S)
    echo "[Command] ${ts} ${cmd}"
    ${cmd}
}

LOG_INFO() {
    local msg=${1}
    local ts
    ts=$(date +%Y%m%d-%H%M%S)
    echo "[INFO] ${ts} ${msg}"
}

DP_ASSERT_CHECK_SKIP() {
    local actual_value=${1}
    local assert_msg=${2}
    if [ "${actual_value}" != "0" ] && [ "${actual_value}" != "200" ]; then
        LOG_ERROR "${assert_msg} is failed."
        exit 1
    else
        LOG_INFO "${assert_msg} is success."
    fi
}

CHECK_ENV_VAR() {
    local var_name=${1}
    local var_value=${!var_name}
    if [[ -z "${var_value}" ]]; then
        LOG_ERROR "Environment variable ${var_name} is not set"
        exit 1
    fi
}

CHECK_ENV_VAR task_name
CHECK_ENV_VAR WORKSPACE
CHECK_ENV_VAR GIT_TARGET_BRANCH

LOG_HEAD "package_name: ${package_name}"
LOG_HEAD "task_name: ${task_name}"
LOG_HEAD "WORKSPACE: ${WORKSPACE}"
LOG_HEAD "GIT_TARGET_BRANCH: ${GIT_TARGET_BRANCH}"
LOG_HEAD "ge_st_rt2: ${ge_st_rt2}"

ASCEND_3RD_LIB_PATH="/home/jenkins/opensource"

if [ -f "/home/jenkins/Ascend/cann/bin/setenv.bash" ]; then
    export ASCEND_HOME_PATH=/home/jenkins/Ascend/cann
elif [ -f "/home/jenkins/Ascend/latest/bin/setenv.bash" ]; then
    export ASCEND_HOME_PATH=/home/jenkins/Ascend/latest
else
    export ASCEND_HOME_PATH=/home/jenkins/Ascend/ascend-toolkit/latest
fi

if [ -d "/home/jenkins/Ascend/ascend-toolkit/latest" ]; then
    export ASCEND_INSTALL_PATH="/home/jenkins/Ascend/ascend-toolkit/latest"
elif [ -d "/home/jenkins/Ascend/cann" ]; then
    export ASCEND_INSTALL_PATH="/home/jenkins/Ascend/cann"
else
    export ASCEND_INSTALL_PATH="/home/jenkins/Ascend/latest"
fi

source ${ASCEND_HOME_PATH}/bin/setenv.bash

if [[ "${task_name}" == *_ubuntu24 ]]; then
    sudo update-alternatives --set gcc /usr/bin/gcc-14
else
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
    echo "source devtoolset"
    source /opt/rh/devtoolset-7/enable
fi
gcc --version
cmake --version

cd "${WORKSPACE}" || exit

if [[ "${task_name}" =~ Compile_Ascend_X86_A2_ubuntu24 ]]; then
    sed -i "1i set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" "CMakeLists.txt"
    echo "api-check=compile" >> "${ATOMGIT_OUTPUT}"
else
    echo "api-check=continue" >> "${ATOMGIT_OUTPUT}"
fi

set +e
if  [[ "${task_name}" == *A2* ]];then
   LOG_DO bash build.sh --soc=ascend910b --pkg
elif [[ "${task_name}" == *A5* ]];then
   LOG_DO bash build.sh --soc=ascend950 --pkg
fi
BUILD_EXIT_CODE=$?
set -e
DP_ASSERT_CHECK_SKIP "${BUILD_EXIT_CODE}" "bash build.sh"
