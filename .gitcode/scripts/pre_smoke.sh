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

set -euo pipefail
CI_START_SECONDS=${SECONDS}

echo "start run test case, please wait ..."

log() {
    local dt
    dt=$(date '+%Y%m%d.%H%M%S')
    echo "===================================================================="
    echo "$dt : $*"
    echo "===================================================================="
}

LOG_HEAD() {
    local msg=${1}
    local ts
    ts=$(date +%Y%m%d-%H%M%S)
    echo "[INFO] ${ts} ${msg}"
}

LOG_INFO() {
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

CHECK_ENV_VAR() {
    local var_name=${1}
    local var_value=${!var_name:-}
    if [[ -z "${var_value}" ]]; then
        LOG_ERROR "Environment variable ${var_name} is not set"
        exit 1
    fi
}

CHECK_ENV_VAR REPOSITORY_NAME
CHECK_ENV_VAR pr_id
CHECK_ENV_VAR WORKSPACE
CHECK_ENV_VAR smoke_run_file_url
CHECK_ENV_VAR obs_smoke_path

LOG_HEAD "REPOSITORY_NAME: ${REPOSITORY_NAME}"
LOG_HEAD "pr_id: ${pr_id}"
LOG_HEAD "smoke_run_file_url: ${smoke_run_file_url}"
LOG_HEAD "WORKSPACE: ${WORKSPACE}"

cd "${WORKSPACE}"

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0

log "init test case, please wait ..."
rm -rf /root/ascend/log



# ==============================
# 下载并安装 smoke run 包
# ==============================
log "start downloading packages and running test case ..."

arm_package=$(basename "${smoke_run_file_url}")
echo "Starting to download file: ${arm_package}"
# set -e 下保留“包不存在则跳过冒烟”的原语义：下载失败不中断，由后续存在性检查兜底。
wget -nv --no-clobber "${smoke_run_file_url}" || LOG_ERROR "download failed (treated as no package)"

if [ ! -f "${arm_package}" ]; then
    echo "File ${arm_package} does not exist, no need to execute smoke test task"
    exit 0
fi

FILE_SIZE=$(stat -c%s "${arm_package}" 2>/dev/null || echo 0)
if [ "${FILE_SIZE}" -lt 15360 ]; then
    echo "No compiled operators, no need to execute smoke test task"
    rm -f "${arm_package}"
    touch slog.tar.gz
    exit 0
fi
echo "File download completed, size ${FILE_SIZE}, starting installation."

set +e
yes "y" | bash "${arm_package}" --install --install-path=/usr/local/Ascend --quiet
set -e

if [ -f "/usr/local/Ascend/cann/bin/setenv.bash" ]; then
    source /usr/local/Ascend/cann/bin/setenv.bash
elif [ -f "/usr/local/Ascend/cann/set_env.sh" ]; then
    source /usr/local/Ascend/cann/set_env.sh
fi

# ==============================
# 确定 SOC_VERSION 和 ARCH_DIR
# ==============================
if echo "${arm_package}" | grep -qi "A5"; then
    SOC_VERSION="ascend950"
    ARCH_DIR="arch35"
else
    SOC_VERSION="ascend910b"
    ARCH_DIR="arch22"
fi
LOG_HEAD "Detected SOC_VERSION=${SOC_VERSION}, ARCH_DIR=${ARCH_DIR} from package: ${arm_package}"

# ==============================
# 确定仓库根目录
# ==============================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_hint="${1:-${REPO_ROOT:-${WORKSPACE:-}}}"
if [[ -n "${root_hint}" ]]; then
    REPO_ROOT="$(cd -- "${root_hint}" && pwd)"
else
    REPO_ROOT="${SCRIPT_DIR}"
    while [[ ! -f "${REPO_ROOT}/build.sh" || ! -d "${REPO_ROOT}/sparse" || ! -d "${REPO_ROOT}/test" ]]; do
        [[ "${REPO_ROOT}" != / ]] || break
        REPO_ROOT="$(dirname -- "${REPO_ROOT}")"
    done
fi

if [[ ! -f "${REPO_ROOT}/build.sh" || ! -d "${REPO_ROOT}/sparse" || ! -d "${REPO_ROOT}/test" ]]; then
    LOG_ERROR "Invalid repository root: ${REPO_ROOT}"
    echo "Pass the checkout path as argument, or set REPO_ROOT / WORKSPACE." >&2
    exit 1
fi
cd -- "${REPO_ROOT}"
echo "[INFO] REPO_ROOT=${REPO_ROOT}, SOC=${SOC_VERSION}, ARCH=${ARCH_DIR}"

# ==============================
# 运行测试主循环
# ==============================
LOG_DIR="${CI_TEST_LOG_DIR:-${REPO_ROOT}/ci_test_logs}"
mkdir -p -- "${LOG_DIR}"
RUN_LOG="${REPO_ROOT}/run_test.log"
: > "${RUN_LOG}"

# 只限制初始化重试次数与退避间隔，不对构建或测试设置超时。
DELAY_MIN="${CI_INIT_RETRY_DELAY_MIN:-5}"
DELAY_MAX="${CI_INIT_RETRY_DELAY_MAX:-15}"
for value in "${DELAY_MIN}" "${DELAY_MAX}"; do
    if [[ ! "${value}" =~ ^(0|[1-9][0-9]{0,3})$ ]]; then
        echo "[ERROR] Invalid retry delay: ${value}" >&2
        exit 1
    fi
done
if (( DELAY_MIN > DELAY_MAX || DELAY_MAX > 60 )); then
    echo "[ERROR] Retry delays must satisfy 0 <= min <= max <= 60s." >&2
    exit 1
fi
MAX_ATTEMPTS=3  # 首次执行 + 最多两次重试。
ATTEMPT_ROOT="$(mktemp -d "${LOG_DIR}/attempts.XXXXXX")"
# SpMM 测试自身没有打印真实 ACL 返回码，需将 CANN 错误日志带到尝试日志。
export ASCEND_SLOG_PRINT_TO_STDOUT=1

has_pass_marker() {
    local log_file="$1"
    local op="$2"
    local line
    local ansi_re=$'\033''\\[[0-9;]*m'

    while IFS= read -r line || [[ -n "${line}" ]]; do
        while [[ "${line}" =~ ${ansi_re} ]]; do
            line="${line/"${BASH_REMATCH[0]}"/}"
        done
        line="${line//$'\r'/}"
        [[ "${line}" == "[PASS] ${op}_test" ]] && return 0
    done < "${log_file}"
    return 1
}

# 保守分类：需要明确的设备子进程启动超时证据，并核对测试失败明细。
# 目前已核对 SpMM/SpVV 的输出格式；未知格式不自动重试。
initialization_only_failure() {
    local log_file="$1"
    local op="$2"
    local marker="========== Running ${op}_test =========="
    local ansi_re=$'\033''\\[[0-9;]*m'
    local wrapper_re="^[[:space:]]*\\[FAIL\\] ${op}_test \\(exit code: [0-9]+\\)[[:space:]]*$"
    local aggregate_re='^[[:space:]]*\[Failed\] dtype generalization: [0-9]+ failed of [0-9]+[[:space:]]*$'
    local sdk_re='^[[:space:]]*\[ERROR\] [A-Z0-9_]+\('
    local tagged_re='^[[:space:]]*\[(Fail|FAIL|Failed|FAILED|ERROR)\]'
    local fail_word_re='(^|[^[:alnum:]_])FAIL([^[:alnum:]_]|$)'
    local line lower error_code tagged_allowed
    local seen=0 startup=0 set_device=0 invalid=0
    local spmm_cases=0 spmm_totals=0 spmm_total=0
    local spvv_errors=0 spvv_totals=0 spvv_total=0 spvv_507033=0

    [[ -r "${log_file}" ]] || return 1
    # Bash 内建正则逐行验证；仅对已核对格式的 SpMM/SpVV 放行重试。
    while IFS= read -r line || [[ -n "${line}" ]]; do
        while [[ "${line}" =~ ${ansi_re} ]]; do
            line="${line/"${BASH_REMATCH[0]}"/}"
        done
        line="${line//$'\r'/}"
        lower="${line,,}"
        if [[ "${lower}" =~ (fatal\ error:|cmake\ error|undefined\ reference|compilation\ terminated|gmake.*error) ]]; then
            invalid=1
        fi

        if (( !seen )); then
            [[ "${line}" == *"${marker}"* ]] || continue
            seen=1
            line="${line#*"${marker}"}"
            [[ -n "${line}" ]] || continue
        fi

        lower="${line,,}"
        [[ "${line}" =~ (E39007|Inner_Error_Device_Subprocess_Startup_Timeout|Starting\ a\ subprocess\ on\ the\ device\ timed\ out) ]] && startup=1
        [[ "${line}" =~ (aclrtSetDevice|rtSetDevice|TsdOpen) ]] && set_device=1
        [[ "${lower}" =~ (case\ fail!|check\ fail!|verification.*fail|kernel\ exception|segmentation\ fault|core\ dumped) ]] && invalid=1

        if [[ "${line}" =~ ${tagged_re} ]]; then
            tagged_allowed=0
            [[ "${line}" =~ ${wrapper_re} ]] && tagged_allowed=1
            [[ "${line}" =~ ${aggregate_re} ]] && tagged_allowed=1
            [[ "${line}" =~ ${sdk_re} ]] && tagged_allowed=1
            [[ "${line}" == *'aclrtSetDevice failed'* ]] && tagged_allowed=1
            (( tagged_allowed )) || invalid=1
        fi
        if [[ "${line}" =~ ${fail_word_re} && ! "${line}" =~ ${wrapper_re} ]]; then
            invalid=1
        fi

        if [[ "${op}" == spmm ]]; then
            [[ "${line}" =~ ^[[:space:]]*\[Fail\].*aclrtSetDevice\ failed.*$ ]] && ((spmm_cases += 1))
            if [[ "${line}" =~ ^Total:\ ([0-9]+)\ failed[[:space:]]*$ ]]; then
                ((spmm_totals += 1))
                spmm_total="${BASH_REMATCH[1]}"
            fi
        elif [[ "${op}" == spvv ]]; then
            if [[ "${line}" =~ ^ACL\ error\ ([0-9]+)\ at\ .+$ ]]; then
                ((spvv_errors += 1))
                error_code="${BASH_REMATCH[1]}"
                [[ "${error_code}" == 507033 ]] && spvv_507033=1
                [[ "${error_code}" == 507033 || "${error_code}" == 100002 ]] || invalid=1
            fi
            if [[ "${line}" =~ ^Total:\ [0-9]+\ passed,\ ([0-9]+)\ failed[[:space:]]*$ ]]; then
                ((spvv_totals += 1))
                spvv_total="${BASH_REMATCH[1]}"
            fi
            [[ "${line}" =~ aclsparseSpvv\ .*\ failed: ]] && invalid=1
        fi
    done < "${log_file}"

    (( !invalid && seen && startup && set_device )) || return 1
    if [[ "${op}" == spmm ]]; then
        (( spmm_cases > 0 && spmm_totals == 1 && spmm_total == spmm_cases ))
    elif [[ "${op}" == spvv ]]; then
        (( spvv_507033 && spvv_errors > 0 && spvv_totals == 1 && spvv_total == spvv_errors ))
    else
        return 1
    fi
}

PASSED_OPS=()
SKIPPED_OPS=()
FAILED_OPS=()
RECOVERED_OPS=()
SELECTED_COUNT=0

shopt -s nullglob
for test_dir in test/*/; do
    op="$(basename -- "${test_dir}")"
    src_arch_dir="sparse/${op}/${ARCH_DIR}"
    test_src="test/${op}/${ARCH_DIR}/${op}_test.cpp"
    if [[ ! -f "${test_src}" ]]; then
        # 兼容历史分支的 variant 嵌套目录；当前 arch22 主测试采用平铺路径。
        nested_test_src="test/${op}/${op}/${ARCH_DIR}/${op}_test.cpp"
        if [[ -f "${nested_test_src}" ]]; then
            test_src="${nested_test_src}"
        fi
    fi

    if [[ ! -d "${src_arch_dir}" || ! -f "${test_src}" ]]; then
        echo "[SKIP] ${op}: missing ${src_arch_dir} or ${test_src}" | tee -a "${RUN_LOG}"
        SKIPPED_OPS+=("${op}")
        continue
    fi

    SELECTED_COUNT=$((SELECTED_COUNT + 1))
    op_log="${LOG_DIR}/${op}.log"
    : > "${op_log}"
    echo "========== [RUN] ${op}: test=${test_src} ==========" | tee -a "${RUN_LOG}"
    mkdir -p -- "${ATTEMPT_ROOT}/${op}"
    attempt=1
    while :; do
        attempt_log="${ATTEMPT_ROOT}/${op}/attempt${attempt}.log"
        echo "[ATTEMPT] ${op}: ${attempt}/${MAX_ATTEMPTS}, log=${attempt_log}" | tee -a "${op_log}" "${RUN_LOG}"
        # 仅本次构建/测试子进程使用小 shape；直接运行 build.sh 默认保持完整规模。
        if SPARSE_TEST_PRESMOKE=1 bash build.sh --ops="${op}" --run --soc="${SOC_VERSION}" 2>&1 | tee "${attempt_log}" -a "${op_log}" "${RUN_LOG}"; then
            pipe_status=("${PIPESTATUS[@]}")
        else
            pipe_status=("${PIPESTATUS[@]}")
        fi
        build_status="${pipe_status[0]}"
        log_status="${pipe_status[1]}"
        if (( build_status == 0 && log_status == 0 )); then
            if has_pass_marker "${attempt_log}" "${op}"; then
                PASSED_OPS+=("${op}")
                if (( attempt > 1 )); then
                    RECOVERED_OPS+=("${op}")
                    echo "[RECOVERED] ${op}: passed on attempt ${attempt}" | tee -a "${op_log}" "${RUN_LOG}"
                else
                    echo "[FIRST_PASS] ${op}: passed on first attempt" | tee -a "${op_log}" "${RUN_LOG}"
                fi
            elif (( attempt > 1 )); then
                # 初始化失败后的重试必须真正执行通过，不能由 SKIP 抹掉先前失败。
                echo "[FAIL] ${op}: retry returned 0 without a test PASS marker" | tee -a "${op_log}" "${RUN_LOG}"
                FAILED_OPS+=("${op}")
            else
                echo "[SKIP] ${op}: build.sh returned 0 without a test PASS marker; no passing test confirmed" | tee -a "${RUN_LOG}"
                SKIPPED_OPS+=("${op}")
            fi
            break
        fi
        echo "[ATTEMPT_FAIL] ${op}: attempt=${attempt}, exit=${build_status}, log_exit=${log_status}" | tee -a "${op_log}" "${RUN_LOG}"
        if (( log_status != 0 || build_status == 137 )); then
            echo "[FAIL] ${op}: killed process or log write failure; no retry" | tee -a "${op_log}" "${RUN_LOG}"
            FAILED_OPS+=("${op}")
            break
        fi
        if ! initialization_only_failure "${attempt_log}" "${op}"; then
            echo "[FAIL] ${op}: failure is not confirmed initialization-only; no retry" | tee -a "${op_log}" "${RUN_LOG}"
            FAILED_OPS+=("${op}")
            break
        fi
        if (( attempt >= MAX_ATTEMPTS )); then
            echo "[FAIL] ${op}: initialization retry limit exhausted (${MAX_ATTEMPTS} attempts)" | tee -a "${op_log}" "${RUN_LOG}"
            FAILED_OPS+=("${op}")
            break
        fi
        delay=$((DELAY_MIN + RANDOM % (DELAY_MAX - DELAY_MIN + 1)))
        echo "[RETRY] ${op}: confirmed device startup timeout; wait ${delay}s before fresh process, attempt $((attempt + 1))/${MAX_ATTEMPTS}" | tee -a "${op_log}" "${RUN_LOG}"
        sleep "${delay}"
        attempt=$((attempt + 1))
    done
done
shopt -u nullglob

# ==============================
# 打包 slog 日志
# ==============================
mkdir -p /root/ascend/log
slog_name="slog.tar.gz"
tar -zcf "${slog_name}" -C /root/ascend log

rm -rf /root/ascend/log/*

# ==============================
# 检查 NPU 状态
# ==============================
log "checking NPU status ..."
mkdir -p ./npu_log
npu-smi info 2>&1 | tee ./npu_log/npu_info.log
if grep "dcmi module initialize failed" "./npu_log/npu_info.log"; then
    LOG_ERROR "dcmi module initialize failed"
    exit 1
fi

# ==============================
# 检查测试结果
# ==============================
log "checking test results ..."
date_time=$(date +%Y%m%d).$(date +%H%M%S)

{
    echo "========================================"
    echo "CI Test Summary (${SOC_VERSION} / ${ARCH_DIR}):"
    echo "  Selected: ${SELECTED_COUNT}"
    echo "  Passed:   ${#PASSED_OPS[@]} - ${PASSED_OPS[*]:-}"
    echo "  Skipped:  ${#SKIPPED_OPS[@]} - ${SKIPPED_OPS[*]:-}"
    echo "  Recovered: ${#RECOVERED_OPS[@]} - ${RECOVERED_OPS[*]:-}"
    echo "  Elapsed:  $((SECONDS - CI_START_SECONDS))s"
    echo "  Attempts: ${ATTEMPT_ROOT}"
    echo "  Failed:   ${#FAILED_OPS[@]} - ${FAILED_OPS[*]:-}"
    echo "========================================"
} | tee -a "${RUN_LOG}"

if [[ "${#FAILED_OPS[@]}" -gt 0 ]]; then
    LOG_ERROR "${date_time} : run test case failed"
    exit 1
fi
if [[ "${SELECTED_COUNT}" -gt 0 && "${#PASSED_OPS[@]}" -eq 0 ]]; then
    LOG_ERROR "${date_time} : ${SELECTED_COUNT} operator test(s) selected but none passed (skipped=${#SKIPPED_OPS[@]})"
    exit 1
fi
if [[ "${#PASSED_OPS[@]}" -eq 0 ]]; then
    LOG_INFO "${date_time} : no operator test passed; skipping is allowed (selected=${SELECTED_COUNT}, skipped=${#SKIPPED_OPS[@]})"
    exit 0
fi
LOG_INFO "${date_time} : run test case success: ${#PASSED_OPS[@]} operator tests passed, ${#SKIPPED_OPS[@]} skipped"
