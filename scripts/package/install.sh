#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -e

OPP_PLATFORM_DIR="ops_sparse"
PARAM_INVALID="0x0002"
FILE_NOT_EXIST="0x0080"

# 默认安装路径：root 用户 /usr/local/Ascend，普通用户 ~/Ascend
if [ "$(id -u)" != "0" ]; then
  DEFAULT_INSTALL_PATH="${HOME}/Ascend"
else
  DEFAULT_INSTALL_PATH="/usr/local/Ascend"
fi

# 解压目录：脚本位于 share/info/ops_sparse/install.sh，归档根目录为 ../../..
RUN_PKG_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TARGET_INSTALL_PATH="${DEFAULT_INSTALL_PATH}"
IS_INSTALL="n"
IS_UNINSTALL="n"
IS_QUIET="n"
PKG_VERSION_DIR="cann"

exitlog() {
  echo "[OpsSparse] [$(date '+%Y-%m-%d %H:%M:%S')] [INFO]: End Time: $(date '+%Y-%m-%d %H:%M:%S')"
}

logandprint() {
  echo "[OpsSparse] [$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 检查安装路径合法性（禁止明显非法字符如空格、<>|&$等）
check_install_path_valid() {
  local path="$1"
  case "${path}" in
    *' '*|*'<'*|*'>'*|*'|'*|*'&'*|*'$'*|*';'*|*'`'*)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

check_install_path() {
  TARGET_INSTALL_PATH="$1"
  if [ -z "${TARGET_INSTALL_PATH}" ]; then
    logandprint "[ERROR]: ERR_NO:${PARAM_INVALID}; --install-path cannot be empty."
    exitlog
    exit 1
  fi
  if [[ "${TARGET_INSTALL_PATH}" == *" "* ]]; then
    logandprint "[ERROR]: ERR_NO:${PARAM_INVALID}; --install-path cannot contain spaces."
    exitlog
    exit 1
  fi
  if ! check_install_path_valid "${TARGET_INSTALL_PATH}"; then
    logandprint "[ERROR]: ERR_NO:${PARAM_INVALID}; --install-path contains invalid characters."
    exitlog
    exit 1
  fi
  # 拒绝 ~user 形式（~ 后紧跟非 / 的字符，如 ~root、~root/path）；
  # 脚本仅支持 ~ 与 ~/...，不查询 passwd 做 ~user 展开
  if [[ "${TARGET_INSTALL_PATH}" =~ ^~[^/] ]]; then
    logandprint "[ERROR]: ERR_NO:${PARAM_INVALID}; --install-path does not support '~user' form (e.g. ~root); use an absolute path or ~/... instead."
    exitlog
    exit 1
  fi
  # 展开 ~（必须在转换为绝对路径之前，否则 ~/... 会被拼成 $PWD/~/...）
  if [[ "${TARGET_INSTALL_PATH}" == ~* ]]; then
    TARGET_INSTALL_PATH="${TARGET_INSTALL_PATH/#\~/$HOME}"
  fi
  # 转换为绝对路径
  if [[ "${TARGET_INSTALL_PATH}" != /* ]]; then
    TARGET_INSTALL_PATH="$(cd "${USER_PWD:-.}" && pwd)/${TARGET_INSTALL_PATH}"
  fi
  # 若路径为软链接，解析为实际路径
  if resolved=$(readlink -f "${TARGET_INSTALL_PATH}" 2>/dev/null) && [ -n "${resolved}" ]; then
    TARGET_INSTALL_PATH="${resolved}"
  fi
}

# 解析参数（makeself 将用户参数传给脚本：./xxx.run --install --install-path=/opt/ascend）
# makeself 可能将 run 文件路径作为第一个参数传入，需跳过
get_opts() {
  while [ $# -gt 0 ]; do
    # 跳过 makeself 传入的路径参数（如 --./xxx.run、--/path/to/dir、./xxx.run）
    # 注意：不能再用 *".run"* 过滤，否则 --install-path=/path/pkg.run 会被当作包路径整体跳过
    if [[ "$1" == ./* ]] || [[ "$1" == /* ]] || [[ "$1" == --./* ]] || [[ "$1" == --/* ]]; then
      shift
      continue
    fi
    case "$1" in
      --install)
        IS_INSTALL="y"
        shift
        ;;
      --uninstall)
        IS_UNINSTALL="y"
        shift
        ;;
      --install-path=*)
        check_install_path "${1#*=}"
        shift
        ;;
      --quiet)
        IS_QUIET="y"
        shift
        ;;
      --help|-h)
        show_help
        exit 0
        ;;
      -*)
        logandprint "[ERROR]: ERR_NO:${PARAM_INVALID}; Unsupported parameter: $1"
        exitlog
        exit 1
        ;;
      *)
        shift
        ;;
    esac
  done
}

show_help() {
  if [ -f "$(dirname "$0")/help.info" ]; then
    cat "$(dirname "$0")/help.info"
  else
    echo "Usage: $0 [options]"
    echo "  --install                 Install ops_sparse libraries"
    echo "  --install-path=<path>     Specify installation path (default: ${DEFAULT_INSTALL_PATH})"
    echo "  --uninstall               Uninstall ops_sparse"
    echo "  --quiet                   Quiet mode"
    echo "  --help                    Show this help"
  fi
}

# 获取目标安装目录（TARGET_INSTALL_PATH + PKG_VERSION_DIR）
get_target_version_dir() {
  echo "${TARGET_INSTALL_PATH}/${PKG_VERSION_DIR}"
}

# 临时授予目录写权限（参考 ops-nn RESET_MOD=750 方式）
# 记录被修改的目录到 DIRS_GRANTED_WRITE 数组，以便后续恢复
DIRS_GRANTED_WRITE=()

grant_write_if_needed() {
  local dir="$1"
  if [ -d "${dir}" ] && [ ! -w "${dir}" ]; then
    chmod u+w "${dir}" 2>/dev/null
    if [ $? -eq 0 ]; then
      DIRS_GRANTED_WRITE+=("${dir}")
    else
      logandprint "[WARNING]: Failed to grant write permission on ${dir}."
    fi
  fi
}

# 恢复之前临时授予写权限的目录
restore_permissions() {
  local i=${#DIRS_GRANTED_WRITE[@]}
  while [ $i -gt 0 ]; do
    i=$((i - 1))
    local dir="${DIRS_GRANTED_WRITE[$i]}"
    if [ -d "${dir}" ]; then
      if ! chmod u-w "${dir}" 2>/dev/null; then
        logandprint "[WARNING]: Failed to restore write permission on ${dir}."
      fi
    fi
  done
  DIRS_GRANTED_WRITE=()
}

trap 'restore_permissions' EXIT

do_install() {
  local target_dir
  target_dir=$(get_target_version_dir)
  local target_lib64="${target_dir}/lib64"
  local target_include="${target_dir}/include"
  local target_share="${target_dir}/share"
  local target_share_info="${target_dir}/share/info"
  local target_info="${target_dir}/share/info/${OPP_PLATFORM_DIR}"

  local src_lib64="${RUN_PKG_ROOT}/lib64"
  local src_include="${RUN_PKG_ROOT}/include"
  local src_info="${RUN_PKG_ROOT}/share/info/${OPP_PLATFORM_DIR}"

  if [ ! -d "${src_lib64}" ] || [ ! -d "${src_include}" ]; then
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; Run package is incomplete."
    exit 1
  fi

  if [ "${IS_QUIET}" != "y" ]; then
    logandprint "[INFO]: Installing ops_sparse to ${target_dir}"
    logandprint "[INFO]: Continue? [y/n]"
    read -r yn
    if [ "${yn}" != "y" ] && [ "${yn}" != "Y" ]; then
      logandprint "[INFO]: Installation cancelled."
      exit 0
    fi
  fi

  # lib64/include 可能是软链接（如 cann-9.0.0/lib64 -> x86_64-linux/lib64），
  # 需要解析到实际目录来检查权限
  local real_lib64="${target_lib64}"
  local real_include="${target_include}"
  if [ -L "${target_lib64}" ]; then
    real_lib64="$(readlink -f "${target_lib64}")"
  fi
  if [ -L "${target_include}" ]; then
    real_include="$(readlink -f "${target_include}")"
  fi

  # 临时授予写权限（参考 ops-nn 的 reset_mod_dirs 机制）
  grant_write_if_needed "${target_dir}"
  grant_write_if_needed "${real_lib64}"
  grant_write_if_needed "${real_include}"
  grant_write_if_needed "${target_share}"
  grant_write_if_needed "${target_share_info}"
  grant_write_if_needed "${target_info}"

  mkdir -p "${target_lib64}" "${target_include}" "${target_info}"

  # 复制 .so 文件，若未找到则报错
  so_count=0
  for f in "${src_lib64}"/*.so; do
    if [ -f "$f" ]; then
      cp -f "$f" "${real_lib64}/"
      so_count=$((so_count + 1))
    fi
  done
  if [ ${so_count} -eq 0 ]; then
    restore_permissions
    logandprint "[ERROR]: No .so files found in run package (expected in ${src_lib64})."
    exit 1
  fi

  # 复制头文件
  for f in "${src_include}"/*.h; do
    [ -f "$f" ] && cp -f "$f" "${real_include}/"
  done

  # 复制 version.info 等
  if [ -d "${src_info}" ]; then
    cp -rf "${src_info}"/* "${target_info}/" 2>/dev/null || true
  fi

  # 恢复被临时修改的目录权限
  restore_permissions

  logandprint "[INFO]: ops_sparse installed successfully to ${target_dir}"
  logandprint "[INFO]: Libraries: ${target_lib64}/ (${so_count} .so files)"
  logandprint "[INFO]: Headers:   ${target_include}/"
}

do_uninstall() {
  local target_dir
  target_dir=$(get_target_version_dir)
  local target_lib64="${target_dir}/lib64"
  local target_include="${target_dir}/include"
  local target_info="${target_dir}/share/info/${OPP_PLATFORM_DIR}"

  if [ ! -d "${target_dir}" ]; then
    logandprint "[INFO]: No installation found at ${target_dir}"
    exit 0
  fi

  if [ "${IS_QUIET}" != "y" ]; then
    logandprint "[INFO]: Uninstall ops_sparse from ${target_dir}? [y/n]"
    read -r yn
    if [ "${yn}" != "y" ] && [ "${yn}" != "Y" ]; then
      logandprint "[INFO]: Uninstall cancelled."
      exit 0
    fi
  fi

  local real_lib64="${target_lib64}"
  local real_include="${target_include}"
  if [ -L "${target_lib64}" ]; then
    real_lib64="$(readlink -f "${target_lib64}")"
  fi
  if [ -L "${target_include}" ]; then
    real_include="$(readlink -f "${target_include}")"
  fi

  # 临时授予写权限以便删除文件
  grant_write_if_needed "${real_lib64}"
  grant_write_if_needed "${real_include}"
  grant_write_if_needed "${target_dir}/share/info"
  grant_write_if_needed "${target_info}"

  if [ -d "${real_lib64}" ]; then
    rm -f "${real_lib64}"/libops_sparse*.so 2>/dev/null || true
  fi
  if [ -d "${real_include}" ]; then
    rm -f "${real_include}"/cann_ops_sparse*.h 2>/dev/null || true
  fi
  if [ -d "${target_info}" ]; then
    rm -rf "${target_info}"
  fi

  restore_permissions

  logandprint "[INFO]: ops_sparse uninstalled from ${target_dir}"
}

main() {
  logandprint "[INFO]: Start ops_sparse run package."
  get_opts "$@"

  if [ "${IS_INSTALL}" = "y" ] && [ "${IS_UNINSTALL}" = "y" ]; then
    logandprint "[ERROR]: Cannot use --install and --uninstall together."
    exit 1
  fi
  if [ "${IS_INSTALL}" != "y" ] && [ "${IS_UNINSTALL}" != "y" ]; then
    show_help
    exit 0
  fi

  if [ "${IS_INSTALL}" = "y" ]; then
    do_install
  else
    do_uninstall
  fi
  exitlog
}

main "$@"
