# Copyright (c) 2026 AscSpMM Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check host catalog entries against the compiled device specializations."""

import argparse
import json
import logging
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "sparse/cube_spmm_1d/arch22"


def require(condition, detail):
    if not condition:
        raise ValueError(str(detail))


def read_source(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def read_catalog():
    pattern = (r'\{Make(Workspace|Direct|Fusion)Tiling\((\d+),(\d+),(\d+),(\d+),'
               r'"[^"]+"\),(\d+),(true|false),(true|false),(true|false)\}')
    catalog = {}
    for _, nt, kt, pack, key, panel, ub, l0c, direct in re.findall(
            pattern, read_source("internal/spmm_1d_catalog.h")):
        key = int(key)
        require(key not in catalog, f"duplicate catalog key {key}")
        catalog[key] = dict(nt=int(nt), kt=int(kt), pack=int(pack), panel=int(panel),
                            ub=ub == "true", l0c=l0c == "true", direct=direct == "true")
    return catalog


def read_dispatch(symbols):
    macros, spaces, dispatch = {}, {}, {}
    cases = []
    for line in read_source("spmm_1d_kernel.cce").splitlines():
        match = re.match(r"#define\s+(SPMM_\w+)\s+(\d+)\s*$", line)
        if match:
            macros[match[1]] = int(match[2])
        match = re.match(r"#undef\s+(SPMM_\w+)", line)
        if match:
            macros.pop(match[1], None)
        match = re.match(r"namespace\s+(\w+)\s*\{", line)
        if match:
            spaces[match[1]] = dict(macros)
        match = re.search(r"case\s+(SPMM_TILING_\w+):", line)
        if match:
            require(match[1] in symbols, f"unknown tiling symbol {match[1]}")
            cases.append(symbols.get(match[1]))
        match = re.search(r"(\w+)::run_(aic|aiv)\(", line)
        if match:
            target, side = match.groups()
            require(target in spaces, f"unknown device namespace {target}")
            assigned = cases if cases else ([9] if target.startswith("direct_") else [])
            for key in assigned:
                dispatch[key, side] = spaces.get(target)
        if re.search(r"\bbreak;", line):
            cases = []
    return dispatch


def validate_fusion(key, entry):
    group = {48: 2, 49: 4, 50: 8}.get(key)
    require(group is not None, f"invalid fusion key {key}")
    capacity = min(64 * group, 2048 // group)
    expected = dict(nt=32 // group, kt=capacity, pack=32, panel=32, ub=True, l0c=True, direct=False)
    require(entry == expected, (key, entry, expected))
    require(2 * (512 // group) * capacity <= 65536, (key, "L0A"))
    require(2 * (16 * group) * capacity <= 65536, (key, "L0B"))
    require(2 * 4 * (16 * group) * (512 // group) <= 65536, (key, "L0C"))
    cmake = read_source("build.cmake")
    require("fusion2 fusion4 fusion8" in cmake, "missing fusion build family")
    require("SPMM_FUSION_GROUP=${CMAKE_MATCH_1}" in cmake, "missing fusion build definition")
    device = read_source("spmm_1d_fusion.cce")
    require("constexpr int FUSION_MAX = SPMM_FUSION_GROUP;" in device, "missing fusion group")
    require("can_fuse" in device and "load_cbuf_to_cb" in device, "missing fusion device path")


def validate_side(key, entry, side, dispatch):
    actual = dispatch.get((key, side))
    require(actual is not None, (key, side, "missing specialization"))
    fields = {"nt": "SPMM_AIC_N_BATCH", "kt": "SPMM_SPARSE_K_TILE",
              "pack": "SPMM_B_PACK_N_BATCH", "direct": "SPMM_DIRECT_B_GATHER_AIC"}
    for field, macro in fields.items():
        require(entry.get(field) == actual.get(macro), (key, side, field))
    if side == "aiv":
        require(entry.get("ub") == actual.get("SPMM_ENABLE_AIV_PANEL_PIPELINE"), (key, side, "ub"))
        require(entry.get("panel") == actual.get("SPMM_B_PACK_SUB_TILE", 16), (key, side, "panel"))
    else:
        require(entry.get("l0c") == actual.get("SPMM_ENABLE_L0C_PIPELINE"), (key, side, "l0c"))


def validate():
    pattern = r"(SPMM_TILING_\w+)\s*=\s*(\d+)"
    matches = re.findall(pattern, read_source("internal/spmm_1d_tiling_keys.h"))
    symbols = {name: int(value) for name, value in matches}
    catalog = read_catalog()
    require(set(catalog) == set(symbols.values()), "catalog/enum key mismatch")
    dispatch = read_dispatch(symbols)
    for key, entry in catalog.items():
        if key in (48, 49, 50):
            validate_fusion(key, entry)
        else:
            for side in ("aic", "aiv"):
                validate_side(key, entry, side, dispatch)
    return dict(keys=len(catalog), device_sides=len(dispatch) + 6, status="PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = validate()
    if not args.quiet:
        logging.info("%s", json.dumps(result))
