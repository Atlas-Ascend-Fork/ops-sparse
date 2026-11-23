# SpSV on Ascend 950

The implementation keeps host planning, device execution and their shared layout separate.

| Directory | Responsibility |
| --- | --- |
| `op_host/` | Public sparse API, parameter validation, workspace allocation, tiling and schedule cache |
| `shared/` | Tiling fields and workspace offset definitions used at the host/device boundary |
| `op_kernel/analysis/` | Format conversion, level construction and analysis entry points |
| `op_kernel/solve/` | FP32 and Complex64 row arithmetic and level traversal |
| `op_kernel/dispatch/` | Schedule selection, entry preparation and type/orientation dispatch |
| `op_kernel/common/` | Shared publication, cache refresh and schedule contracts |
| `op_kernel/data/` | Updates and vector copy/scale operations |
| `op_kernel/validation/` | Forward-error bounds and proposal, prefix, border and scaled certificates |
| `op_kernel/math/` | Compensated, exponent-scaled, integer and software binary64 arithmetic |
| `op_kernel/recovery/` | Dyadic dependency reconstruction and exact CRT recovery |

`op_kernel/spsv_kernel.cpp` owns the scheduled kernel entries and launchers. The dyadic and CRT recovery entries remain separate translation units, as do the float prefix, complex scaled, real scaled and binary64 entries. Every device translation unit retains `-ffp-contract=off -fno-fast-math` in the root CMake configuration.

Tests use one CSV-driven parameterized suite and the shared `spsv_npu_wrapper.h` analysis/update/solve driver. The named regression headers retain independent assertions for rounding, cancellation, predecessor error, block boundaries, aliasing and descriptor reuse. `test/spsv/arch35/spsv_test.csv` remains the executable case manifest.

Dated measurements and historical validation notes are maintained in the PR validation reports. Historical records do not validate a later source revision.
