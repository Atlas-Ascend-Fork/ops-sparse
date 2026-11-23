/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>

#include "test_common.h"

#include "../spsv_golden.h"
#include "../spsv_param.h"
#include "spsv_npu_wrapper.h"
#include "../../../sparse/spsv/arch35/shared/spsv_tiling_data.h"

namespace sparse_test {

TEST(SpsvComplexSuffixArithmeticTest, WidensBeforeAdding)
{
    constexpr int32_t kMaxRows = std::numeric_limits<int32_t>::max();
    constexpr int32_t kSuffixLevels = 1 << 23;
    constexpr int32_t kPrefix = 1;
    constexpr int32_t kSuffix = kSuffixLevels + kPrefix;
    constexpr int32_t kFirstWidth = kMaxRows - kSuffixLevels;
    constexpr int64_t kEnd = SpsvComplexSingletonSuffixBegin(kFirstWidth, kSuffix, kPrefix);

    EXPECT_EQ(kEnd, static_cast<int64_t>(kMaxRows));
    EXPECT_EQ(kEnd - 1, static_cast<int64_t>(kMaxRows) - 1);
    EXPECT_LE(kEnd, static_cast<int64_t>(kMaxRows));
}

// RAII guard for environment variables to ensure cleanup even on exceptions
class RAIIEnvGuard {
public:
    RAIIEnvGuard() : needs_cleanup_(false) {}
    ~RAIIEnvGuard() { cleanup(); }

    void set(const char* name, const char* value) {
        if (needs_cleanup_) {
            unsetenv(name_.c_str());
        }
        name_ = name;
        value_ = value;
        setenv(name, value, 1);
        needs_cleanup_ = true;
    }

    void cleanup() {
        if (needs_cleanup_) {
            unsetenv(name_.c_str());
            needs_cleanup_ = false;
        }
    }

private:
    std::string name_;
    std::string value_;
    bool needs_cleanup_;
};

// The identity except for a full last row.  That row has m nonzeros, so it
// goes to the cooperative wide-row solver.
static CsrMatrix MakeCancelWideCsr(int64_t m) {
    CsrMatrix out;
    out.rows = m;
    out.cols = m;
    out.nnz = m - 1 + m;
    out.rowOffsets.resize(m + 1);
    out.colIndices.reserve(out.nnz);
    out.values.assign(out.nnz, 1.0f);
    for (int64_t i = 0; i < m - 1; i++) {
        out.rowOffsets[i] = static_cast<int32_t>(i);
        out.colIndices.push_back(static_cast<int32_t>(i));
    }
    out.rowOffsets[m - 1] = static_cast<int32_t>(m - 1);
    for (int64_t j = 0; j < m; j++) {
        out.colIndices.push_back(static_cast<int32_t>(j));
    }
    out.rowOffsets[m] = static_cast<int32_t>(out.nnz);
    return out;
}

// Lower bidiagonal: row 0 is its own diagonal, every later row has the
// subdiagonal and its diagonal.  Two rows of it are what the pre-divided
// finish needs -- row 1's right-hand side leaves range and is taken off in
// the precompute, so the row reaches the finish with a finite dot product
// and a diagonal scale of two.
static CsrMatrix MakeBidiagCsr(int64_t m) {
    CsrMatrix out;
    out.rows = m;
    out.cols = m;
    out.nnz = 2 * m - 1;
    out.rowOffsets.resize(m + 1);
    out.rowOffsets[0] = 0;
    out.colIndices.push_back(0);
    out.values.push_back(1.0f);
    for (int64_t i = 1; i < m; i++) {
        out.rowOffsets[i] = static_cast<int32_t>(out.colIndices.size());
        out.colIndices.push_back(static_cast<int32_t>(i - 1));
        out.values.push_back(1.0f);
        out.colIndices.push_back(static_cast<int32_t>(i));
        out.values.push_back(2.0f);
    }
    out.rowOffsets[m] = static_cast<int32_t>(out.nnz);
    return out;
}

// The shapes the numerical regressions use.  Each is a diagonal whose
// magnitude puts some intermediate of the complex solve near or past the
// edge of fp32 while the answer stays inside it; which intermediate, and
// what x and alpha go with it, is in the predicates further down.
//
//   huge / tiny / max   the square of the diagonal leaves range: 1e20
//                       squares to +Inf, 1e-30 to zero, and 2^127 overflows
//                       even the scaled denominator if it sums components
//   overflow_rhs        |rhs| = 4.2e38 is already past range
//   small_scale         scale below one, so taking it off a large residual
//                       first is a growth
//   big_alpha           alpha * x leaves range, alpha * x / diag does not
//   big_alpha_small_d   the larger component is 0.75 but |diag| is 1.06
//   big_alpha_cplx      a large complex alpha where the recovery must divide
//                       before it multiplies
//   crossterm           alpha = F + i against x = 2 + iF: the small component
//                       of alpha is what puts F into the real part, so a
//                       scaling that flushes it is wrong by one and finite
//   cancel_*            the identity with a full last row, whose dot product
//                       cancels against the right-hand side
//   divfirst_cplx       alpha / diag is unrepresentable while alpha * x / diag
//                       is not, so no operand may be divided on its own
//   predivdot_cplx      a finite dot product the pre-divided finish used to
//                       overflow on
//   warpsum             a row whose butterfly partial sums leave range while
//                       every term, and the answer, stay inside it
//
// Independent two-row blocks B = [[1,0],[2,1]]: the rows that leave range
// have no dependency on one another.
static CsrMatrix MakeTwoRowBlocksCsr(int64_t m) {
    CsrMatrix out;
    out.rows = m;
    out.cols = m;
    out.rowOffsets.assign(static_cast<size_t>(m + 1), 0);
    for (int64_t i = 0; i < m; i++) {
        if (i % 2 != 0) {
            out.colIndices.push_back(static_cast<int32_t>(i - 1));
            out.values.push_back(2.0f);
        }
        out.colIndices.push_back(static_cast<int32_t>(i));
        out.values.push_back(1.0f);
        out.rowOffsets[static_cast<size_t>(i + 1)] =
            static_cast<int32_t>(out.colIndices.size());
    }
    out.nnz = static_cast<int64_t>(out.colIndices.size());
    return out;
}

// Rows 0..m-2 are unit diagonal rows and the last is all ones with a diagonal
// of two.  At m = 3 and m = 5 these are the review's finite counter-examples.
static CsrMatrix MakeTailDiagTwoCsr(int64_t m) {
    CsrMatrix out;
    out.rows = m;
    out.cols = m;
    out.rowOffsets.assign(static_cast<size_t>(m + 1), 0);
    for (int64_t i = 0; i + 1 < m; i++) {
        out.colIndices.push_back(static_cast<int32_t>(i));
        out.values.push_back(1.0f);
        out.rowOffsets[static_cast<size_t>(i + 1)] =
            static_cast<int32_t>(out.colIndices.size());
    }
    for (int64_t j = 0; j < m; j++) {
        out.colIndices.push_back(static_cast<int32_t>(j));
        out.values.push_back((j + 1 == m) ? 2.0f : 1.0f);
    }
    out.rowOffsets[static_cast<size_t>(m)] =
        static_cast<int32_t>(out.colIndices.size());
    out.nnz = static_cast<int64_t>(out.colIndices.size());
    return out;
}

// Those blocks, then one independent three-row block of the shape above.
static CsrMatrix MakeBlocksThenTailCsr(int64_t m) {
    CsrMatrix out = MakeTwoRowBlocksCsr(m - 3);
    CsrMatrix tail = MakeTailDiagTwoCsr(3);
    out.rows = m;
    out.cols = m;
    for (int64_t i = 0; i < 3; i++) {
        int32_t from = tail.rowOffsets[static_cast<size_t>(i)];
        int32_t to = tail.rowOffsets[static_cast<size_t>(i + 1)];
        for (int32_t k = from; k < to; k++) {
            out.colIndices.push_back(static_cast<int32_t>(
                m - 3 + tail.colIndices[static_cast<size_t>(k)]));
            out.values.push_back(tail.values[static_cast<size_t>(k)]);
        }
        out.rowOffsets.push_back(static_cast<int32_t>(out.colIndices.size()));
    }
    out.nnz = static_cast<int64_t>(out.colIndices.size());
    return out;
}

// [[1,0],[1e10,1e10]]: a product of two finite entries that is not finite.
static CsrMatrix MakeProdOverflowCsr() {
    CsrMatrix out;
    out.rows = 2;
    out.cols = 2;
    out.rowOffsets = {0, 1, 3};
    out.colIndices = {0, 0, 1};
    out.values = {1.0f, 1.0e10f, 1.0e10f};
    out.nnz = 3;
    return out;
}

// [[1,0,0],[0,1,0],[1,1,1.5*2^127]]: the dot product leaves range and the
// diagonal is deliberately not a power of two, so a scale whose mantissa is
// dropped shows up.  The answer is -4/3.
static CsrMatrix MakeScaleMantissaCsr() {
    CsrMatrix out;
    out.rows = 3;
    out.cols = 3;
    out.rowOffsets = {0, 1, 2, 5};
    out.colIndices = {0, 1, 0, 1, 2};
    out.values = {1.0f, 1.0f, 1.0f, 1.0f, 2.55211775e38f};   // 1.5 * 2^127
    out.nnz = 5;
    return out;
}

// [[1,0,0],[0,1,0],[2,-2,2^127]]: the dependencies cancel and the answer is
// exactly the smallest normal.
static CsrMatrix MakeCancelMinNormalCsr() {
    CsrMatrix out;
    out.rows = 3;
    out.cols = 3;
    out.rowOffsets = {0, 1, 2, 5};
    out.colIndices = {0, 1, 0, 1, 2};
    out.values = {1.0f, 1.0f, 2.0f, -2.0f, 1.70141183e38f};
    out.nnz = 5;
    return out;
}

// Identity except for a full last row of [F, -F, t, 1], with F = 2^127 and
// t = 2^-126.  Against X = [1, 1, F, 0] the row's two large products cancel
// and the small one, t * Y[2] = 2, is the whole answer -- so a single shift
// for the row cannot hold both: bringing F * Y[0] into range costs at least
// 127 bits, and t is then below what fp32 can represent at all.
static CsrMatrix MakeRowScaleCsr() {
    CsrMatrix out;
    out.rows = 4;
    out.cols = 4;
    out.rowOffsets = {0, 1, 2, 3, 7};
    out.colIndices = {0, 1, 2, 0, 1, 2, 3};
    out.values = {1.0f, 1.0f, 1.0f,
                  1.70141183e38f, -1.70141183e38f, 1.1754944e-38f, 1.0f};
    out.nnz = 7;
    return out;
}

// Two rows, the second with one off-diagonal term: the row's single shift
// has to serve that term and a right-hand side at the other end of the
// range at once.  The first row solves on its own and sets what the second
// one's term is multiplied by.
static CsrMatrix MakeTwoRowRangeCsr(float d0, float a10, float d1) {
    CsrMatrix out;
    out.rows = 2;
    out.cols = 2;
    out.rowOffsets = {0, 1, 3};
    out.colIndices = {0, 0, 1};
    out.values = {d0, a10, d1};
    out.nnz = 3;
    return out;
}

// The mirror of MakeRowScaleCsr: the small entry is 2^-30 rather than
// 2^-126 and it is the large Y that the big entries meet, so the row's two
// large products cancel exactly and the whole answer is the small term.
static CsrMatrix MakeRowSpanCsr() {
    CsrMatrix out;
    out.rows = 4;
    out.cols = 4;
    out.rowOffsets = {0, 1, 2, 3, 7};
    out.colIndices = {0, 1, 2, 0, 1, 2, 3};
    out.values = {1.0f, 1.0f, 1.0f,
                  1.70141183e38f, -1.70141183e38f, 9.31322575e-10f, 1.0f};
    out.nnz = 7;
    return out;
}

// A row whose two large products cancel and whose answer is a third product
// the row's shift KEEPS.  At m = 4 the row fits one entry per warp lane and
// the kept product is 24 * 1.5; at m = 36 it does not and the kept product is
// 24 * 24, the filler columns solving to zero.  Both land on exactly the
// smallest normal once scaled, which is the sharpest place to ask whether a
// recovery can tell kept from emptied.
static CsrMatrix MakeKeptTermCsr(int64_t m) {
    const float kF = 1.70141183e38f;
    CsrMatrix out;
    out.rows = m;
    out.cols = m;
    for (int64_t i = 0; i + 1 < m; i++) {
        out.rowOffsets.push_back(i);
        out.colIndices.push_back(i);
        out.values.push_back(1.0f);
    }
    out.rowOffsets.push_back(m - 1);
    out.rowOffsets.push_back(2 * m - 1);
    for (int64_t j = 0; j < m; j++) {
        out.colIndices.push_back(j);
        out.values.push_back(j == 0 ? kF : (j == 1 ? -kF : (j == 2 ? 24.0f : 1.0f)));
    }
    out.nnz = 2 * m - 1;
    return out;
}

// Two 2^179 products cancel, then the RHS cancels 2^127. The remaining
// product is 2^-70 (subnormal after scaling) or 2^-60 (normal after scaling).
// Dividing by the same diagonal gives exactly -1 in either case. Wider rows
// also exercise lane-local accumulation and the cooperative-row retry.
static CsrMatrix MakeRhsCancelCsr(int64_t m = 5, bool normal = false) {
    CsrMatrix out = MakeCancelWideCsr(m);
    const size_t tail = static_cast<size_t>(m - 1);
    out.values[tail] = std::ldexp(1.0f, 90);
    out.values[tail + 1] = -std::ldexp(1.0f, 90);
    out.values[tail + 2] = std::ldexp(1.0f, 127);
    out.values[tail + 3] = std::ldexp(1.0f, normal ? -60 : -70);
    out.values.back() = out.values[tail + 3];
    return out;
}

// A separate lost-product regression: scaling Y underflows before the multiply,
// whereas MakeRhsCancelCsr keeps a nonzero product and loses it during summation.
static CsrMatrix MakeRhsCancelFloatCsr() {
    CsrMatrix out;
    out.rows = 3;
    out.cols = 3;
    out.rowOffsets = {0, 1, 2, 5};
    out.colIndices = {0, 1, 0, 1, 2};
    out.values = {1.7014118e38f, 1.7014118e38f,   // 2^127 diagonals
                  1.7014118e38f,                  // 2^127, the term kept
                  9.536743e-07f,                  // 2^-20, the term emptied
                  7.523164e-37f};                 // 2^-120 on the diagonal
    out.nnz = 5;
    return out;
}

static CsrMatrix MakeMultiLevelInplaceCsr(int64_t m) {
    CsrMatrix out;
    out.rows = m;
    out.cols = m;
    out.rowOffsets.reserve(static_cast<size_t>(m + 1));
    out.rowOffsets.push_back(0);
    for (int64_t row = 0; row < m; ++row) {
        if (row >= 64) {
            out.colIndices.push_back(static_cast<int32_t>(row - 64));
            out.values.push_back(1.0f);
        }
        out.rowOffsets.push_back(static_cast<int32_t>(out.values.size()));
    }
    out.nnz = static_cast<int64_t>(out.values.size());
    return out;
}

// The three the recovery's own review added, apart from MakeRangeShapeCsr so
// that neither function grows past what the checker allows.
static bool MakeRecoveryShapeCsr(const SpSVParam& p, CsrMatrix& out) {
    if (p.structure == "multilevel_inplace") {
        out = MakeMultiLevelInplaceCsr(p.m);
        return true;
    }
    if (p.structure == "keptterm") {
        out = MakeKeptTermCsr(p.m);
        return true;
    }
    if (p.structure == "rhscancel" || p.structure == "rhscancel_normal") {
        out = MakeRhsCancelCsr(p.m, p.structure == "rhscancel_normal");
        return true;
    }
    if (p.structure == "rhscancelf") {
        out = MakeRhsCancelFloatCsr();
        return true;
    }
    return false;
}

// The shapes built for range behaviour, kept together and apart from the
// numeric shapes so that neither function grows past what the checker allows.
static bool MakeRangeShapeCsr(const SpSVParam& p, CsrMatrix& out) {
    if (p.structure == "rowscale4") {
        out = MakeRowScaleCsr();
        return true;
    }
    if (p.structure == "rowspan4") {
        out = MakeRowSpanCsr();
        return true;
    }
    if (p.structure == "blocks33") {
        out = MakeTwoRowBlocksCsr(p.m);
        return true;
    }
    if (p.structure == "blocks32c") {
        out = MakeBlocksThenTailCsr(p.m);
        return true;
    }
    if (p.structure == "prodovf") {
        out = MakeProdOverflowCsr();
        return true;
    }
    if (p.structure == "cancelminnorm") {
        out = MakeCancelMinNormalCsr();
        return true;
    }
    if (p.structure == "scalemant") {
        out = MakeScaleMantissaCsr();
        return true;
    }
    if (p.structure == "rhsflush") {
        out = MakeTwoRowRangeCsr(p.alpha, 4.0f, 4.0f);
        return true;
    }
    if (p.structure == "rhsbigexp") {
        out = MakeTwoRowRangeCsr(1.34217728e8f, 1.0f, 1.70141183e38f);
        return true;
    }
    if (p.structure == "fp32_tail") {
        out = MakeTailDiagTwoCsr(p.m);
        return true;
    }
    if (p.structure == "predivdot_cplx") {
        out = MakeBidiagCsr(p.m);
        return true;
    }
    return MakeRecoveryShapeCsr(p, out);
}

// Returns false when p.structure is not one of them.
static bool MakeNumericCsr(const SpSVParam& p, CsrMatrix& out) {
    if (MakeRangeShapeCsr(p, out)) {
        return true;
    }
    if (p.structure == "warpsum") {
        out = MakeCancelWideCsr(p.m);
        // The float paths read these directly, where the complex ones get
        // MakeWarpSumValues instead; same alternating pairs either way, and
        // the last row's diagonal stays at one.
        for (int64_t j = 0; j + 1 < p.m; j++) {
            out.values[static_cast<size_t>(p.m - 1 + j)] =
                ((j / 2) % 2 == 0) ? 3.0e38f : -3.0e38f;
        }
        return true;
    }
    if (p.structure == "cancel_wide" || p.structure == "cancel_short") {
        out = MakeCancelWideCsr(p.m);
        return true;
    }
    // The review's range cases: every row but the last is an identity row,
    // and the last depends on all of them.  Only its diagonal differs.
    if (p.structure == "restore5" || p.structure == "shortacc3") {
        out = MakeCancelWideCsr(p.m);
        out.values[static_cast<size_t>(out.nnz - 1)] = 2.0f;
        return true;
    }
    if (p.structure == "shortrhs3") {
        out = MakeCancelWideCsr(p.m);
        return true;
    }
    struct DiagShape { const char *name; float diag; };
    static const DiagShape kShapes[] = {
        {"huge_diag", 1.0e20f},
        {"tiny_diag", 1.0e-30f},
        {"max_diag", 1.7014118e38f},      // 2^127
        {"overflow_rhs", 2.0f},
        {"small_scale", 0.5f},
        {"big_alpha", 4.0f},
        {"big_alpha_small_d", 0.75f},
        {"big_alpha_cplx", 2.0f},
        {"crossterm", 1.70141183e38f},     // 2^127; see the note above
        {"divfirst_cplx", 1.0f},
        {"minnormal", 1.70141183e38f},     // 2^127, with x = 2 the answer is 2^-126
    };
    for (const DiagShape& sh : kShapes) {
        if (p.structure == sh.name) {
            out = makeDiagCsr(p.m, sh.diag);
            return true;
        }
    }
    return false;
}

// The same matrix with one extra entry in its last row whose column is out
// of range.  Both COO builders have to skip it, so a solve over this must
// give what the matrix without it gives -- the golden is computed from the
// clean one.  Only the row index was checked on the parallel path, so the
// illegal column reached the workspace CSR and the solve's prefix loop
// indexed vecY with it.
static CsrMatrix AppendOutOfRangeColumn(const CsrMatrix& csr, bool lower) {
    CsrMatrix out = csr;
    // Which way out of range matters.  The solve's prefix loop takes an entry
    // when col < row going forward and col > row going backward, so a column
    // past the end is never read by a forward solve even when it is wrongly
    // stored -- it has to be negative to be reached.  Backward is the mirror.
    const int32_t bad = lower ? -5 : static_cast<int32_t>(csr.rows + 5);
    // It also has to land in the row's dependency prefix to be read at all.
    // SpsvComputeDiagAndValidCount counts the LEADING run of entries the row
    // depends on, so an entry placed after the diagonal is past the prefix and
    // never touched even when the builder wrongly stores it.  A forward
    // solve's dependencies are the low columns, so the entry goes first.
    const size_t at = lower
        ? static_cast<size_t>(csr.rowOffsets[static_cast<size_t>(csr.rows - 1)])
        : out.colIndices.size();
    out.colIndices.insert(out.colIndices.begin() + static_cast<long>(at), bad);
    out.values.insert(out.values.begin() + static_cast<long>(at), 1.0f);
    out.rowOffsets[static_cast<size_t>(csr.rows)] =
        static_cast<int32_t>(out.colIndices.size());
    out.nnz = static_cast<int64_t>(out.colIndices.size());
    return out;
}

static CsrMatrix GenerateTriangularMatrix(const SpSVParam& p) {
    bool lower = p.isLower();
    bool unitDiag = p.isUnitDiag();

    if (p.structure == "diagonal") {
        return makeDiagCsr(p.m, unitDiag ? 1.0f : 3.0f);
    }
    if (p.structure == "tridiagonal") {
        return makeTridiagTriangularCsr(p.m, lower, p.seed);
    }
    if (p.structure == "identity") {
        return makeIdentityCsr(p.m);
    }
    if (p.structure == "diag_dominant") {
        return makeDiagDominantTriangularCsr(p.m, lower, p.seed);
    }
    if (p.structure == "empty_unit" || p.structure == "empty_non_unit") {
        // nnz=0 matrix: no stored values at all (no diagonal entries either).
        // For UNIT: y[i] = rhs[i] / 1 = alpha*x[i]   (identity-like)
        // For NON_UNIT: y[i] = rhs[i] / 0 = Inf/NaN   (singular)
        return makeEmptyTriangularCsr(p.m);
    }
    if (p.structure == "missing_diag") {
        return makeMissingDiagTriangularCsr(p.m, lower, p.seed);
    }
    CsrMatrix extreme;
    if (MakeNumericCsr(p, extreme)) {
        return extreme;
    }
    if (p.sparsity <= 0.0) {
        return makeTriangularCsr(p.m, lower, unitDiag, 0.0, p.seed);
    }
    return makeTriangularCsr(p.m, lower, unitDiag, p.sparsity, p.seed);
}

class SpSVTestBase {
protected:
    static void InitSuite(std::unique_ptr<AclEnvScope>& env,
                          std::unique_ptr<HandleManager>& handle) {
        env = std::make_unique<AclEnvScope>();
        handle = std::make_unique<HandleManager>();
        handle->setStream(env->stream());
    }
    static void CleanupSuite(std::unique_ptr<AclEnvScope>& env,
                             std::unique_ptr<HandleManager>& handle) {
        handle.reset();
        env.reset();
    }
};

class SpSVTest : public SpSVTestBase, public ::testing::TestWithParam<SpSVParam> {
public:
    static void SetUpTestSuite() { InitSuite(env_, spHandle_); }
    static void TearDownTestSuite() { CleanupSuite(env_, spHandle_); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    inline static std::unique_ptr<HandleManager> spHandle_;
};

static void VerifyResult(const std::vector<float>& actual,
                         const std::vector<float>& golden,
                         const SpSVParam& p,
                         const std::string& label) {
    VerifyConfig verifyCfg;
    verifyCfg.SetMode(PrecisionMode::MERE_MARE)
             .SetMERE(p.mere_threshold)
             .SetMARE(p.mare_multiplier);
    EXPECT_TRUE(Verifier::verifyVector(actual, golden, verifyCfg, label))
        << "[" << label << "] Precision verification FAILED";
}

static void RunUpdateVerification(const SpSVNpuResult& npuResult,
                                  const CsrMatrix& csr,
                                  const std::vector<float>& xVec,
                                  const SpSVParam& p,
                                  bool lower, bool unitDiag, bool transpose) {
    if (!npuResult.hasSecondSolve) return;
    if (p.update_mode != "GENERAL" && p.update_mode != "DIAGONAL") return;
    CsrMatrix csrUpdated = csr;
    std::mt19937 rng(9999);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    if (p.update_mode == "GENERAL") {
        for (int64_t i = 0; i < p.m; i++) {
            float rowAbsSum = 0.0f;
            int32_t diagIdx = -1;
            for (int32_t k = csrUpdated.rowOffsets[i]; k < csrUpdated.rowOffsets[i + 1]; k++) {
                if (csrUpdated.colIndices[k] == static_cast<int32_t>(i)) {
                    diagIdx = k;
                } else {
                    float v = dist(rng);
                    csrUpdated.values[k] = v;
                    rowAbsSum += std::abs(v);
                }
            }
            if (diagIdx >= 0) {
                csrUpdated.values[diagIdx] = rowAbsSum + 1.0f;
            }
        }
    } else {
        for (int64_t i = 0; i < p.m; i++) {
            for (int32_t k = csrUpdated.rowOffsets[i]; k < csrUpdated.rowOffsets[i + 1]; k++) {
                if (csrUpdated.colIndices[k] == static_cast<int32_t>(i)) {
                    csrUpdated.values[k] = dist(rng);
                }
            }
        }
    }
    auto yGolden2 = SpSVGolden(csrUpdated, xVec, p.alpha, lower, unitDiag, transpose);
    VerifyResult(npuResult.y2, yGolden2, p, p.case_name + "_update");
}

static void RunDeterministicCheck(aclsparseHandle_t handle, aclrtStream stream,
                                   const CsrMatrix& csr, const std::vector<float>& xVec,
                                   const SpSVParam& p, const SpSVNpuConfig& cfg,
                                   const SpSVNpuResult& npuResult) {
    if (p.case_name.find("deterministic") == std::string::npos) return;
    SpSVNpuResult npuResult2;
    try {
        npuResult2 = SpSVNpuWrapper(handle, stream, csr, xVec, p.alpha, cfg);
    } catch (const std::exception& e) {
        FAIL() << "[" << p.case_name << "] 2nd NPU execution threw: " << e.what();
    }
    VerifyConfig exactCfg;
    exactCfg.SetMode(PrecisionMode::EXACT);
    EXPECT_TRUE(Verifier::verifyVector(npuResult2.y, npuResult.y, exactCfg,
                                       p.case_name + "_deterministic"))
        << "[" << p.case_name << "] Deterministic check FAILED";
}

static std::vector<float> MakeFp32TailInputVector(const SpSVParam& p) {
    std::vector<float> x(static_cast<size_t>(p.m), 0.0f);
    float v = (p.m <= 3) ? 1.70141183e38f : 8.50705917e37f;
    for (int64_t i = 0; i + 1 < p.m; i++) { x[static_cast<size_t>(i)] = v; }
    return x;
}

static std::vector<float> GenerateInputVector(const SpSVParam& p) {
    if (p.structure == "fp32_tail") { return MakeFp32TailInputVector(p); }
    if (p.structure == "rhsflush") {
        // the first row solves to 2^127 for any alpha, since its diagonal is
        // alpha; alpha * X[1] is then 2*alpha^2, which fp32 cannot represent
        // and the second row has to treat as the zero it would have been
        return {1.70141183e38f, 2.0f * p.alpha};
    }
    if (p.structure == "rhsbigexp") {
        // alpha * X[1] = 2^254 is not finite while the quotient 2^127 is, so
        // the shift comes from the right-hand side and wants 131 bits
        return {1.0f, 1.70141183e38f};
    }
    if (p.structure == "big_alpha") {
        // alpha comes from the case (2e38) and the diagonal is 4, so alpha*x
        // is 4e38 and out of range while the answer 1e38 is not.  The complex
        // twin of this is CX_REG_big_alpha_m1; this is the real one.
        return std::vector<float>(static_cast<size_t>(p.m), 2.0f);
    }
    if (p.structure == "warpsum") {
        // Every earlier row solves to one, so the last row's dot product is
        // the +-3e38 pattern summed.  Its right-hand side has to clear the
        // reference's own rounding: 3e38 has an ulp of 2^76 in double and
        // 2^105 in fp32, and 1e33 is above both.
        std::vector<float> x(static_cast<size_t>(p.m), 1.0f);
        x[static_cast<size_t>(p.m - 1)] = 1.0e33f;
        return x;
    }
    if (p.structure == "keptterm") {
        // F * F leaves fp32 range on the fast path, so the row reaches the
        // scaled fallback; see MakeRangeGuardX for what the row then forms
        std::vector<float> x(static_cast<size_t>(p.m), 0.0f);
        x[0] = 1.70141183e38f;
        x[1] = 1.70141183e38f;
        x[2] = (p.m <= 4) ? 1.5f : 24.0f;
        return x;
    }
    if (p.structure == "rhscancel" || p.structure == "rhscancel_normal") {
        std::vector<float> x(static_cast<size_t>(p.m), 0.0f);
        x[0] = x[1] = std::ldexp(1.0f, 89);
        x[2] = x[3] = 1.0f;
        x.back() = std::ldexp(1.0f, 127);
        return x;
    }
    if (p.structure == "rhscancelf") {
        // alpha is 2^127, so rows 0 and 2 get alpha * x = 2^254 and cannot
        // publish a finite answer without the scaled path, while row 1 gets
        // 2^27 and solves to the 2^-100 the emptied term needs; see
        // MakeRhsCancelFloatCsr
        return {1.7014118e38f, 7.888609e-31f, 1.7014118e38f};
    }
    if (p.structure == "multilevel_inplace") {
        std::vector<float> x(static_cast<size_t>(p.m), 1.0f);
        for (int64_t row = 64; row < p.m; ++row) {
            x[static_cast<size_t>(row)] = 2.0f;
        }
        return x;
    }
    if (p.case_name.find("x_zero") != std::string::npos) {
        return std::vector<float>(p.m, 0.0f);
    }
    return makeDenseFloat(p.m, -5.0, 5.0, p.seed + 1);
}

static std::vector<float> ComputeGolden(const CsrMatrix& csr,
                                         const std::vector<float>& xVec,
                                         const SpSVParam& p,
                                         bool lower, bool unitDiag,
                                         bool transpose) {
    if (p.structure == "missing_diag" || p.structure == "empty_non_unit") {
        // Use manual forward/backward substitution for structures where the exact
        // IEEE-754 division-by-zero semantics must be preserved (diagVal=0 → Inf/NaN).
        return SpSVGoldenManual(csr, xVec, p.alpha, lower, unitDiag, transpose);
    }
    return SpSVGolden(csr, xVec, p.alpha, lower, unitDiag, transpose);
}

// A diagonal whose square leaves fp32 range is paired with x = diag and
// alpha = 1, so the exact answer is 1 in every row however large or small the
// diagonal is -- only the intermediate is out of range.
// The cases above whose matrix entries are real, so that the quotient is
// exact in both components and the imaginary part has an exact expectation.
static bool IsRealCplxRange(const SpSVParam& p) {
    return p.structure == "minnormal" || p.structure == "blocks33" ||
           p.structure == "blocks32c" || p.structure == "prodovf" ||
           p.structure == "cancelminnorm" || p.structure == "crossterm" ||
           p.structure == "scalemant" || p.structure == "rowscale4" ||
           p.structure == "rowspan4" || p.structure == "keptterm" ||
           p.structure == "rhscancel";
}

// Those of them whose answer is exactly representable, so a tolerance is not
// enough: at the smallest normal any relative tolerance that admits the
// answer also admits zero, which is what the defect produces.
static bool IsExactRange(const SpSVParam& p) {
    return p.structure == "minnormal" || p.structure == "cancelminnorm" ||
           p.structure == "crossterm" || p.structure == "rowscale4" ||
           p.structure == "rowspan4" || p.structure == "keptterm" ||
           p.structure == "rhscancel";
}

static bool IsExtremeDiag(const SpSVParam& p) {
    return p.structure == "huge_diag" || p.structure == "tiny_diag" ||
           p.structure == "max_diag";
}

// diag = 2+i against rhs = (3e38, -3e38); the exact answer is
// (6e37, -1.8e38).  Both are fixed, because the point of the case is the
// one pair of magnitudes, not a distribution.
static bool IsOverflowRhs(const SpSVParam& p) {
    return p.structure == "overflow_rhs";
}

// diag = 0.5+0.5i against rhs = (3e38, 0); the exact answer is
// (3e38, -3e38), both components representable.
static bool IsSmallScale(const SpSVParam& p) {
    return p.structure == "small_scale";
}

// diag = 4, x = 2, alpha from the case (2e38): alpha*x is 4e38 and out of
// range, while the answer 1e38 is not.
static bool IsBigAlpha(const SpSVParam& p) {
    return p.structure == "big_alpha";
}

// x = (2^120 ... 2^120, 2^127).  The identity rows give y = 2^120, so the
// last row's dot product is 2^128 and out of range, while its answer
// 2^127 - 2^128 = -2^127 is not.
static bool IsCancelWide(const SpSVParam& p) {
    return p.structure == "cancel_wide";
}

// Same matrix, but x = 2^127 throughout and only a handful of rows.  Three
// nonzeros stay under the warp threshold and are walked by one thread; four
// reach it, which is why _3 passed the butterfly defect and _4 did not.
static bool IsCancelShort(const SpSVParam& p) {
    return p.structure == "cancel_short";
}

// diag = 0.75+0.75i with alpha = 2^127 and x = 2: alpha*x is 2^128 and out
// of range while the quotient is not.
static bool IsBigAlphaSmallD(const SpSVParam& p) {
    return p.structure == "big_alpha_small_d";
}

// diag = 2+i, x = 1.5, alpha = 3e38+3e38i: alpha*x is out of range while the
// answer 2.7e38 + 9e37i is not.
static bool IsBigAlphaCplx(const SpSVParam& p) {
    return p.structure == "big_alpha_cplx";
}

// diag = 1+0.5i, x = 0.65-0.65i, alpha = 3e38+3e38i.  alpha*x leaves range,
// and so does alpha/diag at 3.6e38, so the recovery may divide neither
// operand on its own; the answer 3.12e38 - 1.56e38i is representable.
static bool IsDivFirstCplx(const SpSVParam& p) {
    return p.structure == "divfirst_cplx";
}

// Lower bidiagonal, diag [1, 2+i], x = [1, 1.5], the same alpha.  Row 1's
// right-hand side leaves range and is pre-divided in the precompute, so the
// finish meets a finite dot product of -3e38 in both components; multiplying
// by the reciprocal before dividing by the scale formed -3.6e38 on the way.
// The answer is 9e37 + 3e37i.
static bool IsPreDivDot(const SpSVParam& p) {
    return p.structure == "predivdot_cplx";
}

// The identity with a full last row of m nonzeros, which goes to the warp
// solver with one element per lane.  The entries are +3e38, +3e38, -3e38,
// -3e38 repeating against x = 1, so every term and the answer are finite and
// the dot product is exactly zero -- but lanes paired by the first butterfly
// step hold the same sign, so that step forms 6e38 and the next subtracts two
// infinities.
//
// The last row's x is 1e33 rather than 1 so that the reference can hold it.
// The golden subtracts in double and sequentially, and 1 - 3e38 rounds to
// -3e38, so with x = 1 the reference loses the right-hand side and answers 0
// while the kernel's float-float sum keeps it and answers 1.  At 1e33 the
// right-hand side clears double's ulp at 3e38 by 2^52 and fp32's by 2^5, and
// both sides agree.
static bool IsWarpSum(const SpSVParam& p) {
    return p.structure == "warpsum";
}

// The three range cases the review asked for; see MakeNumericCsr.
static bool IsRangeRow(const SpSVParam& p) {
    return p.structure == "restore5" || p.structure == "shortacc3" ||
           p.structure == "shortrhs3";
}

// The identity diagonals stay at one; the full last row alternates the two
// signs in pairs, and its own diagonal stays at one.
static std::vector<std::complex<float>> MakeWarpSumValues(const CsrMatrix& csr,
                                                           int64_t m) {
    std::vector<std::complex<float>> out(static_cast<size_t>(csr.nnz),
                                         {1.0f, 0.0f});
    for (int64_t j = 0; j + 1 < m; j++) {
        float v = ((j / 2) % 2 == 0) ? 3.0e38f : -3.0e38f;
        out[static_cast<size_t>(m - 1 + j)] = {v, 0.0f};
    }
    return out;
}

// The stored values of the three range-guard regressions, which do not set
// one imaginary part across the whole matrix the way the others do.  Returns
// false for every other structure.
static bool MakeRangeGuardValues(const CsrMatrix& csr, const SpSVParam& p,
                                 std::vector<std::complex<float>>& out) {
    if (IsDivFirstCplx(p)) {
        // A = 1+0.5i.  The diagonal has to be this one and not the 2
        // MakeBidiagCsr leaves: over 2+0.5i, alpha/diag is 1.76e38+1.06e38i
        // and stays in range, so the case would not reach what it is for.
        for (int64_t k = 0; k < csr.nnz; k++) out[k] = {1.0f, 0.5f};
        return true;
    }
    if (IsPreDivDot(p)) {
        // A = [[1, 0], [1, 2+i]].  A unit first diagonal is what makes row
        // one's dot product exactly (-3e38, -3e38).
        for (int64_t k = 0; k < csr.nnz; k++) out[k] = {1.0f, 0.0f};
        for (int64_t i = 1; i < p.m; i++) out[2 * i] = {2.0f, 1.0f};
        return true;
    }
    if (IsWarpSum(p)) {
        out = MakeWarpSumValues(csr, p.m);
        return true;
    }
    if (IsRangeRow(p)) {
        // real only, so that the failure is the accumulation's range and not
        // anything the complex multiply does
        for (int64_t k = 0; k < csr.nnz; k++) out[k] = {csr.values[k], 0.0f};
        return true;
    }
    return false;
}

// Their right-hand sides, the same way.
static bool MakeRangeGuardX(const SpSVParam& p,
                            std::vector<std::complex<float>>& x) {
    if (IsDivFirstCplx(p)) {
        for (int64_t i = 0; i < p.m; i++) x[i] = {0.65f, -0.65f};
        return true;
    }
    if (IsPreDivDot(p)) {
        x[0] = {1.0f, 0.0f};
        for (int64_t i = 1; i < p.m; i++) x[i] = {1.5f, 0.0f};
        return true;
    }
    if (IsWarpSum(p)) {
        for (int64_t i = 0; i + 1 < p.m; i++) x[i] = {1.0f, 0.0f};
        x[p.m - 1] = {1.0e33f, 0.0f};
        return true;
    }
    if (IsRangeRow(p)) {
        // 2^126 for the five-row case and 2^127 for the three-row ones, both
        // exact; the last entry is zero except where the review's case keeps
        // the right-hand side, which is what decides whether the row seeds
        // its accumulator with something or with nothing.
        float big = (p.structure == "restore5") ? 8.5070592e37f
                                                : 1.7014118e38f;
        for (int64_t i = 0; i + 1 < p.m; i++) x[i] = {big, 0.0f};
        x[p.m - 1] = (p.structure == "shortrhs3") ? std::complex<float>{big, 0.0f}
                                                  : std::complex<float>{0.0f, 0.0f};
        return true;
    }
    if (p.structure == "keptterm") {
        // the identity rows solve to [F, F, v] with v = 1.5 at m = 4 and 24
        // at m = 36, so the last row's third product is 24 * v -- which the
        // shift brings to exactly the smallest normal and therefore keeps
        for (int64_t i = 0; i < p.m; i++) x[i] = {0.0f, 0.0f};
        x[0] = {1.70141183e38f, 0.0f};
        x[1] = {1.70141183e38f, 0.0f};
        x[2] = {(p.m <= 4) ? 1.5f : 24.0f, 0.0f};
        return true;
    }
    if (p.structure == "rhscancel") {
        // rows 0..3 solve to [2^89, 2^89, 1, 1]; alpha * x = 2^127 then
        // cancels the 2^127 * 1 the shift kept, which leaves the 2^-70 * 1 it
        // emptied as the whole answer -- so the answer is -1 and not the 0 a
        // row publishes without the emptied product
        for (int64_t i = 0; i < p.m; i++) x[i] = {1.0f, 0.0f};
        x[0] = {6.18970020e26f, 0.0f};          // 2^89
        x[1] = {6.18970020e26f, 0.0f};
        x[p.m - 1] = {1.70141183e38f, 0.0f};    // 2^127
        return true;
    }
    return false;
}

// The complex values travel beside the shared float CsrMatrix.  For the
// ordinary cases the imaginary part is an independent draw, so that T and H
// differ and the float-float substitution is exercised; for the extreme
// diagonals it equals the real part, which is what makes |diag|^2 overflow.
static std::vector<std::complex<float>> MakeComplexValues(const CsrMatrix& csr,
                                                          const SpSVParam& p) {
    std::vector<std::complex<float>> out(static_cast<size_t>(csr.nnz));
    if (MakeRangeGuardValues(csr, p, out)) {
        return out;
    }
    if (IsRealCplxRange(p)) {
        for (int64_t k = 0; k < csr.nnz; k++) out[k] = {csr.values[k], 0.0f};
        return out;
    }
    if (IsExtremeDiag(p)) {
        for (int64_t k = 0; k < csr.nnz; k++) out[k] = {csr.values[k], csr.values[k]};
        return out;
    }
    if (IsOverflowRhs(p)) {
        for (int64_t k = 0; k < csr.nnz; k++) out[k] = {csr.values[k], 1.0f};
        return out;
    }
    if (IsSmallScale(p)) {
        for (int64_t k = 0; k < csr.nnz; k++) out[k] = {csr.values[k], csr.values[k]};
        return out;
    }
    if (IsBigAlpha(p) || IsCancelWide(p) || IsCancelShort(p)) {
        for (int64_t k = 0; k < csr.nnz; k++) out[k] = {csr.values[k], 0.0f};
        return out;
    }
    if (IsBigAlphaSmallD(p)) {
        for (int64_t k = 0; k < csr.nnz; k++) out[k] = {csr.values[k], csr.values[k]};
        return out;
    }
    if (IsBigAlphaCplx(p)) {
        for (int64_t k = 0; k < csr.nnz; k++) out[k] = {csr.values[k], 1.0f};
        return out;
    }
    auto im = makeDenseFloat(csr.nnz, -1.0, 1.0, p.seed + 777);
    for (int64_t k = 0; k < csr.nnz; k++) out[k] = {csr.values[k], im[k]};
    return out;
}

static std::complex<float> MakeRealCplxRangeValue(const SpSVParam& p, int64_t i, float maxFloat)
{
    if (p.structure == "rowspan4") { return {(i < 2) ? maxFloat : ((i == 2) ? 1.0f : 0.0f), 0.0f}; }
    if (p.structure == "rowscale4") { return {(i == 2) ? maxFloat : ((i == 3) ? 0.0f : 1.0f), 0.0f}; }
    if (p.structure == "minnormal") { return {2.0f, 0.0f}; }
    if (p.structure == "prodovf") {
        return i == 0 ? std::complex<float>(1.0e30f, 0.0f) : std::complex<float>(0.0f, 0.0f);
    }
    if (p.structure == "crossterm") { return {2.0f, maxFloat}; }
    if (p.structure == "scalemant" || p.structure == "cancelminnorm") {
        float last = p.structure == "cancelminnorm" ? 2.0f : 0.0f;
        return (i + 1 < p.m) ? std::complex<float>(maxFloat, 0.0f) : std::complex<float>(last, 0.0f);
    }
    if (p.structure == "blocks32c" && i + 3 >= p.m) {
        return (i + 1 < p.m) ? std::complex<float>(maxFloat, 0.0f) : std::complex<float>(0.0f, 0.0f);
    }
    return (i % 2 == 0) ? std::complex<float>(maxFloat, 0.0f) : std::complex<float>(0.0f, 0.0f);
}

static bool MakeComplexSpecialX(const CsrMatrix& csr,
    const std::vector<std::complex<float>>& values, const SpSVParam& p,
    std::vector<std::complex<float>>& x);

static std::vector<std::complex<float>> MakeComplexX(const CsrMatrix& csr,
                                                     const std::vector<std::complex<float>>& values,
                                                     const SpSVParam& p) {
    std::vector<std::complex<float>> x(static_cast<size_t>(p.m));
    if (MakeRangeGuardX(p, x)) {
        return x;
    }
    if (IsRealCplxRange(p)) {
        const float kF = 1.70141183e38f;   // 2^127
        for (int64_t i = 0; i < p.m; i++) {
            x[i] = MakeRealCplxRangeValue(p, i, kF);
        }
        return x;
    }
    if (MakeComplexSpecialX(csr, values, p, x)) { return x; }
    auto re = makeDenseFloat(p.m, -5.0, 5.0, p.seed + 1);
    auto im = makeDenseFloat(p.m, -5.0, 5.0, p.seed + 2);
    for (int64_t i = 0; i < p.m; i++) x[i] = {re[i], im[i]};
    return x;
}

static bool MakeComplexSpecialX(const CsrMatrix& csr,
    const std::vector<std::complex<float>>& values, const SpSVParam& p,
    std::vector<std::complex<float>>& x)
{
    if (IsExtremeDiag(p)) {
        // x = diag, so y is exactly one.
        for (int64_t i = 0; i < p.m; i++) x[i] = values[csr.rowOffsets[i]];
        return true;
    }
    if (IsOverflowRhs(p)) {
        for (int64_t i = 0; i < p.m; i++) x[i] = {3.0e38f, -3.0e38f};
        return true;
    }
    if (IsSmallScale(p)) {
        for (int64_t i = 0; i < p.m; i++) x[i] = {3.0e38f, 0.0f};
        return true;
    }
    if (IsBigAlpha(p)) {
        for (int64_t i = 0; i < p.m; i++) x[i] = {2.0f, 0.0f};
        return true;
    }
    if (IsCancelWide(p)) {
        for (int64_t i = 0; i + 1 < p.m; i++) x[i] = {1.32922800e36f, 0.0f};  // 2^120
        x[p.m - 1] = {1.70141183e38f, 0.0f};                                  // 2^127
        return true;
    }
    if (IsCancelShort(p)) {
        for (int64_t i = 0; i < p.m; i++) x[i] = {1.70141183e38f, 0.0f};      // 2^127
        return true;
    }
    if (IsBigAlphaSmallD(p)) {
        for (int64_t i = 0; i < p.m; i++) x[i] = {2.0f, 0.0f};
        return true;
    }
    if (IsBigAlphaCplx(p)) {
        for (int64_t i = 0; i < p.m; i++) x[i] = {1.5f, 0.0f};
        return true;
    }
    return false;
}

// What the matrix holds once updateMatrix has run.  GENERAL replaces every
// stored value, parallel to the value array.  DIAGONAL replaces only the
// diagonal and is indexed by row: newValues[i] is row i's new diagonal,
// whichever slot that row's diagonal happens to occupy.
static std::vector<std::complex<float>> EffectiveValuesAfterUpdate(
    const CsrMatrix& csr,
    const std::vector<std::complex<float>>& original,
    const std::vector<std::complex<float>>& newValues,
    const std::string& updateMode) {
    if (updateMode == "GENERAL") return newValues;
    auto out = original;
    for (int64_t i = 0; i < csr.rows; i++) {
        for (int32_t k = csr.rowOffsets[i]; k < csr.rowOffsets[i + 1]; k++) {
            if (csr.colIndices[k] == static_cast<int32_t>(i)) out[k] = newValues[i];
        }
    }
    return out;
}

static void VerifyComplex(const std::vector<std::complex<float>>& actual,
                          const std::vector<std::complex<double>>& golden,
                          const SpSVParam& p, const std::string& label) {
    size_t n = actual.size();
    std::vector<float> ar(n), ai(n), gr(n), gi(n);
    for (size_t i = 0; i < n; i++) {
        ar[i] = actual[i].real();
        ai[i] = actual[i].imag();
        gr[i] = static_cast<float>(golden[i].real());
        gi[i] = static_cast<float>(golden[i].imag());
    }
    VerifyResult(ar, gr, p, label + "[re]");
    VerifyResult(ai, gi, p, label + "[im]");
}

// nan, +inf, -inf or finite, which is the distinction a relative tolerance
// cannot make and the one the range cases are about.
static const char* ComponentKind(float v) {
    if (std::isnan(v)) {
        return "nan";
    }
    if (std::isinf(v)) {
        return (v < 0.0f) ? "-inf" : "+inf";
    }
    return "finite";
}

// __builtin_memcpy rather than std::memcpy: the checker rejects the latter as
// an unsafe function, and this is the same way SpsvPow2 reads a float's bits.
static uint32_t FloatBits(float v) {
    uint32_t b = 0;
    __builtin_memcpy(&b, &v, sizeof(b));
    return b;
}

// Every component of the first two rows and the last -- the last is the one
// the counter-example is about and the ones before it are what it is formed
// from.  The bits are printed because a decimal does not separate a negative
// zero from a positive one, nor one nan from another.
static void PrintRangeRow(const std::vector<std::complex<float>>& y,
                          const std::vector<std::complex<double>>& golden,
                          const SpSVParam& p, size_t i) {
    float wr = static_cast<float>(golden[i].real());
    float wi = static_cast<float>(golden[i].imag());
    std::cout << "  [" << p.case_name << "] row " << i
              << "  re=" << y[i].real() << " (0x" << std::hex << FloatBits(y[i].real())
              << std::dec << ", want " << ComponentKind(wr) << " " << wr << ")"
              << "  im=" << y[i].imag() << " (0x" << std::hex << FloatBits(y[i].imag())
              << std::dec << ", want " << ComponentKind(wi) << " " << wi << ")\n";
}

// The component-by-component check the review asks for, on top of the
// tolerance VerifyComplex already applies.
static void VerifyRangeComponents(const std::vector<std::complex<float>>& y,
                                  const std::vector<std::complex<double>>& golden,
                                  const SpSVParam& p) {
    for (size_t i = 0; i < y.size(); i++) {
        if (i < 2 || i + 1 >= y.size()) {
            PrintRangeRow(y, golden, p, i);
        }
        const float got[2] = {y[i].real(), y[i].imag()};
        const float want[2] = {static_cast<float>(golden[i].real()),
                               static_cast<float>(golden[i].imag())};
        for (int c = 0; c < 2; c++) {
            std::string at = "[" + p.case_name + "] row " + std::to_string(i) +
                             (c == 0 ? " real" : " imag");
            EXPECT_EQ(std::isnan(got[c]), std::isnan(want[c])) << at << ": nan";
            EXPECT_EQ(std::isinf(got[c]), std::isinf(want[c])) << at << ": inf";
            if (std::isinf(want[c])) {
                EXPECT_EQ(got[c] < 0.0f, want[c] < 0.0f) << at << ": inf sign";
            }
        }
    }
}

// Both components of every row, bit for bit.  Used where the answer is
// exactly representable and a tolerance cannot pin it down.
static void VerifyExactComponents(const std::vector<std::complex<float>>& y,
                                  const std::vector<std::complex<double>>& golden,
                                  const SpSVParam& p) {
    for (size_t i = 0; i < y.size(); i++) {
        uint32_t wantRe = FloatBits(static_cast<float>(golden[i].real()));
        uint32_t wantIm = FloatBits(static_cast<float>(golden[i].imag()));
        EXPECT_EQ(FloatBits(y[i].real()), wantRe)
            << "[" << p.case_name << "] row " << i << " real bits";
        EXPECT_EQ(FloatBits(y[i].imag()), wantIm)
            << "[" << p.case_name << "] row " << i << " imag bits";
    }
    size_t last = y.size() - 1;
    uint32_t refRe = FloatBits(static_cast<float>(golden[last].real()));
    // scientific, because the default format prints the smallest normal as
    // 0.000000 and that reads exactly like the failure this pins down
    std::cout << "  [" << p.case_name << "] row " << last << "  re=0x" << std::hex
              << FloatBits(y[last].real()) << " im=0x" << FloatBits(y[last].imag())
              << " want=0x" << refRe
              << std::dec << " (exact, " << std::scientific
              << static_cast<double>(y[last].real()) << ")" << std::defaultfloat
              << "\n";
}

// The part of the config that does not depend on the value type.
static SpSVNpuConfig MakeConfig(const SpSVParam& p) {
    SpSVNpuConfig cfg;
    cfg.format = p.format;
    cfg.isI64 = p.isI64();
    cfg.isRowPtrI64 = p.isRowPtrI64();
    cfg.isColIndI64 = p.isColIndI64();
    cfg.idxBase = p.index_base;
    cfg.lower = p.isLower();
    cfg.unitDiag = p.isUnitDiag();
    cfg.opA = (p.op_type == "TRANSPOSE") ? ACL_SPARSE_OP_TRANSPOSE :
              (p.op_type == "CONJUGATE_TRANSPOSE") ? ACL_SPARSE_OP_CONJUGATE_TRANSPOSE :
              ACL_SPARSE_OP_NON_TRANSPOSE;
    cfg.inPlace = p.in_place;
    cfg.updateMode = p.update_mode;
    cfg.sliceWidth = p.slice_width;
    return cfg;
}

// The complex64 case end to end: solve, verify, then -- when the case asks
// for it -- updateMatrix and verify the second solve against a golden built
// from the values the wrapper actually sent.
static void RunComplexCase(const SpSVParam& p, const CsrMatrix& csr,
                           aclsparseHandle_t handle, aclrtStream stream) {
    auto values = MakeComplexValues(csr, p);
    auto xC = MakeComplexX(csr, values, p);
    std::complex<float> alphaC(p.alpha, p.alpha_imag);
    auto goldenC = SpSVGoldenComplex(csr, values, xC, alphaC,
                                     p.isLower(), p.isUnitDiag(), p.opCode());
    if (p.structure == "cancelminnorm") {
        // The reference loses this one to rounding rather than to range: it
        // substitutes in double, so (2 - 2^128) + 2^128 is zero because 2 is
        // far below the ulp of 2^128.  The closed form is 2 / 2^127, the
        // smallest normal, and that is what is asserted; see the note above
        // the case in the csv.
        goldenC.back() = std::complex<double>(std::ldexp(1.0, -126), 0.0);
    }

    SpSVNpuResultComplex r;
    try {
        r = SpSVNpuWrapperComplex(handle, stream, csr, values, xC, alphaC,
                                  MakeConfig(p));
    } catch (const std::exception& e) {
        FAIL() << "[" << p.case_name << "] NPU execution threw: " << e.what();
    }
    ASSERT_EQ(r.y.size(), goldenC.size()) << "[" << p.case_name << "] Size mismatch";
    VerifyComplex(r.y, goldenC, p, p.case_name);
    if (IsRangeRow(p) || IsCancelWide(p) || IsCancelShort(p) || IsRealCplxRange(p)) {
        VerifyRangeComponents(r.y, goldenC, p);
    }
    if (IsExactRange(p)) {
        VerifyExactComponents(r.y, goldenC, p);
    }
    if (!r.hasSecondSolve) return;

    auto after = EffectiveValuesAfterUpdate(csr, values, r.newValues, p.update_mode);
    auto golden2 = SpSVGoldenComplex(csr, after, xC, alphaC,
                                     p.isLower(), p.isUnitDiag(), p.opCode());
    VerifyComplex(r.y2, golden2, p, p.case_name + "_update");
}

// Exercise the rejection contract directly at all three stages, including
// solve before a successful analysis, so late analysis-only checks cannot pass.
static void VerifyUnsupportedComplex(const SpSVParam& p, const CsrMatrix& csr,
                                      aclsparseHandle_t handle) {
    std::vector<std::complex<float>> values(csr.nnz, {1.0f, 0.0f});
    SpSVNpuConfig cfg = MakeConfig(p);
    cfg.valueType = ACL_COMPLEX64;
    cfg.complexValues = &values;
    auto setup = SetupMatrixDescriptor(csr, cfg);
    std::vector<std::complex<float>> x(p.m, {1.0f, 0.0f});
    auto dX = DeviceBuffer::copyFrom(x.data(), x.size() * sizeof(x[0]));
    auto dY = DeviceBuffer::alloc(x.size() * sizeof(x[0]));
    auto vx = DnVecManager::createConst(p.m, dX.get(), ACL_COMPLEX64);
    auto vy = DnVecManager::create(p.m, dY.get(), ACL_COMPLEX64);
    SpSVDescrManager descr;
    std::complex<float> alpha{1.0f, 0.0f};
    size_t bytes = 0;
    auto ws = DeviceBuffer::alloc(512);
    RAIIEnvGuard guard;
    if (p.force_permtype64) {
        guard.set("SPSV_FORCE_PERMT_64", "1");
    }
    EXPECT_EQ(aclsparseSpSV_bufferSize(handle, cfg.opA, &alpha, setup.matA.cget(),
        vx.cget(), vy.get(), ACL_COMPLEX64, ACL_SPARSE_SPSV_ALG_DEFAULT, descr.get(), &bytes),
        ACL_SPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(aclsparseSpSV_analysis(handle, cfg.opA, &alpha, setup.matA.cget(),
        vx.cget(), vy.get(), ACL_COMPLEX64, ACL_SPARSE_SPSV_ALG_DEFAULT, descr.get(), ws.get()),
        ACL_SPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(aclsparseSpSV_solve(handle, cfg.opA, &alpha, setup.matA.cget(),
        vx.cget(), vy.get(), ACL_COMPLEX64, ACL_SPARSE_SPSV_ALG_DEFAULT, descr.get()),
        ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

#include "spsv_pipeline_and_errors_tests.h"
#include "spsv_rounding_and_nonfinite_tests.h"
#include "spsv_dependency_and_aliasing_tests.h"
#include "spsv_descriptor_reuse_tests.h"
