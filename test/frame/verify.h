/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_FRAME_VERIFY_H_
#define TEST_FRAME_VERIFY_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "types.h"
#include "mixed_anchor.h"

namespace sparse_test {

class VerifyStrategy {
public:
    virtual ~VerifyStrategy() = default;

    bool verify(const float* output, const float* golden, size_t count, int64_t stride,
                const std::string& caseId) {
        printHead(output, count, stride, "Output", caseId);
        printHead(golden, count, stride, "Golden", caseId);

        size_t skippedCount = 0;
        for (size_t i = 0; i < count; i++) {
            float outVal = output[static_cast<int64_t>(i) * stride];
            float goldVal = golden[static_cast<int64_t>(i) * stride];
            if (shouldSkip(outVal, goldVal)) {
                skippedCount++;
                continue;
            }
            processElement(outVal, goldVal);
        }

        return reportResult(count, skippedCount, caseId);
    }

protected:
    virtual bool shouldSkip(float outVal, float goldVal) {
        if (outVal == goldVal) return true;
        if (std::isnan(outVal) && std::isnan(goldVal)) return true;
        return false;
    }

    virtual void processElement(float outVal, float goldVal) = 0;
    virtual bool reportResult(size_t count, size_t skippedCount, const std::string& caseId) = 0;

    static void printHead(const float* data, size_t count, int64_t stride,
                          const std::string& label, const std::string& caseId) {
        std::cout << std::fixed << std::setprecision(6);
        constexpr size_t kMaxPrint = 10;
        std::cout << "[" << caseId << "] " << label << ": ";
        for (size_t i = 0; i < count && i < kMaxPrint; i++) {
            std::cout << data[static_cast<int64_t>(i) * stride] << " ";
        }
        if (count > kMaxPrint) std::cout << "...";
        std::cout << std::endl;
    }
};

class AbsStrategy : public VerifyStrategy {
public:
    explicit AbsStrategy(double absTol) : absTol_(absTol) {}
protected:
    void processElement(float outVal, float goldVal) override {
        // Equality (including equal infinities) was handled by shouldSkip().
        // A remaining non-finite value is necessarily a mismatch; comparing it
        // with a tolerance would otherwise yield false for NaN and hide it.
        if (!std::isfinite(outVal) || !std::isfinite(goldVal)) {
            failCount_++;
            return;
        }
        if (std::abs(outVal - goldVal) > absTol_) failCount_++;
    }
    bool reportResult(size_t count, size_t /*skippedCount*/, const std::string& caseId) override {
        bool pass = (failCount_ == 0);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (absTol=" << absTol_ << ", " << failCount_ << "/" << count << " failures)" << std::endl;
        return pass;
    }
private:
    double absTol_;
    size_t failCount_ = 0;
};

class RelStrategy : public VerifyStrategy {
public:
    RelStrategy(double relTol, double eps) : relTol_(relTol), eps_(eps) {}
protected:
    void processElement(float outVal, float goldVal) override {
        if (!std::isfinite(outVal) || !std::isfinite(goldVal)) {
            hasNonFiniteMismatch_ = true;
            return;
        }
        double relErr = std::abs(outVal - goldVal) / (std::abs(goldVal) + eps_);
        if (relErr > maxRelErr_) maxRelErr_ = relErr;
    }
    bool reportResult(size_t /*count*/, size_t /*skippedCount*/, const std::string& caseId) override {
        bool pass = !hasNonFiniteMismatch_ && (maxRelErr_ < relTol_);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (maxRelErr=" << maxRelErr_ << ", relTol=" << relTol_ << ")" << std::endl;
        return pass;
    }
private:
    double relTol_;
    double eps_;
    double maxRelErr_ = 0.0;
    bool hasNonFiniteMismatch_ = false;
};

class MereMareStrategy : public VerifyStrategy {
public:
    MereMareStrategy(double threshold, double multiplier)
        : threshold_(threshold), multiplier_(multiplier), outlierLimit_(multiplier * threshold) {}
protected:
    bool shouldSkip(float outVal, float goldVal) override {
        if (VerifyStrategy::shouldSkip(outVal, goldVal)) return true;
        bool outAbnormal = std::isnan(outVal) || std::isinf(outVal);
        bool goldAbnormal = std::isnan(goldVal) || std::isinf(goldVal);
        if (outAbnormal && goldAbnormal) return true;
        return false;
    }
    void processElement(float outVal, float goldVal) override {
        bool outAbnormal = std::isnan(outVal) || std::isinf(outVal);
        bool goldAbnormal = std::isnan(goldVal) || std::isinf(goldVal);
        if (outAbnormal || goldAbnormal) {
            mismatchCount_++;
            return;
        }
        double relErr = std::abs(outVal - goldVal) / (std::abs(goldVal) + kEpsilon);
        sumRelErr_ += relErr;
        if (relErr > maxRelErr_) maxRelErr_ = relErr;
        if (relErr > outlierLimit_) outlierCount_++;
    }
    bool reportResult(size_t count, size_t skippedCount, const std::string& caseId) override {
        size_t effectiveCount = count - skippedCount - mismatchCount_;
        double mere = (effectiveCount > 0) ? sumRelErr_ / static_cast<double>(effectiveCount) : 0.0;

        std::cout << "[" << caseId << "] MERE=" << mere << " MARE=" << maxRelErr_
                  << " (threshold=" << threshold_ << ", outlier_limit=" << outlierLimit_;
        if (skippedCount > 0) std::cout << ", skipped " << skippedCount << " elements";
        if (mismatchCount_ > 0) std::cout << ", mismatch " << mismatchCount_ << " elements";
        std::cout << ")" << std::endl;

        bool pass = (mismatchCount_ == 0) && (mere < threshold_) && (maxRelErr_ < outlierLimit_);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (MERE < threshold && MARE < " << multiplier_ << "*threshold, "
                  << outlierCount_ << " outliers out of " << count << " elements)" << std::endl;
        return pass;
    }
private:
    static constexpr double kEpsilon = 0.00006103515625;
    double threshold_;
    double multiplier_;
    double outlierLimit_;
    double sumRelErr_ = 0.0;
    double maxRelErr_ = 0.0;
    size_t outlierCount_ = 0;
    size_t mismatchCount_ = 0;
};

class ExactStrategy : public VerifyStrategy {
protected:
    void processElement(float /*outVal*/, float /*goldVal*/) override { failCount_++; }
    bool reportResult(size_t count, size_t /*skippedCount*/, const std::string& caseId) override {
        bool pass = (failCount_ == 0);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (exact match, " << failCount_ << "/" << count << " mismatches)" << std::endl;
        return pass;
    }
private:
    size_t failCount_ = 0;
};

class IntegerStrategy : public VerifyStrategy {
protected:
    bool shouldSkip(float /*outVal*/, float /*goldVal*/) override { return false; }
    void processElement(float outVal, float goldVal) override {
        if (static_cast<int64_t>(outVal) != static_cast<int64_t>(goldVal)) failCount_++;
    }
    bool reportResult(size_t count, size_t /*skippedCount*/, const std::string& caseId) override {
        bool pass = (failCount_ == 0);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (integer match, " << failCount_ << "/" << count << " mismatches)" << std::endl;
        return pass;
    }
private:
    size_t failCount_ = 0;
};

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers: MIXED_TOLERANCE thresholds (mixed_tolerance_standard.md, 2026-09-17 修订版)
//   per-element max_abs_error_limit = max(fixed_value, 32 * ULP)
//   ULP(x) = 2^(floor(log2|x|) - mantissaBits)  for finite x != 0
//   ULP(0) = 2^(emin - mantissaBits)              (subnormal ULP)
// ═══════════════════════════════════════════════════════════════════════════════
inline double getUlpsAt(double magnitude, int mantissaBits, int emin)
{
    double absMag = std::abs(magnitude);
    if (absMag == 0.0)
        return std::pow(2.0, static_cast<double>(emin - mantissaBits));
    int exp = 0;
    std::frexp(absMag, &exp);
    return std::ldexp(1.0, std::max(exp - 1, emin) - mantissaBits);
}

struct MixedToleranceDefaults {
    double rtol;
    double atol;
    double maxAbsErrorLimitFixed;
    int mantissaBits;
    int emin;
};

inline MixedToleranceDefaults getMixedToleranceDefaults(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT16:       return {0.0009765625, 0.0009765625, 1e-1, 10, -14};
        case ACL_BF16:          return {0.0078125, 0.0078125, 1e-1, 7, -126};
        case ACL_FLOAT:         return {0.0001220703125, 0.0001220703125, 1e-2, 23, -126};
        case ACL_FLOAT8_E4M3FN: return {0.125, 0.125, 1.0, 3, -6};
        case ACL_FLOAT8_E5M2:   return {0.25, 0.25, 1.0, 2, -14};
        default:
            // 未列举 dtype（如 ACL_DOUBLE/ACL_HIFLOAT8/ACL_FLOAT8_E8M0 等宽或新格式）无标准行，
            // 不按 FP16 的 ULP 数学低估误差：保守回落到 FLOAT32 行口径，并显式告警避免静默放宽。
            std::cout << "[WARN] applyMixedTolerance: unsupported aclDataType " << static_cast<int>(dtype)
                      << ", falling back to FLOAT32 thresholds" << std::endl;
            return {0.0001220703125, 0.0001220703125, 1e-2, 23, -126};
    }
}

inline void applyMixedToleranceInternal(VerifyConfig& cfg, const MixedToleranceDefaults& d)
{
    cfg.mode = PrecisionMode::MIXED_TOLERANCE;
    cfg.mixedRtol = d.rtol;
    cfg.mixedAtol = d.atol;
    cfg.mixedRequiredMatchedRatio = 0.99;
    cfg.mixedMantissaBits = d.mantissaBits;
    cfg.mixedEmin = d.emin;
}

inline void applyMixedTolerance(VerifyConfig& cfg, aclDataType dtype,
                                const float* goldenData, size_t count)
{
    auto d = getMixedToleranceDefaults(dtype);
    applyMixedToleranceInternal(cfg, d);
    cfg.mixedMaxAbsErrorLimit = d.maxAbsErrorLimitFixed;
    (void)goldenData;
    (void)count;
}

inline void applyMixedTolerance(VerifyConfig& cfg, aclDataType dtype, float goldenScalar)
{
    auto d = getMixedToleranceDefaults(dtype);
    applyMixedToleranceInternal(cfg, d);
    cfg.mixedMaxAbsErrorLimit = d.maxAbsErrorLimitFixed;
    (void)goldenScalar;
}

class MixedToleranceStrategy : public VerifyStrategy {
public:
    MixedToleranceStrategy(double absTol, double relTol, double requiredMatchedRatio,
                           double fixedLimit, int mantissaBits, int emin)
        : absTol_(absTol), relTol_(relTol), requiredMatchedRatio_(requiredMatchedRatio),
          fixedLimit_(fixedLimit), mantissaBits_(mantissaBits), emin_(emin)
    {}

protected:
    void processElement(float outVal, float goldVal) override
    {
        if (std::isnan(outVal) || std::isnan(goldVal) || std::isinf(outVal) || std::isinf(goldVal)) {
            maxAbsErrorLimitFailed_ = true;
            failCount_++;
            return;
        }
        double diff = std::abs(static_cast<double>(outVal) - static_cast<double>(goldVal));
        if (diff > maxAbsError_)
            maxAbsError_ = diff;
        double goldAbs = std::abs(static_cast<double>(goldVal));
        double tolLimit = absTol_ + relTol_ * goldAbs;
        double ulpLimit = std::max(fixedLimit_, 32.0 * getUlpsAt(goldAbs, mantissaBits_, emin_));
        if (diff > tolLimit)
            failCount_++;
        if (diff > ulpLimit)
            maxAbsErrorLimitFailed_ = true;
    }

    bool reportResult(size_t count, size_t /*skippedCount*/, const std::string& caseId) override
    {
        size_t matchedCount = (count >= failCount_) ? count - failCount_ : 0;
        double matchedRatio = (count > 0) ? static_cast<double>(matchedCount) / static_cast<double>(count) : 1.0;
        bool ratioOk = matchedRatio >= requiredMatchedRatio_;
        bool errorLimitOk = !maxAbsErrorLimitFailed_;
        bool pass = ratioOk && errorLimitOk;

        std::cout << std::scientific << std::setprecision(4);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (matchedRatio=" << matchedRatio << " (req=" << requiredMatchedRatio_ << ")"
                  << ", maxAbsErr=" << maxAbsError_
                  << ", per-element-limit=max(" << fixedLimit_ << ", 32*ULP_at_each_element)"
                  << ", " << failCount_ << "/" << count << " failures"
                  << ", atol=" << absTol_ << ", rtol=" << relTol_ << ")" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        return pass;
    }

private:
    double absTol_;
    double relTol_;
    double requiredMatchedRatio_;
    double fixedLimit_;
    int mantissaBits_;
    int emin_;
    size_t failCount_ = 0;
    double maxAbsError_ = 0.0;
    bool maxAbsErrorLimitFailed_ = false;
};

class Verifier {
public:
    static bool verifyVector(const std::vector<float>& output, const std::vector<double>& golden,
                             const VerifyConfig& cfg, const std::string& caseId,
                             MixedAnchorResult* details = nullptr) {
        const auto result = EvaluateMixedAnchor(output, golden, cfg);
        if (details != nullptr) { *details = result; }
        std::cout << "[" << caseId << "] " << (result.passed ? "PASSED" : "FAILED")
                  << " (matched=" << result.matched << "/" << result.total
                  << ", maxAbsErr=" << result.maxAbsError << ", anchor=" << result.maxErrorIndex
                  << ", limit=" << result.absoluteLimit << ", " << result.reason << ")" << std::endl;
        return result.passed;
    }

    // Preserve integer values while comparing them.  Converting int32 values
    // to float first loses the low bit at and above 2^24.
    template <typename T, typename std::enable_if_t<std::is_integral_v<T>, int> = 0>
    static bool verifyVector(const std::vector<T>& output, const std::vector<T>& golden,
                             const VerifyConfig& /*cfg*/, const std::string& caseId) {
        if (output.size() != golden.size()) {
            std::cout << "[" << caseId << "] FAILED: size mismatch, output=" << output.size()
                      << " golden=" << golden.size() << std::endl;
            return false;
        }
        const bool pass = std::equal(output.begin(), output.end(), golden.begin());
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (integer exact match)" << std::endl;
        return pass;
    }

    static bool verifyVector(const std::vector<float>& output, const std::vector<float>& golden,
                             const VerifyConfig& cfg, const std::string& caseId) {
        if (output.size() != golden.size()) {
            std::cout << "[" << caseId << "] FAILED: size mismatch, output=" << output.size()
                      << " golden=" << golden.size() << std::endl;
            return false;
        }
        return verifyVector(output.data(), golden.data(), output.size(), 1, cfg, caseId);
    }

    static bool verifyVector(const float* output, const float* golden, size_t count, int64_t stride,
                             const VerifyConfig& cfg, const std::string& caseId) {
        if (cfg.mode == PrecisionMode::MIXED_TOLERANCE_ANCHOR) {
            std::cout << "[" << caseId << "] FAILED: anchor mode requires FLOAT64 Golden" << std::endl;
            return false;
        }
        auto strategy = createStrategy(cfg);
        return strategy->verify(output, golden, count, stride, caseId);
    }

    static bool verifyScalar(float output, float golden, const VerifyConfig& cfg, const std::string& caseId) {
        if (cfg.mode == PrecisionMode::MIXED_TOLERANCE_ANCHOR) {
            return verifyVector(std::vector<float>{output}, std::vector<double>{golden}, cfg, caseId);
        }
        std::cout << "[" << caseId << "] Output: " << output << " Golden: " << golden << std::endl;
        if (output == golden) return true;
        if (std::isnan(output) && std::isnan(golden)) return true;
        if (cfg.mode == PrecisionMode::MIXED_TOLERANCE) {
            if (std::isnan(output) || std::isnan(golden) || std::isinf(output) || std::isinf(golden)) {
                std::cout << "[" << caseId << "] FAILED (special value mismatch)" << std::endl;
                return false;
            }
            double diff = std::abs(static_cast<double>(output) - static_cast<double>(golden));
            double goldAbs = std::abs(static_cast<double>(golden));
            double tolLimit = cfg.mixedAtol + cfg.mixedRtol * goldAbs;
            double ulpLimit = std::max(cfg.mixedMaxAbsErrorLimit,
                                       32.0 * getUlpsAt(goldAbs, cfg.mixedMantissaBits, cfg.mixedEmin));
            bool pass = (diff <= tolLimit) && (diff <= ulpLimit);
            std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                      << " (mixed tolerance, diff=" << diff << ", tolLimit=" << tolLimit
                      << ", ulpLimit=" << ulpLimit << ")" << std::endl;
            return pass;
        }
        return std::abs(output - golden) < cfg.absTol;
    }

    template <typename T>
    static std::vector<float> toFloat(const std::vector<T>& v) {
        std::vector<float> out(v.size());
        for (size_t i = 0; i < v.size(); i++) out[i] = static_cast<float>(v[i]);
        return out;
    }

private:
    static std::unique_ptr<VerifyStrategy> createStrategy(const VerifyConfig& cfg) {
        switch (cfg.mode) {
            case PrecisionMode::ABS:
                return std::make_unique<AbsStrategy>(cfg.absTol);
            case PrecisionMode::REL:
                return std::make_unique<RelStrategy>(cfg.relTol, cfg.epsilonForRel);
            case PrecisionMode::MERE_MARE:
                return std::make_unique<MereMareStrategy>(cfg.mereThreshold, cfg.mareMultiplier);
            case PrecisionMode::EXACT:
                return std::make_unique<ExactStrategy>();
            case PrecisionMode::INTEGER:
                return std::make_unique<IntegerStrategy>();
            case PrecisionMode::MIXED_TOLERANCE:
                return std::make_unique<MixedToleranceStrategy>(
                    cfg.mixedAtol, cfg.mixedRtol, cfg.mixedRequiredMatchedRatio,
                    cfg.mixedMaxAbsErrorLimit, cfg.mixedMantissaBits, cfg.mixedEmin);
            case PrecisionMode::COMBINED:
            default:
                return std::make_unique<MereMareStrategy>(cfg.mereThreshold, cfg.mareMultiplier);
        }
    }
};

}

#endif
