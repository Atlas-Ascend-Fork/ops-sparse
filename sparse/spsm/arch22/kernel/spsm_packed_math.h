/**
 * Copyright (c) 2026 liutaw.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSM_ARCH22_PACKED_MATH_H
#define SPSM_ARCH22_PACKED_MATH_H
#include <cstdint>
// "Packed" is an internal extended-range number representation, not a sparse
// input format or a separate operator. Each real/imaginary component has its own
// signed 16-bit exponent and signed 48-bit compensated mantissa (two 32-bit words).
// Computation uses two FP32 components plus the exponent. Keep normalization,
// operation order and explicit subnormal flushing: algebraic rewrites can change
// rounding. TimesNormalized accepts only normalized stored values/FromFloat,
// never a running accumulation or a division numerator. Re-normalize at encoding;
// an encoding/range failure requests the original-path fallback.
// Conversion rounds to FP32 with integer round-to-nearest-even, including
// representable subnormals. Out-of-FP32-range results may still be signed INF;
// this finite-input path is neither an unlimited-range nor an exact solver.
// Compensated finite arithmetic with an independent exponent per component.
// Preserve operation order and explicit subnormal flushing when changing this code.
#ifndef SPSM_PACKED_INLINE
#define SPSM_PACKED_INLINE inline
#endif
namespace spsm_packed { union Bits {uint32_t u;float f;}; }
namespace spsm_packed {
struct CompensatedPair {float high,low;};
SPSM_PACKED_INLINE float FlushSubnormal(float a) {
    Bits b;b.f=a;if((b.u&0x7f800000u)==0)b.u &= 0x80000000u;return b.f;
}
// Internal operands are already FTZ: Normalize/Sum sanitize external inputs,
// Scale preserves this invariant, and every arithmetic result is flushed.
SPSM_PACKED_INLINE float AddScalar(float a,float b) {return FlushSubnormal(a+b);}
SPSM_PACKED_INLINE float SubtractScalar(float a,float b) {return FlushSubnormal(a-b);}
SPSM_PACKED_INLINE float MultiplyScalar(float a,float b) {return FlushSubnormal(a*b);}
SPSM_PACKED_INLINE CompensatedPair TwoSum(float a,float b) {
    float h=AddScalar(a,b),v=SubtractScalar(h,a);
    return {h,AddScalar(SubtractScalar(b,v),SubtractScalar(a,SubtractScalar(h,v)))};
}
SPSM_PACKED_INLINE CompensatedPair AddPair(CompensatedPair a,CompensatedPair b) {
    CompensatedPair s=TwoSum(a.high,b.high);
    return TwoSum(s.high,AddScalar(AddScalar(a.low,b.low),s.low));
}
SPSM_PACKED_INLINE float SplitHigh(float a) {Bits b;b.f=a;b.u &= 0xfffff000u;return b.f;}
SPSM_PACKED_INLINE CompensatedPair ProductPair(CompensatedPair a,float b) {
    float ah=SplitHigh(a.high),al=SubtractScalar(a.high,ah),bh=SplitHigh(b),bl=SubtractScalar(b,bh);
    float h=MultiplyScalar(a.high,b),l=SubtractScalar(MultiplyScalar(ah,bh),h);
    l=AddScalar(l,MultiplyScalar(ah,bl));l=AddScalar(l,MultiplyScalar(al,bh));l=AddScalar(l,MultiplyScalar(al,bl));
    l=AddScalar(l,MultiplyScalar(a.low,b));return {h,l};
}

struct ScaledValue {float high,low;int32_t exponent;};
SPSM_PACKED_INLINE bool IsFinite(float value) {
    Bits bits;bits.f=value;return (bits.u&0x7f800000u)!=0x7f800000u;
}
// Exact power-of-two shift when normal; explicitly FTZ otherwise.
SPSM_PACKED_INLINE float ScalePowerOfTwo(float value,int32_t shift) {
    Bits bits;bits.f=value;
    int32_t old=static_cast<int32_t>((bits.u>>23)&255u);
    if(old==0){bits.u&=0x80000000u;return bits.f;}
    int32_t exponent=old+shift;
    if(exponent<=0)bits.u&=0x80000000u;
    else if(exponent>=255)bits.u=(bits.u&0x80000000u)|0x7f800000u;
    else bits.u=(bits.u&0x807fffffu)|(static_cast<uint32_t>(exponent)<<23);
    return bits.f;
}
SPSM_PACKED_INLINE ScaledValue Normalize(float high,float low,int32_t exponent) {
    // External pair ingestion is here. Internal TwoSum callers pass operands
    // already sanitized by ScalePowerOfTwo or arithmetic result flushing.
    high=FlushSubnormal(high);low=FlushSubnormal(low);
    CompensatedPair sum=TwoSum(high,low);
    if(!IsFinite(sum.high))return {sum.high,0,0};
    if(sum.high==0)return {0,0,0};
    Bits bits;bits.f=sum.high;
    int32_t shift=static_cast<int32_t>((bits.u>>23)&255u)-126;
    bits.u=(bits.u&0x807fffffu)|(126u<<23);
    return {bits.f,ScalePowerOfTwo(sum.low,-shift),exponent+shift};
}
SPSM_PACKED_INLINE ScaledValue FromFloat(float value) {
    Bits bits;bits.f=value;int32_t e=static_cast<int32_t>((bits.u>>23)&255u);
    if(e==0)return {0,0,0};
    if(e==255)return {value,0,0};
    bits.u=(bits.u&0x807fffffu)|(126u<<23);
    return {bits.f,0,e-126};
}
SPSM_PACKED_INLINE ScaledValue AddScaled(ScaledValue a,ScaledValue b) {
    if(a.high==0)return b;
    if(b.high==0)return a;
    if(a.exponent<b.exponent){ScaledValue temp=a;a=b;b=temp;}
    CompensatedPair sum=AddPair({a.high,a.low},
        {ScalePowerOfTwo(b.high,b.exponent-a.exponent),ScalePowerOfTwo(b.low,b.exponent-a.exponent)});
    // Internal additions may retain their exponent when both components stay
    // safely inside binary32's normal range. Packed stores still normalize.
    // This normalization policy is part of the numerical recipe under FTZ.
    Bits magnitude;magnitude.f=sum.high;uint32_t abs=magnitude.u&0x7fffffffu;
    if(abs>=(101u<<23)&&abs<(153u<<23))return {sum.high,sum.low,a.exponent};
    return Normalize(sum.high,sum.low,a.exponent);
}
SPSM_PACKED_INLINE ScaledValue MultiplyScaled(ScaledValue a,float coefficient) {
    coefficient=FlushSubnormal(coefficient);
    if(a.high==0||coefficient==0)return {0,0,0};
    ScaledValue b=FromFloat(coefficient);
    CompensatedPair product=ProductPair({a.high,a.low},b.high);
    // Keep a bounded, uncompacted product until an operation needs its exponent.
    // Preserve compensation and multiplication order; FTZ prevents assuming
    // equivalence with an eagerly normalized product.
    Bits magnitude;magnitude.f=product.high;uint32_t abs=magnitude.u&0x7fffffffu;
    if(abs>=(101u<<23)&&abs<(153u<<23))
        return {product.high,product.low,a.exponent+b.exponent};
    return Normalize(product.high,product.low,a.exponent+b.exponent);
}
}

// Complex operations and packed storage for the extended-range solver.
// The host build is a model of the same arithmetic, not an independent oracle.
namespace spsm_packed {
struct ScaledComplex { ScaledValue real,imag; };
SPSM_PACKED_INLINE float DivideScalar(float a,float b) {return FlushSubnormal(FlushSubnormal(a)/FlushSubnormal(b));}
SPSM_PACKED_INLINE float Absolute(float a) {return a<0?-a:a;}
SPSM_PACKED_INLINE ScaledValue Negate(ScaledValue a) {return {-a.high,-a.low,a.exponent};}
SPSM_PACKED_INLINE ScaledComplex Times(ScaledComplex a,float re,float im) {
    return {AddScaled(MultiplyScaled(a.real,re),MultiplyScaled(a.imag,-im)),
            AddScaled(MultiplyScaled(a.real,im),MultiplyScaled(a.imag,re))};
}
// For canonical packed inputs (or FromFloat), each nonzero high mantissa
// is in [0.5,1) in magnitude. Its product is in [0.25,1), so the bounded-product
// path is certain; avoid four redundant normalize/range decisions per entry.
// Do not use this for a running accumulation or Divide's numerator.
SPSM_PACKED_INLINE ScaledValue NormalizedProduct(ScaledValue a,ScaledValue b) {
    if(a.high==0||b.high==0)return {0,0,0};
    CompensatedPair p=ProductPair({a.high,a.low},b.high);
    return {p.high,p.low,a.exponent+b.exponent};
}
SPSM_PACKED_INLINE ScaledComplex TimesNormalized(ScaledComplex a,float re,float im) {
    ScaledValue cr=FromFloat(re),ci=FromFloat(im);
    return {AddScaled(NormalizedProduct(a.real,cr),NormalizedProduct(a.imag,Negate(ci))),
            AddScaled(NormalizedProduct(a.real,ci),NormalizedProduct(a.imag,cr))};
}
// Both operands are normalized independently; q and its residual stay near 1.
SPSM_PACKED_INLINE ScaledValue DivideReal(ScaledValue a,ScaledValue b) {
    if(a.high==0)return {0,0,0};
    float q=DivideScalar(a.high,b.high);
    CompensatedPair p=ProductPair({b.high,b.low},q);
    CompensatedPair r=AddPair({a.high,a.low},{-p.high,-p.low});
    float correction=DivideScalar(AddScalar(r.high,r.low),b.high);
    CompensatedPair result=TwoSum(q,correction);
    return Normalize(result.high,result.low,a.exponent-b.exponent);
}
// Unlike direct FP32 c*c+d*d, the denominator cannot overflow its mantissa.
[[gnu::noinline]] SPSM_PACKED_INLINE ScaledComplex Divide(ScaledComplex a,float re,float im) {
    ScaledValue cr=FromFloat(re),ci=FromFloat(im);
    ScaledValue denominator=AddScaled(MultiplyScaled(cr,re),MultiplyScaled(ci,im));
    ScaledComplex numerator=Times(a,re,-im);
    return {DivideReal(numerator.real,denominator),DivideReal(numerator.imag,denominator)};
}
SPSM_PACKED_INLINE int32_t RoundEven(float v) {
    int32_t q=static_cast<int32_t>(v);float f=v-static_cast<float>(q);
    if(f>.5f||(f==.5f&&(q&1)))++q;
    if(f<-.5f||(f==-.5f&&(q&1)))--q;
    return q;
}
constexpr uint64_t MASK48=(uint64_t(1)<<48)-1;
SPSM_PACKED_INLINE uint64_t Encode(ScaledValue a,uint32_t &status) {
    a=Normalize(a.high,a.low,a.exponent);
    if(!IsFinite(a.high)||!IsFinite(a.low)){status=2;return 0;}
    if(a.high==0&&a.low==0)return 0;
    if(Absolute(a.high)<.5f||Absolute(a.high)>=1.f||Absolute(a.low)>0x1p-24f){status=3;return 0;}
    int64_t main=static_cast<int32_t>(a.high*0x1p24f);
    int64_t q=main*(int64_t(1)<<23)+RoundEven(a.low*0x1p47f);
    if(q>=(int64_t(1)<<47)||q<=-(int64_t(1)<<47)){q/=2;++a.exponent;}
    if(a.exponent < -32768||a.exponent > 32767){status=1;return 0;}
    return (uint64_t(static_cast<uint16_t>(a.exponent))<<48)|(uint64_t(q)&MASK48);
}
SPSM_PACKED_INLINE ScaledValue Decode(uint64_t word) {
    uint64_t raw=word&MASK48;
    int64_t q=(raw&(uint64_t(1)<<47))?-static_cast<int64_t>((~raw&MASK48)+1):static_cast<int64_t>(raw);
    int32_t e=static_cast<int32_t>(word>>48);if(e>=32768)e-=65536;
    int32_t high=static_cast<int32_t>(q/(int64_t(1)<<23));
    int32_t low=static_cast<int32_t>(q%(int64_t(1)<<23));
    // Canonical mantissas have 47 magnitude bits. Round to 24 high bits in
    // integer arithmetic, retaining the exact signed residual as the low part.
    // This avoids a floating TwoSum and scaling at every predecessor load.
    if(q >= (int64_t(1)<<46) || q <= -(int64_t(1)<<46)) {
        int32_t magnitude=low<0?-low:low;
        if(magnitude>(1<<22)||(magnitude==(1<<22)&&(high&1))) {
            int32_t delta=low<0?-1:1;high+=delta;low-=delta*(1<<23);
        }
        if(high==(1<<24)||high==-(1<<24))
            return {static_cast<float>(high)*0x1p-25f,static_cast<float>(low)*0x1p-48f,e+1};
        return {static_cast<float>(high)*0x1p-24f,static_cast<float>(low)*0x1p-47f,e};
    }
    return Normalize(static_cast<float>(high)*0x1p-24f,static_cast<float>(low)*0x1p-47f,e);
}
}

// Exact integer round-to-nearest-even conversion of the finite packed format.
// This handles output subnormals explicitly; it does not rely on device FTZ.
#ifndef SPSM_PACKED_INLINE
#define SPSM_PACKED_INLINE inline
#endif
namespace spsm_packed {
SPSM_PACKED_INLINE uint64_t RoundShiftEven(uint64_t value,int shift){
    if(shift>=64)return 0;
    uint64_t q=value>>shift,mask=(uint64_t(1)<<shift)-1;
    uint64_t tail=value&mask,half=uint64_t(1)<<(shift-1);
    return q+(tail>half||(tail==half&&(q&1)));
}
SPSM_PACKED_INLINE uint32_t PackedFloatBits(uint64_t word){
    const uint64_t mask=(uint64_t(1)<<48)-1;
    uint64_t raw=word&mask;bool negative=(raw&(uint64_t(1)<<47))!=0;
    uint64_t magnitude=negative?((~raw&mask)+1):raw;
    if(magnitude==0)return 0;
    uint32_t sign=negative?0x80000000u:0;
    int exponent=int(word>>48);if(exponent>=32768)exponent-=65536;
    if(exponent>=129)return sign|0x7f800000u;
    if(exponent < -125){
        int shift=-exponent-102;
        return sign|uint32_t(RoundShiftEven(magnitude,shift));
    }
    uint64_t mantissa=RoundShiftEven(magnitude,23);
    if(mantissa==(uint64_t(1)<<24)){mantissa>>=1;++exponent;}
    if(exponent>=129)return sign|0x7f800000u;
    return sign|(uint32_t(exponent+126)<<23)|(uint32_t(mantissa)&0x7fffffu);
}
}
#endif
