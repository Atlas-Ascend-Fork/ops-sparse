#pragma once

__aicore__ inline bool SpmmFinite(float value)
{
    return (value == value) && ((value - value) == 0.0f);
}

__aicore__ inline float SpmmScaleProduct(float value, int exponent)
{
    constexpr float up = 4294967296.0f;
    constexpr float down = 0.00000000023283064365386962890625f;
    while (exponent > 0) {
        value *= up;
        exponent -= 32;
    }
    while (exponent < 0) {
        value *= down;
        exponent += 32;
    }
    return value;
}

__aicore__ inline float SpmmNormalizeProduct(float value, int &exponent)
{
    constexpr float up = 4294967296.0f;
    constexpr float down = 0.00000000023283064365386962890625f;
    float magnitude = value < 0.0f ? -value : value;
    while (magnitude > up) {
        value *= down;
        magnitude *= down;
        exponent += 32;
    }
    while (magnitude != 0.0f && magnitude < down) {
        value *= up;
        magnitude *= up;
        exponent -= 32;
    }
    return value;
}

__aicore__ inline float SpmmProductResidual(float left, float right, float product)
{
    // Power-of-two normalization keeps the Dekker splitter and cross products finite.
    int exponent = 0;
    left = SpmmNormalizeProduct(left, exponent);
    right = SpmmNormalizeProduct(right, exponent);
    const float normalizedProduct = SpmmScaleProduct(product, -exponent);
    const volatile float leftSplit = 4097.0f * left;
    const float leftHigh = leftSplit - (leftSplit - left);
    const float leftLow = left - leftHigh;
    const volatile float rightSplit = 4097.0f * right;
    const float rightHigh = rightSplit - (rightSplit - right);
    const float rightLow = right - rightHigh;
    const float residual = ((leftHigh * rightHigh - normalizedProduct) +
        leftHigh * rightLow + leftLow * rightHigh) + leftLow * rightLow;
    return SpmmScaleProduct(residual, exponent);
}
