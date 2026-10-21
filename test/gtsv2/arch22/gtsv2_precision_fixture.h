/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Regression fixture: original m=17 seed-8 matrix and first RHS column.
// FP64 expected values are also cross-checked by the Python dense-solve test.
#ifndef GTSV2_ARCH22_PRECISION_FIXTURE_H_
#define GTSV2_ARCH22_PRECISION_FIXTURE_H_
#include <vector>
namespace gtsv2_precision_fixture {
inline const std::vector<float> dl = {
    0.0000000000000000f,
    -1.1892017126083374f,
    1.3932349681854248f,
    2.1058979034423828f,
    -0.35456731915473938f,
    0.37098965048789978f,
    1.0956829786300659f,
    2.1414225101470947f,
    0.13168704509735107f,
    -0.63878297805786133f,
    1.3384251594543457f,
    -1.1907936334609985f,
    -0.76009404659271240f,
    1.1422104835510254f,
    0.93116682767868042f,
    0.026217162609100342f,
    -0.10505484044551849f
};

inline const std::vector<float> d = {
    0.44140410423278809f,
    0.25134131312370300f,
    -0.64283543825149536f,
    0.92220133543014526f,
    -0.12904611229896545f,
    1.6694331169128418f,
    -0.15842108428478241f,
    0.14104604721069336f,
    1.0240715742111206f,
    -1.0194613933563232f,
    -0.56572419404983521f,
    2.2932906150817871f,
    -1.0793939828872681f,
    -0.44866949319839478f,
    0.034906290471553802f,
    -1.0177773237228394f,
    1.6447601318359375f
};

inline const std::vector<float> du = {
    -1.4181894063949585f,
    1.0285246372222900f,
    1.2821575403213501f,
    -1.0550875663757324f,
    1.4990910291671753f,
    1.0125193595886230f,
    -0.075702294707298279f,
    -1.0601079463958740f,
    0.62170988321304321f,
    0.70711022615432739f,
    1.9424937963485718f,
    1.3530771732330322f,
    2.2177300453186035f,
    0.20169235765933990f,
    1.1558145284652710f,
    -0.17115037143230438f,
    0.0000000000000000f
};

inline const std::vector<float> rhs = {
    0.085056044161319733f,
    0.77034646272659302f,
    -1.1658542156219482f,
    0.81477195024490356f,
    -0.29940354824066162f,
    -1.4699387550354004f,
    -0.41314461827278137f,
    -1.3632562160491943f,
    0.097021847963333130f,
    1.1696950197219849f,
    2.8354761600494385f,
    0.38533303141593933f,
    0.62335968017578125f,
    1.2295553684234619f,
    -1.1504057645797729f,
    -0.80245405435562134f,
    0.21555420756340027f
};

inline const std::vector<double> expected = {
    -1931.0444329003305,
    -601.08754891153035,
    -2085.0768069213668,
    -393.14273733491439,
    -4506.1003058528349,
    -481.08465866171167,
    2442.8029711848776,
    -12069.576306496409,
    3329.9160797990703,
    -2928.3249144994238,
    -1212.0469492291511,
    1666.1545789045479,
    -3890.3096000550609,
    -1322.1263084966965,
    19096.332001524774,
    487.43775843555636,
    31.264893373680920
};
}  // namespace gtsv2_precision_fixture

// Independent mixed-row-scale regression, m=12, generated with Python Random(96).
// This is not the reviewer's unavailable original input. Hex literals retain
// the exact FP32 inputs. Expected values were solved from those inputs by dense
// partial-pivot Gaussian elimination using 150 decimal digits, not this solver.
// Subtracting rowExponent from every coefficient and RHS exponent gives the
// ordinary-scale counterpart with exactly the same mathematical solution.
namespace gtsv2_mixed_scale_fixture {
inline const std::vector<float> dl = {
    0x0.0p+0f,
    0x1.b2720c0000000p-90f,
    0x1.9615520000000p-22f,
    -0x1.a6fa560000000p-68f,
    0x1.1adf5a0000000p+32f,
    -0x1.10fc640000000p-17f,
    0x1.e6bc600000000p+26f,
    -0x1.69b07e0000000p+27f,
    0x1.4e39080000000p+49f,
    0x1.f6b1e40000000p+96f,
    -0x1.942d0e0000000p-38f,
    0x1.7d4a840000000p+97f
};

inline const std::vector<float> d = {
    0x1.0f64520000000p+2f,
    0x1.557ad40000000p-88f,
    0x1.3b5d480000000p-19f,
    0x1.6cd99c0000000p-66f,
    0x1.2ced440000000p+40f,
    0x1.ee41820000000p-12f,
    0x1.671b9e0000000p+32f,
    0x1.b61b6c0000000p+29f,
    0x1.7080840000000p+51f,
    0x1.940f0a0000000p+99f,
    0x1.45663e0000000p-35f,
    0x1.21b2ce0000000p+100f
};

inline const std::vector<float> du = {
    -0x1.0905da0000000p-1f,
    -0x1.9db18e0000000p-90f,
    -0x1.e9a9b60000000p-21f,
    0x1.fa28040000000p-68f,
    -0x1.f476fe0000000p+38f,
    -0x1.7511fe0000000p-14f,
    0x1.c038540000000p+30f,
    -0x1.5796d20000000p+28f,
    -0x1.053aae0000000p+48f,
    0x1.c54a120000000p+97f,
    0x1.53a4a00000000p-37f,
    0x0.0p+0f
};

inline const std::vector<float> rhs = {
    -0x1.b195c80000000p-1f,
    0x1.0707640000000p-88f,
    -0x1.5b12be0000000p-19f,
    0x1.46906e0000000p-66f,
    -0x1.ecb8a80000000p+38f,
    0x1.e84f880000000p-14f,
    0x1.acde340000000p+31f,
    0x1.aae11c0000000p+29f,
    -0x1.00992a0000000p+45f,
    0x1.92bb640000000p+98f,
    -0x1.25bae80000000p-39f,
    -0x1.48076a0000000p+99f
};

inline const std::vector<int> rowExponent = {
    1, -89, -20, -67, 39, -12, 31, 29, 50, 98, -36, 99
};

inline const std::vector<double> expected = {
    -0.13385480512149009,
    0.53945622140774074,
    -0.90263546583272725,
    0.73354261271435983,
    -0.28858736009660624,
    0.29690155356018361,
    0.29087879045024073,
    0.96130567474263351,
    -0.18642278005430693,
    0.47880530246664732,
    0.17308811573931698,
    -0.59463244068649967
};
}  // namespace gtsv2_mixed_scale_fixture

// Independent counterexamples to unconditional row equilibration. References
// come from 500-decimal-digit dense elimination of the exact FP32 inputs.
namespace gtsv2_pivot_scale_fixture {
struct Case {
    std::vector<float> dl, d, du, rhs;
    std::vector<double> expected;
};
inline const std::vector<Case> cases = {
    {
        {
            0x0.0p+0f,
            0x1.8000000000000p+11f,
            0x1.0000000000000p-65f,
            -0x1.0000000000000p-82f
        },
        {
            0x1.0000000000000p+62f,
            -0x1.0000000000000p-38f,
            -0x1.0000000000000p-112f,
            -0x1.0000000000000p+35f
        },
        {
            -0x1.0000000000000p-112f,
            -0x1.0000000000000p+2f,
            -0x1.0000000000000p+64f,
            0x0.0p+0f
        },
        {
            -0x1.0000000000000p+60f,
            -0x1.0000000000000p+9f,
            0x1.0000020000000p+64f,
            0x1.0000000000000p+35f
        },
        {
            -0.25,
            8.1129638716838137e+31,
            -7.3786976569716113e+19,
            -0.99999999999999956
        }
    },
    {
        {
            0x0.0p+0f,
            -0x1.8000000000000p+37f,
            0x1.4000000000000p-76f
        },
        {
            0x1.8000000000000p+90f,
            0x1.8000000000000p+38f,
            0x1.8000000000000p-76f
        },
        {
            -0x1.4000000000000p+90f,
            -0x1.8000000000000p-42f,
            0x0.0p+0f
        },
        {
            0x1.8000000000000p+114f,
            -0x1.4000000000000p+37f,
            0x1.8000000000000p+38f
        },
        {
            24571431203.976189,
            29465584785.57143,
            2.0769187434139311e+34
        }
    },
    {
        {
            0x0.0p+0f,
            -0x1.ca8c7c0000000p-88f,
            -0x1.9812fc0000000p-110f,
            0x1.7e3ee80000000p-5f,
            -0x1.a792f40000000p-30f,
            0x1.a29f740000000p+9f,
            0x1.85d3040000000p+24f,
            -0x1.ccbe340000000p-51f,
            0x1.ea755e0000000p-64f,
            -0x1.5ddcd00000000p+51f,
            -0x1.66a2340000000p-46f,
            -0x1.dfe79e0000000p-42f
        },
        {
            0x1.a5cebe0000000p+99f,
            0x1.d4e0800000000p-72f,
            0x1.481dd40000000p-11f,
            0x1.751bb40000000p-4f,
            0x1.367c160000000p-29f,
            0x1.2e0aba0000000p+10f,
            0x1.f9b10e0000000p+24f,
            0x1.44322a0000000p+54f,
            0x1.4af8fa0000000p-63f,
            0x1.0727760000000p+75f,
            0x1.7684ca0000000p+113f,
            0x1.0ead940000000p-41f
        },
        {
            0x1.2420e40000000p+99f,
            -0x1.a9566c0000000p-72f,
            0x1.369c540000000p-11f,
            -0x1.5f383e0000000p-62f,
            0x1.bd2ee40000000p-75f,
            0x1.0605800000000p-88f,
            -0x1.7a87e80000000p-54f,
            -0x1.2f653e0000000p+54f,
            0x1.c32b6e0000000p-95f,
            -0x1.e8a2500000000p+74f,
            0x1.8029360000000p+112f,
            0x0.0p+0f
        },
        {
            0x1.9cb7c60000000p-121f,
            -0x1.8a7f5c0000000p-101f,
            0x1.ef80b60000000p-47f,
            -0x1.36e7840000000p-39f,
            0x1.3724e40000000p+97f,
            0x1.18066c0000000p-107f,
            0x1.05641e0000000p-49f,
            0x1.d90f980000000p-103f,
            -0x1.31bf1c0000000p+37f,
            -0x1.8bb5380000000p+104f,
            -0x1.3683320000000p-21f,
            -0x1.55c0220000000p+19f
        },
        {
            3.2147323044830793e+20,
            -4.6417880234323621e+20,
            -5.116994107956379e+20,
            5.4053853055322731e+20,
            8.5251260784714897e+37,
            -5.9078106141331697e+37,
            4.5541747722450515e+37,
            -6.4716035468994865e+29,
            -6.9152899061666655e+29,
            -5.4799167552981807e+22,
            5.1322251097470362e+17,
            -1.0006816998566317e+18
        }
    },
};
}  // namespace gtsv2_pivot_scale_fixture
#endif
