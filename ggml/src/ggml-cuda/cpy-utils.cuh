#pragma once

#include "ggml-common.h"

template<typename src_t, typename dst_t>
static __device__ __forceinline__ void convert_flt(const src_t * src, dst_t * dst) {
    if constexpr (std::is_same_v<src_t, dst_t>) {
        *dst = *src;
    } else {
        *dst = float(*src);
    }
}

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

static __device__ void quantize_f32_q6_0_block(const float * __restrict__ xi, block_q6_0 * __restrict__ y) {

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK6_0; ++j) {
        const float v  = xi[j];
        const float av = fabsf(xi[j]);
        if (amax < av) {
            amax = av;
            vmax = v;
        }
    }

    const float d  = vmax / -32;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;
    memset(y->qh, 0, QK6_0/4);

    for (int j = 0; j < QK6_0/2; ++j) {
        const float x0 = xi[0       + j]*id;
        const float x1 = xi[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(63, (int8_t)(x0 + 32.5f));
        const uint8_t xi1 = min(63, (int8_t)(x1 + 32.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        const uint8_t h = (xi0 >> 4) | ((xi1 >> 4) << 2);
        y->qh[j%(QK6_0/4)] |= (h << 4*(j/(QK6_0/4)));
    }
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q6_0(const char * cxi, char * cdsti) {
    quantize_f32_q6_0_block((const float *)cxi, (block_q6_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

// TQ3_0 constants for CUDA
__device__ static const float TQ3_0_CENTROIDS_D[8] = {
    -2.1519f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1519f
};
__device__ static const float TQ3_0_BOUNDARIES_D[7] = {
    -1.7479f, -1.0500f, -0.5005f, 0.0f, 0.5005f, 1.0500f, 1.7479f
};
__device__ static const float TQ3_0_SIGNS_D[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
};

static __device__ void tq3_0_wht_forward(float * __restrict__ buf) {
    for (int i = 0; i < 32; i++) {
        buf[i] *= TQ3_0_SIGNS_D[i];
    }
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j];
                float b = buf[j + step];
                buf[j]        = a + b;
                buf[j + step] = a - b;
            }
        }
    }
    const float norm = 1.0f / sqrtf(32.0f);
    for (int i = 0; i < 32; i++) {
        buf[i] *= norm;
    }
}

static __device__ void tq3_0_wht_inverse(float * __restrict__ buf) {
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j];
                float b = buf[j + step];
                buf[j]        = a + b;
                buf[j + step] = a - b;
            }
        }
    }
    const float norm = 1.0f / sqrtf(32.0f);
    for (int i = 0; i < 32; i++) {
        buf[i] *= norm * TQ3_0_SIGNS_D[i];
    }
}

static __device__ void quantize_f32_tq3_0_block(const float * __restrict__ x, block_tq3_0 * __restrict__ y) {
    float buf[QK_TQ3_0];

    // Compute RMS scale
    float sum_sq = 0.0f;
    for (int j = 0; j < QK_TQ3_0; j++) {
        sum_sq += x[j] * x[j];
    }
    float rms = sqrtf(sum_sq / QK_TQ3_0);
    if (rms < 1e-10f) { rms = 1.0f; }

    y->d = __float2half(rms);

    // Normalize to unit variance
    const float inv_rms = 1.0f / rms;
    for (int j = 0; j < QK_TQ3_0; j++) {
        buf[j] = x[j] * inv_rms;
    }

    // Apply forward WHT
    tq3_0_wht_forward(buf);

    // Quantize to nearest Lloyd-Max centroid and pack
    uint8_t indices[QK_TQ3_0];
    for (int j = 0; j < QK_TQ3_0; j++) {
        float v = buf[j];
        uint8_t idx = 0;
        for (int b = 0; b < 7; b++) {
            if (v > TQ3_0_BOUNDARIES_D[b]) { idx = b + 1; }
        }
        indices[j] = idx;
    }

    // Pack 3-bit indices (groups of 8 -> 3 bytes)
    for (int g = 0; g < 4; g++) {
        const uint8_t * idx = indices + g * 8;
        uint8_t * qp = y->qs + g * 3;
        qp[0] = (idx[0])      | (idx[1] << 3) | (idx[2] << 6);
        qp[1] = (idx[2] >> 2) | (idx[3] << 1) | (idx[4] << 4) | (idx[5] << 7);
        qp[2] = (idx[5] >> 1) | (idx[6] << 2) | (idx[7] << 5);
    }
}

static __device__ void dequantize_tq3_0_block(const block_tq3_0 * __restrict__ x, float * __restrict__ y) {
    const float d = __half2float(x->d);
    float buf[QK_TQ3_0];

    // Unpack 3-bit indices and look up centroids
    for (int g = 0; g < 4; g++) {
        const uint8_t * qp = x->qs + g * 3;
        int base = g * 8;
        buf[base + 0] = TQ3_0_CENTROIDS_D[ qp[0]       & 7];
        buf[base + 1] = TQ3_0_CENTROIDS_D[(qp[0] >> 3) & 7];
        buf[base + 2] = TQ3_0_CENTROIDS_D[((qp[0] >> 6) | (qp[1] << 2)) & 7];
        buf[base + 3] = TQ3_0_CENTROIDS_D[(qp[1] >> 1) & 7];
        buf[base + 4] = TQ3_0_CENTROIDS_D[(qp[1] >> 4) & 7];
        buf[base + 5] = TQ3_0_CENTROIDS_D[((qp[1] >> 7) | (qp[2] << 1)) & 7];
        buf[base + 6] = TQ3_0_CENTROIDS_D[(qp[2] >> 2) & 7];
        buf[base + 7] = TQ3_0_CENTROIDS_D[(qp[2] >> 5) & 7];
    }

    // Apply inverse WHT
    tq3_0_wht_inverse(buf);

    // Scale back
    for (int j = 0; j < QK_TQ3_0; j++) {
        y[j] = buf[j] * d;
    }
}

static __device__ void cpy_blck_f32_tq3_0(const char * cxi, char * cdsti) {
    quantize_f32_tq3_0_block((const float *)cxi, (block_tq3_0 *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_flt(const char * cxi, char * cdsti) {
    convert_flt((const src_t *)cxi, (dst_t *)cdsti);
}
