#include <stdint.h>
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static inline uint8_t i2s_packed_at(const uint8_t *packed, int idx) {
    if (idx < 0) {
        return 0;
    }
    const int block = 128;
    const int block_bytes = 32;
    int bi = idx / block;
    int off = idx % block;
    int gp = off % 32;
    int group = off / 32;
    int p = bi * block_bytes + gp;
    uint8_t v = packed[p];
    uint8_t shift = (uint8_t)(6 - 2 * group);
    return (uint8_t)((v >> shift) & 0x3);
}

void matvec_i2s_cgo(float *dst, const uint8_t *packed, int rows, int cols, const float *vec, float scale) {
    static const float lut[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (rows % 128 == 0) {
        int blocks = rows / 128;
        for (int c = 0; c < cols; c++) {
            float v = vec[c] * scale;
            if (v == 0.0f) {
                continue;
            }
            int base_packed = (c * rows / 128) * 32;
            int row_base = 0;
            for (int b = 0; b < blocks; b++) {
                const uint8_t *p = packed + base_packed;
                for (int gp = 0; gp < 32; gp += 4) {
                    uint8_t b0 = p[gp];
                    uint8_t b1 = p[gp + 1];
                    uint8_t b2 = p[gp + 2];
                    uint8_t b3 = p[gp + 3];

                    float32x4_t w0 = {lut[b0 >> 6], lut[b1 >> 6], lut[b2 >> 6], lut[b3 >> 6]};
                    float32x4_t w1 = {lut[(b0 >> 4) & 0x3], lut[(b1 >> 4) & 0x3], lut[(b2 >> 4) & 0x3], lut[(b3 >> 4) & 0x3]};
                    float32x4_t w2 = {lut[(b0 >> 2) & 0x3], lut[(b1 >> 2) & 0x3], lut[(b2 >> 2) & 0x3], lut[(b3 >> 2) & 0x3]};
                    float32x4_t w3 = {lut[b0 & 0x3], lut[b1 & 0x3], lut[b2 & 0x3], lut[b3 & 0x3]};

                    int r0 = row_base + gp;
                    float32x4_t d0 = vld1q_f32(dst + r0);
                    d0 = vfmaq_n_f32(d0, w0, v);
                    vst1q_f32(dst + r0, d0);

                    int r1 = r0 + 32;
                    float32x4_t d1 = vld1q_f32(dst + r1);
                    d1 = vfmaq_n_f32(d1, w1, v);
                    vst1q_f32(dst + r1, d1);

                    int r2 = r0 + 64;
                    float32x4_t d2 = vld1q_f32(dst + r2);
                    d2 = vfmaq_n_f32(d2, w2, v);
                    vst1q_f32(dst + r2, d2);

                    int r3 = r0 + 96;
                    float32x4_t d3 = vld1q_f32(dst + r3);
                    d3 = vfmaq_n_f32(d3, w3, v);
                    vst1q_f32(dst + r3, d3);
                }
                base_packed += 32;
                row_base += 128;
            }
        }
        return;
    }
#endif
    for (int c = 0; c < cols; c++) {
        float v = vec[c] * scale;
        if (v == 0.0f) {
            continue;
        }
        for (int r = 0; r < rows; r++) {
            int idx = r + rows * c;
            uint8_t q = i2s_packed_at(packed, idx);
            dst[r] += lut[q] * v;
        }
    }
}

void matvec_t_i2s_cgo(float *dst, const uint8_t *packed, int rows, int cols, const float *vec, float scale) {
    static const float lut[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (rows % 128 == 0) {
        int blocks = rows / 128;
        for (int c = 0; c < cols; c++) {
            float sum = 0.0f;
            int base_packed = (c * rows / 128) * 32;
            int row_base = 0;
            for (int b = 0; b < blocks; b++) {
                const uint8_t *p = packed + base_packed;
                for (int gp = 0; gp < 32; gp += 4) {
                    uint8_t b0 = p[gp];
                    uint8_t b1 = p[gp + 1];
                    uint8_t b2 = p[gp + 2];
                    uint8_t b3 = p[gp + 3];

                    float32x4_t w0 = {lut[b0 >> 6], lut[b1 >> 6], lut[b2 >> 6], lut[b3 >> 6]};
                    float32x4_t w1 = {lut[(b0 >> 4) & 0x3], lut[(b1 >> 4) & 0x3], lut[(b2 >> 4) & 0x3], lut[(b3 >> 4) & 0x3]};
                    float32x4_t w2 = {lut[(b0 >> 2) & 0x3], lut[(b1 >> 2) & 0x3], lut[(b2 >> 2) & 0x3], lut[(b3 >> 2) & 0x3]};
                    float32x4_t w3 = {lut[b0 & 0x3], lut[b1 & 0x3], lut[b2 & 0x3], lut[b3 & 0x3]};

                    int r0 = row_base + gp;
                    float32x4_t v0 = vld1q_f32(vec + r0);
                    float32x4_t v1 = vld1q_f32(vec + r0 + 32);
                    float32x4_t v2 = vld1q_f32(vec + r0 + 64);
                    float32x4_t v3 = vld1q_f32(vec + r0 + 96);

                    float32x4_t acc = vmulq_f32(w0, v0);
                    acc = vfmaq_f32(acc, w1, v1);
                    acc = vfmaq_f32(acc, w2, v2);
                    acc = vfmaq_f32(acc, w3, v3);
                    sum += vaddvq_f32(acc);
                }
                base_packed += 32;
                row_base += 128;
            }
            dst[c] = sum * scale;
        }
        return;
    }
#endif
    for (int c = 0; c < cols; c++) {
        float sum = 0.0f;
        for (int r = 0; r < rows; r++) {
            int idx = r + rows * c;
            uint8_t q = i2s_packed_at(packed, idx);
            sum += lut[q] * vec[r];
        }
        dst[c] = sum * scale;
    }
}
