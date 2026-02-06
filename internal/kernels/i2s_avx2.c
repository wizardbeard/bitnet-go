#include <immintrin.h>
#include <stdint.h>

static inline float hsum256_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

static inline float hsum128_ps(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

static float i2s_lut[256][4];
static int i2s_lut_init = 0;

static void init_i2s_lut(void) {
    if (i2s_lut_init) return;
    for (int b = 0; b < 256; b++) {
        uint8_t v = (uint8_t)b;
        for (int i = 0; i < 4; i++) {
            uint8_t q = (v >> (6 - 2*i)) & 0x3;
            float w = 0.0f;
            if (q == 0) w = -1.0f;
            else if (q == 2) w = 1.0f;
            i2s_lut[b][i] = w;
        }
    }
    i2s_lut_init = 1;
}

void matvec_i2s_avx2(float *dst, const uint8_t *packed, int rows, int cols, const float *vec, float scale) {
    const float map[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            float w[8];
            for (int i = 0; i < 8; i++) {
                int idx = r + rows * (c + i);
                uint8_t b = packed[idx / 4];
                uint8_t shift = (uint8_t)(6 - 2 * (idx % 4));
                uint8_t q = (b >> shift) & 0x3;
                w[i] = map[q];
            }
            __m256 vw = _mm256_loadu_ps(w);
            __m256 vx = _mm256_loadu_ps(vec + c);
            __m256 prod = _mm256_mul_ps(vw, vx);
            sum += hsum256_ps(prod);
        }
        for (; c < cols; c++) {
            int idx = r + rows * c;
            uint8_t b = packed[idx / 4];
            uint8_t shift = (uint8_t)(6 - 2 * (idx % 4));
            uint8_t q = (b >> shift) & 0x3;
            sum += map[q] * vec[c];
        }
        dst[r] = sum * scale;
    }
}

void matvec_t_i2s_avx2(float *dst, const uint8_t *packed, int rows, int cols, const float *vec, float scale) {
    init_i2s_lut();
    const float map[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
    for (int c = 0; c < cols; c++) {
        float sum = 0.0f;
        if ((rows % 4) == 0) {
            const uint8_t *p = packed + (rows * c) / 4;
            __m128 acc = _mm_setzero_ps();
            for (int r = 0; r < rows; r += 4) {
                const float *w = i2s_lut[p[r/4]];
                __m128 vw = _mm_loadu_ps(w);
                __m128 vx = _mm_loadu_ps(vec + r);
                acc = _mm_add_ps(acc, _mm_mul_ps(vw, vx));
            }
            sum = hsum128_ps(acc) * scale;
        } else {
            int r = 0;
            for (; r + 8 <= rows; r += 8) {
                float w[8];
                for (int i = 0; i < 8; i++) {
                    int idx = (r + i) + rows * c;
                    uint8_t b = packed[idx / 4];
                    uint8_t shift = (uint8_t)(6 - 2 * (idx % 4));
                    uint8_t q = (b >> shift) & 0x3;
                    w[i] = map[q];
                }
                __m256 vw = _mm256_loadu_ps(w);
                __m256 vx = _mm256_loadu_ps(vec + r);
                __m256 prod = _mm256_mul_ps(vw, vx);
                sum += hsum256_ps(prod);
            }
            for (; r < rows; r++) {
                int idx = r + rows * c;
                uint8_t b = packed[idx / 4];
                uint8_t shift = (uint8_t)(6 - 2 * (idx % 4));
                uint8_t q = (b >> shift) & 0x3;
                sum += map[q] * vec[r];
            }
            sum *= scale;
        }
        dst[c] = sum;
    }
}
