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

static inline uint8_t i2s_get(const uint8_t *packed, int idx) {
    const int block = 128;
    const int block_bytes = 32;
    int bi = idx / block;
    int off = idx % block;
    int gp = off % 32;
    int group = off / 32;
    const uint8_t b = packed[bi * block_bytes + gp];
    const uint8_t shift = (uint8_t)(6 - 2 * group);
    return (b >> shift) & 0x3;
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
                uint8_t q = i2s_get(packed, idx);
                w[i] = map[q];
            }
            __m256 vw = _mm256_loadu_ps(w);
            __m256 vx = _mm256_loadu_ps(vec + c);
            __m256 prod = _mm256_mul_ps(vw, vx);
            sum += hsum256_ps(prod);
        }
        for (; c < cols; c++) {
            int idx = r + rows * c;
            uint8_t q = i2s_get(packed, idx);
            sum += map[q] * vec[c];
        }
        dst[r] = sum * scale;
    }
}

void matvec_t_i2s_avx2(float *dst, const uint8_t *packed, int rows, int cols, const float *vec, float scale) {
    const float map[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
    for (int c = 0; c < cols; c++) {
        float sum = 0.0f;
        int r = 0;
        for (; r + 8 <= rows; r += 8) {
            float w[8];
            for (int i = 0; i < 8; i++) {
                int idx = (r + i) + rows * c;
                uint8_t q = i2s_get(packed, idx);
                w[i] = map[q];
            }
            __m256 vw = _mm256_loadu_ps(w);
            __m256 vx = _mm256_loadu_ps(vec + r);
            __m256 prod = _mm256_mul_ps(vw, vx);
            sum += hsum256_ps(prod);
        }
        for (; r < rows; r++) {
            int idx = r + rows * c;
            uint8_t q = i2s_get(packed, idx);
            sum += map[q] * vec[r];
        }
        sum *= scale;
        dst[c] = sum;
    }
}
