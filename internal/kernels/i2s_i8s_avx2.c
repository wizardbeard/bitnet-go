#include <stdint.h>
#include <immintrin.h>

static int g_i2s_table_init = 0;
static int8_t g_i2s_table[256][4];

static void init_i2s_table(void) {
    if (g_i2s_table_init) {
        return;
    }
    for (int b = 0; b < 256; b++) {
        g_i2s_table[b][0] = (int8_t)((b >> 6) & 0x3);
        g_i2s_table[b][1] = (int8_t)((b >> 4) & 0x3);
        g_i2s_table[b][2] = (int8_t)((b >> 2) & 0x3);
        g_i2s_table[b][3] = (int8_t)(b & 0x3);
    }
    g_i2s_table_init = 1;
}

static inline int32_t dot_i8x128_avx2(const int8_t *a, const int8_t *b) {
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < 128; i += 16) {
        __m128i va8 = _mm_loadu_si128((const __m128i *)(a + i));
        __m128i vb8 = _mm_loadu_si128((const __m128i *)(b + i));
        __m256i va16 = _mm256_cvtepi8_epi16(va8);
        __m256i vb16 = _mm256_cvtepi8_epi16(vb8);
        __m256i prod = _mm256_madd_epi16(va16, vb16);
        acc = _mm256_add_epi32(acc, prod);
    }
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(sum128);
}

void matvec_t_i2s_i8s_avx2(float *dst, const unsigned char *packed, int rows, int cols, const signed char *vec, float weight_scale, float act_scale, int act_sum) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    init_i2s_table();
    const float scale = (act_scale == 0.0f) ? 0.0f : (weight_scale / act_scale);
    const int block = 128;
    const int block_bytes = 32;
    const int blocks = rows / block;
    for (int c = 0; c < cols; c++) {
        int32_t sum = 0;
        int r = 0;
        if (rows % 128 == 0) {
            int basePacked = (c * blocks) * block_bytes;
            int8_t wblock[128];
            for (int b = 0; b < blocks; b++) {
                const unsigned char *p = packed + basePacked;
                for (int gp = 0; gp < 32; gp++) {
                    const int8_t *vals = g_i2s_table[p[gp]];
                    wblock[gp] = vals[0];
                    wblock[32 + gp] = vals[1];
                    wblock[64 + gp] = vals[2];
                    wblock[96 + gp] = vals[3];
                }
                sum += dot_i8x128_avx2(wblock, (const int8_t *)(vec + r));
                basePacked += block_bytes;
                r += block;
            }
        } else {
            for (; r < rows; r++) {
                int idx = r + rows * c;
                int bi = idx / block;
                int off = idx % block;
                int gp = off % 32;
                int group = off / 32;
                unsigned char b = packed[bi * block_bytes + gp];
                int8_t q = g_i2s_table[b][group];
                sum += (int32_t)q * (int32_t)vec[r];
            }
        }
        dst[c] = (float)(sum - act_sum) * scale;
    }
}

void matvec_i2s_i8s_avx2(float *dst, const unsigned char *packed, int rows, int cols, const signed char *vec, float weight_scale, float act_scale, int act_sum) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    init_i2s_table();
    const float scale = (act_scale == 0.0f) ? 0.0f : (weight_scale / act_scale);
    const int block = 128;
    const int block_bytes = 32;
    if (rows % 128 != 0) {
        for (int r = 0; r < rows; r++) {
            int32_t sum = 0;
            for (int c = 0; c < cols; c++) {
                int idx = r + rows * c;
                int bi = idx / block;
                int off = idx % block;
                int gp = off % 32;
                int group = off / 32;
                unsigned char b = packed[bi * block_bytes + gp];
                int8_t q = g_i2s_table[b][group];
                sum += (int32_t)q * (int32_t)vec[c];
            }
            dst[r] = (float)(sum - act_sum) * scale;
        }
        return;
    }
    const int blocks = rows / block;
    int32_t sums[128];
    for (int rb = 0; rb < rows; rb += 128) {
        for (int i = 0; i < 128; i++) {
            sums[i] = 0;
        }
        for (int c = 0; c < cols; c++) {
            int idx = rb + rows * c;
            int bi = idx / block;
            const unsigned char *p = packed + bi * block_bytes;
            int8_t wblock[128];
            for (int gp = 0; gp < 32; gp++) {
                const int8_t *vals = g_i2s_table[p[gp]];
                wblock[gp] = vals[0];
                wblock[32 + gp] = vals[1];
                wblock[64 + gp] = vals[2];
                wblock[96 + gp] = vals[3];
            }
            const __m256i vv = _mm256_set1_epi32((int32_t)vec[c]);
            for (int i = 0; i < 128; i += 8) {
                __m128i w8 = _mm_loadl_epi64((const __m128i *)(wblock + i));
                __m256i w32 = _mm256_cvtepi8_epi32(w8);
                __m256i prod = _mm256_mullo_epi32(w32, vv);
                __m256i acc = _mm256_loadu_si256((const __m256i *)(sums + i));
                acc = _mm256_add_epi32(acc, prod);
                _mm256_storeu_si256((__m256i *)(sums + i), acc);
            }
        }
        for (int i = 0; i < 128; i++) {
            dst[rb + i] = (float)(sums[i] - act_sum) * scale;
        }
    }
}
