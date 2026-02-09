#include <stdint.h>

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
            for (int b = 0; b < blocks; b++) {
                const unsigned char *p = packed + basePacked;
                for (int gp = 0; gp < 32; gp++) {
                    const int8_t *vals = g_i2s_table[p[gp]];
                    int idx = r + gp;
                    sum += (int32_t)vals[0] * (int32_t)vec[idx];
                    sum += (int32_t)vals[1] * (int32_t)vec[idx + 32];
                    sum += (int32_t)vals[2] * (int32_t)vec[idx + 64];
                    sum += (int32_t)vals[3] * (int32_t)vec[idx + 96];
                }
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
