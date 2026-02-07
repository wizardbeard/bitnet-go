#include <stdint.h>

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
