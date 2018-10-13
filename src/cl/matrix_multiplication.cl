#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define TILE_SIZE 16

__kernel void matrix_multiplication(__global const float *a,
                                    __global const float *b,
                                    __global float *c,
                                    unsigned int m,
                                    unsigned int k,
                                    unsigned int n) {
    const size_t i = get_global_id(0);
    const size_t j = get_global_id(1);
    const size_t local_i = get_local_id(0);
    const size_t local_j = get_local_id(1);

    __local float tile_a[TILE_SIZE * TILE_SIZE];
    __local float tile_b[TILE_SIZE * TILE_SIZE];

    float sum = 0;
    for (size_t tile = 0; tile <= k / TILE_SIZE; ++tile) {
        if (j < m && tile * TILE_SIZE + local_i < k) {
            tile_a[local_j * TILE_SIZE + local_i] = a[j * k + tile * TILE_SIZE + local_i];
        } else {
            tile_a[local_j * TILE_SIZE + local_i] = 0;
        }

        if (tile * TILE_SIZE + local_j < k && i < n) {
            tile_b[local_j * TILE_SIZE + local_i] = b[(tile * TILE_SIZE + local_j) * n + i];
        } else {
            tile_b[local_j * TILE_SIZE + local_i] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (i < n && j < m) {
            for (size_t k = 0; k < TILE_SIZE; ++k) {
                sum += tile_a[local_j * TILE_SIZE + k] * tile_b[k * TILE_SIZE + local_i];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < n && j < m) {
        c[j * n + i] = sum;
    }
}