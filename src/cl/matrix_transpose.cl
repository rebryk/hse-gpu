#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float *a,
                               __global float *at,
                               unsigned int m,
                               unsigned int k) {
    const size_t i = get_global_id(0);
    const size_t j = get_global_id(1);

    __local float tile[TILE_SIZE * TILE_SIZE];

    const size_t local_i = get_local_id(0);
    const size_t local_j = get_local_id(1);

    tile[local_j * TILE_SIZE + local_i] = a[j * k + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_j < local_i) {
        float tmp = tile[local_j * TILE_SIZE + local_i];
        tile[local_j * TILE_SIZE + local_i] = tile[local_i * TILE_SIZE + local_j];
        tile[local_i * TILE_SIZE + local_j] = tmp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    size_t new_j = (i / TILE_SIZE) * TILE_SIZE + local_j;
    size_t new_i = (j / TILE_SIZE) * TILE_SIZE + local_i;
    at[new_j * m + new_i] = tile[local_j * TILE_SIZE + local_i];
}