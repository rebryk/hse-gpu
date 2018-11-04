#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 32

__kernel void merge_sort_slow(__global float* a,
                              const unsigned int n,
                              const unsigned int max_len)
{
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local float local_buff_a[WORK_GROUP_SIZE];
    __local float local_buff_b[WORK_GROUP_SIZE];

    __local float* local_a = local_buff_a;
    __local float* local_b = local_buff_b;
    __local float* tmp;

    local_a[local_id] = global_id < n ? a[global_id] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int sz = 1; sz <= max_len; sz *= 2) {
        unsigned int l = 2 * local_id * sz;
        unsigned int m = l + sz;
        unsigned int r = m + sz;

        if (local_id < WORK_GROUP_SIZE / (2 * sz)) {
            for (unsigned int it = l, i = l, j = m; it < r; ++it) {
                if (j == r || (i < m && local_a[i] <= local_a[j])) {
                    local_b[it] = local_a[i];
                    ++i;
                } else {
                    local_b[it] = local_a[j];
                    ++j;
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }

        tmp = local_a;
        local_a = local_b;
        local_b = tmp;
    }

    if (global_id < n) {
        a[global_id] = local_a[local_id];
    }
}

int find_diag(__global float* a, __global float* b, int block_sz, int diag) {
    int l = max(0, diag - block_sz);
    int r = min(diag, block_sz);
    int m;

    while (l < r) {
        m = (l + r) / 2;
        if (a[m] <= b[diag - m - 1]) {
            l = m + 1;
        } else {
            r = m;
        }
    }

    return l;
}

__kernel void merge_sort_fast(__global float* a,
                               __global float* b,
                               const unsigned int n,
                               const unsigned int block_sz,
                               const unsigned int step)
{
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    const int threads_per_block = block_sz / step;
    const int global_block = 2 * (global_id / threads_per_block);
    const int local_block = global_id % threads_per_block;

    int offset_1 = global_block * block_sz;
    int offset_2 = (global_block + 1) * block_sz;

    int diag_left = 2 * local_block * step;
    int diag_right = 2 * (local_block + 1) * step;

    __global float* block_1 = a + offset_1;
    __global float* block_2 = a + offset_2;
    int l_1 = find_diag(block_1, block_2, block_sz, diag_left);
    int r_1 = find_diag(block_1, block_2, block_sz, diag_right);

    int l_2 = diag_left - l_1;
    int r_2 = diag_right - r_1;

    __global float* res = b + offset_1;
    int i = l_1;
    int j = l_2;
    while (i < r_1 || j < r_2) {
        if (j == r_2 || (i < r_1 && block_1[i] <= block_2[j])) {
            res[i + j] = block_1[i];
            ++i;
        } else {
            res[i + j] = block_2[j];
            ++j;
        }
    }
}
