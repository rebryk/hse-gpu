#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256

__kernel void maxprefixsum(__global const int* block_sum,
                           __global const int* block_max,
                           __global int* block_sum_out,
                           __global int* block_max_out,
                           unsigned int n)
{
    const unsigned int local_id = get_local_id(0);
    const unsigned int global_id = get_global_id(0);

    __local int local_block_sum[WORK_GROUP_SIZE];
    __local int local_block_max[WORK_GROUP_SIZE];

    local_block_sum[local_id] = global_id < n ? block_sum[global_id] : 0;
    local_block_max[local_id] = global_id < n ? block_max[global_id] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int sz = WORK_GROUP_SIZE; sz > 1; sz /= 2) {
        int index = 2 * local_id;

        int sum_l = 0;
        int sum_r = 0;
        int max_l = 0;
        int max_r = 0;

        if (index + 1 < sz) {
            sum_l = local_block_sum[index];
            sum_r = local_block_sum[index + 1];
            max_l = local_block_max[index];
            max_r = local_block_max[index + 1];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        local_block_sum[local_id] = sum_l + sum_r;
        local_block_max[local_id] = max(max_l, sum_l + max_r);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        block_sum_out[global_id / WORK_GROUP_SIZE] = local_block_sum[0];
        block_max_out[global_id / WORK_GROUP_SIZE] = local_block_max[0];
    }
}
