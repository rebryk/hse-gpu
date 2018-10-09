#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256

__kernel void sum(__global const unsigned int* x,
                  __global unsigned int* sum,
                  unsigned int n)
{
    const unsigned int local_id = get_local_id(0);
    const unsigned int global_id = get_global_id(0);

    __local unsigned int local_x[WORK_GROUP_SIZE];
    local_x[local_id] = global_id < n ? x[global_id] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int sz = WORK_GROUP_SIZE; sz > 1; sz /= 2) {
        if (2 * local_id < sz) {
            local_x[local_id] += local_x[local_id + sz / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        atomic_add(sum, local_x[0]);
    }
}
