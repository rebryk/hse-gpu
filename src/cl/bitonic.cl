#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SZ 256

__kernel void bitonic_global(__global float *as,
                             unsigned int block_sz,
                             unsigned int sz, unsigned int n) {
    const unsigned int id = get_global_id(0);
    bool ascending = id % (2 * block_sz) < block_sz;

    if (id % (2 * sz) < sz && id + sz < n) {
        float a = as[id];
        float b = as[id + sz];
        if ((a > b) == ascending) {
            as[id + sz] = a;
            as[id] = b;
        }
    }
}


__kernel void bitonic_local(__global float *as,
                            unsigned int block_sz,
                            unsigned int sz,
                            unsigned int n) {
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local float data[WORK_GROUP_SZ];

    if (global_id < n) {
        data[local_id] = as[global_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    bool ascending = global_id % (2 * block_sz) < block_sz;

    while (sz >= 1) {
        if (global_id % (2 * sz) < sz && global_id + sz < n) {
            float a = data[local_id];
            float b = data[local_id + sz];
            if ((a > b) == ascending) {
                data[local_id + sz] = a;
                data[local_id] = b;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        sz /= 2;
    }

    if (global_id < n) {
        as[global_id] = data[local_id];
    }
}

