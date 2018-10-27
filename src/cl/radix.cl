#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SZ 256
#define BITS_COUNT 2
#define MASK (1 << BITS_COUNT) - 1


__kernel void fill_bits(__global const unsigned int *as,
                        const unsigned int n,
                        __global unsigned int *count,
                        const unsigned int shift) {
    const unsigned int global_id = get_global_id(0);

    if (global_id < n) {
        const unsigned int value = ((as[global_id] >> shift) & MASK);
        for (unsigned int i = 0; i < (1 << BITS_COUNT); ++i) {
            count[n * i + global_id] = (value == i);
        }
    }
}


__kernel void replace(__global const unsigned int *as,
                      const unsigned int n,
                      __global const unsigned int *pref,
                      __global unsigned int *result,
                      const unsigned int shift) {
    const unsigned int global_id = get_global_id(0);
    if (global_id < n) {
        const unsigned int x = as[global_id];
        const unsigned int value = ((x >> shift) & MASK);
        const unsigned int index = pref[n * value + global_id] - 1;
        result[index] = x;
    }
}


__kernel void partial_prefix_sum(__global const unsigned int *as,
                                 const unsigned int n,
                                 __global unsigned int *block_sum,
                                 __global unsigned int *result) {
    const unsigned int local_id = get_local_id(0);
    const unsigned int global_id = get_global_id(0);

    __local unsigned int data_[WORK_GROUP_SZ];
    __local unsigned int pref_[WORK_GROUP_SZ];

    __local unsigned int *data = data_;
    __local unsigned int *pref = pref_;

    __local unsigned int *tmp;

    data[local_id] = global_id < n ? as[global_id] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int sz = 1; sz < WORK_GROUP_SZ; sz *= 2) {
        if (local_id >= sz) {
            pref[local_id] = data[local_id] + data[local_id - sz];
        } else {
            pref[local_id] = data[local_id];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // swap arrays
        tmp = data;
        data = pref;
        pref = tmp;
    }

    if (global_id < n) {
        result[global_id] = data[local_id];
    }

    if (local_id == 0) {
        block_sum[global_id / WORK_GROUP_SZ] = data[WORK_GROUP_SZ - 1];
    }
}


__kernel void add_shift(__global const unsigned int *as,
                        const unsigned int n,
                        __global const unsigned int *shift,
                        __global unsigned int *result) {
    const unsigned int global_id = get_global_id(0);
    if (global_id < n) {
        const int index = global_id / WORK_GROUP_SZ - 1;
        if (index >= 0) {
            result[global_id] += shift[index];
        }
    }
}