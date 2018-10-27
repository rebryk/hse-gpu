#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


gpu::gpu_mem_32u prefix_sum(ocl::Kernel &partial_prefix_sum,
                            ocl::Kernel &add_shift,
                            const gpu::gpu_mem_32u &as_gpu,
                            const unsigned int n,
                            const unsigned int work_group_size) {
    unsigned int global_work_size = (n + work_group_size - 1) / work_group_size * work_group_size;

    gpu::gpu_mem_32u block_sum;
    const unsigned int new_sz = (n + work_group_size - 1) / work_group_size;
    block_sum.resizeN(new_sz);

    gpu::gpu_mem_32u pref_sum;
    pref_sum.resizeN(n);

    partial_prefix_sum.exec(gpu::WorkSize(work_group_size, global_work_size), as_gpu, n, block_sum, pref_sum);

    if (n <= work_group_size) {
        return pref_sum;
    }

    gpu::gpu_mem_32u shift = prefix_sum(partial_prefix_sum, add_shift, block_sum, new_sz, work_group_size);

    add_shift.exec(gpu::WorkSize(work_group_size, global_work_size), as_gpu, n, shift, pref_sum);

    return pref_sum;
}

void radix_sort(ocl::Kernel &fill_bits,
                ocl::Kernel &replace,
                ocl::Kernel &partial_prefix_sum,
                ocl::Kernel &add_shift,
                gpu::gpu_mem_32u &as_gpu,
                const unsigned int n,
                const unsigned int work_group_size,
                const unsigned int bits_count) {
    unsigned int global_work_size = (n + work_group_size - 1) / work_group_size * work_group_size;

    gpu::gpu_mem_32u bs_gpu;
    bs_gpu.resizeN(n);

    for (int shift = 0; shift < 32; shift += bits_count) {
        gpu::gpu_mem_32u bits;
        const unsigned int new_sz = (1 << bits_count) * n;
        bits.resizeN(new_sz);

        fill_bits.exec(gpu::WorkSize(work_group_size, global_work_size), as_gpu, n, bits, shift);

        gpu::gpu_mem_32u pref_sum = prefix_sum(partial_prefix_sum, add_shift, bits, new_sz, work_group_size);

        replace.exec(gpu::WorkSize(work_group_size, global_work_size), as_gpu, n, pref_sum, bs_gpu, shift);

        bs_gpu.swap(as_gpu);
    }
}


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    {
        ocl::Kernel fill_bits(radix_kernel, radix_kernel_length, "fill_bits");
        fill_bits.compile();

        ocl::Kernel replace(radix_kernel, radix_kernel_length, "replace");
        replace.compile();

        ocl::Kernel partial_prefix_sum(radix_kernel, radix_kernel_length, "partial_prefix_sum");
        partial_prefix_sum.compile();

        ocl::Kernel add_shift(radix_kernel, radix_kernel_length, "add_shift");
        add_shift.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            radix_sort(fill_bits, replace, partial_prefix_sum, add_shift, as_gpu, n, 256, 2);
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
