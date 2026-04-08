#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h> // 必须包含这个
#endif

int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
{
    // TODO check dimensions
    long batch, ref_nb, query_nb, dim, k;
    batch = ref.size(0);
    dim = ref.size(1);
    k = idx.size(1);
    ref_nb = ref.size(2);
    query_nb = query.size(2);

    // 【修复1】将废弃的 .data<float>() 替换为现代的 .data_ptr<float>()
    float *ref_dev = ref.data_ptr<float>();
    float *query_dev = query.data_ptr<float>();
    long *idx_dev = (long*)idx.data_ptr<int64_t>();

    // 【修复2】将废弃的 .type().is_cuda() 直接替换为 .is_cuda()
    if (ref.is_cuda()) {

#ifdef WITH_CUDA
        // 1. 使用原生的 cudaMalloc 替换 THCudaMalloc
        float *dist_dev = nullptr;
        cudaError_t err_malloc = cudaMalloc((void**)&dist_dev, ref_nb * query_nb * sizeof(float));
        if (err_malloc != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err_malloc));
            return 0;
        }

        for (int b = 0; b < batch; b++)
        {
            // 2. 获取当前流，注意命名空间是 at::cuda
            knn_device(ref_dev + b * dim * ref_nb, ref_nb, 
                       query_dev + b * dim * query_nb, query_nb, dim, k,
                       dist_dev, idx_dev + b * k * query_nb, 
                       (cudaStream_t)at::cuda::getCurrentCUDAStream().stream());
        }

        // 3. 使用 cudaFree 替换 THCudaFree
        cudaFree(dist_dev);

        // 4. 错误检查 (重新显式声明 cudaError_t err 避免 C2065 错误)
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("error in knn: %s\n", cudaGetErrorString(err));
            // THError 也是旧的，改用 AT_ERROR
            AT_ERROR("knn failed");
        }
        return 1;
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }

    // CPU 回退逻辑
    float *dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
    long *ind_buf = (long*)malloc(ref_nb * sizeof(long));
    for (int b = 0; b < batch; b++) {
        knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
                dist_dev, idx_dev + b * k * query_nb, ind_buf);
    }

    free(dist_dev);
    free(ind_buf);

    return 1;
}