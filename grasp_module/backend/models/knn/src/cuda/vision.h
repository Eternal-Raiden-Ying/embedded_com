#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
// #include <THC/THC.h> // pytorch older than 1.11

void knn_device(float* ref_dev, int ref_width,
    float* query_dev, int query_width,
    int height, int k, float* dist_dev, long* ind_dev, cudaStream_t stream);
