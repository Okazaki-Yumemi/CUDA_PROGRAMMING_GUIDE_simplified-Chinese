---
title: 2.1 Intro to CUDA C++
description: CUDA C++、kernel launch、内建变量和错误检查
---

# 2.1 Intro to CUDA C++

## Compilation with NVCC

CUDA C++ 文件可以同时包含 host code 和 device code。`nvcc` 识别 CUDA 扩展，把 device 部分编译为 GPU 代码，把 host 部分交给系统 C++ 编译器并完成链接。编译时要决定 C++ 标准、host 编译器、目标 compute capability、是否保留 PTX，以及所需 CUDA 库。

## Kernels

用 `__global__` 声明可从 host 启动、在 device 执行的函数。结果通常写入 device/global/managed memory：

```cpp
__global__ void scale(float* data, int count, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) data[i] *= factor;
}
```

## Launching Kernels

CUDA C++ 使用 `<<<grid, block>>>` 指定执行配置：

```cpp
int threads = 256;
int blocks = (count + threads - 1) / threads;
scale<<<blocks, threads>>>(device_data, count, 2.0f);
```

网格大小通常向上取整，因此 kernel 必须做边界检查。block 太小可能降低调度效率，太大则可能受寄存器和 shared memory 限制。

## Thread and Grid Index Intrinsics

`threadIdx` 是线程在 block 内的坐标，`blockIdx` 是 block 在 grid 内的坐标，`blockDim` 和 `gridDim` 是对应维度。二维和三维数据要同时确认数学坐标、内存行列顺序和线性地址公式。

## Memory in GPU Computing

`cudaMalloc`/`cudaMemcpy` 把数据迁移显式化；`cudaMallocManaged` 简化 CPU/GPU 共享指针，但页面迁移和同步成本仍然存在。先选择正确的内存生命周期和访问布局，再决定是否需要更高级的统一内存策略。

## Synchronizing CPU and GPU

kernel launch 通常是异步的。`cudaDeviceSynchronize()` 等待设备工作完成；`cudaStreamSynchronize()` 只等待指定 stream；`__syncthreads()` 只同步同一 block，不能做 grid 级同步。同步既是正确性边界，也是可能的性能瓶颈。

## Error Checking

检查最近 launch 与检查设备完成是两步不同的动作：

```cpp
cudaError_t launch_error = cudaGetLastError();
cudaError_t async_error = cudaDeviceSynchronize();
```

生产代码应在错误中记录设备、stream、调用位置和上下文。继续阅读[第二部分长篇中文导读](../chapters/02-programming-gpus.html#2-1-cuda-c)。
