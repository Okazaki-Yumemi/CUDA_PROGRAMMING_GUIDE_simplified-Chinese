---
title: 2.1 CUDA C++ 入门
description: CUDA C++ 的编译、kernel、执行配置、内存、同步、错误检查与线程块集群
---

# 2.1 CUDA C++ 入门（Intro to CUDA C++）

> 本页按 NVIDIA CUDA Programming Guide Release 13.3 的官方网页逐段翻译。原文页面：[2.1. Intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)；官方页面标注的更新时间为 2026-05-27。

本章通过展示 CUDA 编程模型在 C++ 中的表达方式，介绍 CUDA 编程模型的一些基本概念。

本编程指南重点介绍 CUDA Runtime API。CUDA Runtime API 是在 C++ 中使用 CUDA 最常见的方式，它建立在更底层的 CUDA Driver API 之上。

[CUDA Runtime API 与 CUDA Driver API](../01-introduction/cuda-platform.html#1321-cuda-runtime-api-与-cuda-driver-api) 介绍两种 API 的区别；[CUDA Driver API](../03-advanced-cuda/driver-api.html) 介绍如何编写混合使用两种 API 的代码。

本指南假设已安装 CUDA Toolkit 和 NVIDIA Driver，并且系统中存在受支持的 NVIDIA GPU。安装所需 CUDA 组件的说明见 [CUDA Quickstart Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)。

## 2.1.1 使用 NVCC 编译（Compilation with NVCC）

使用 NVIDIA CUDA 编译器 `nvcc` 编译用 C++ 编写的 GPU 代码。`nvcc` 是一个编译器驱动程序，用于简化 C++ 或 PTX 代码的编译过程：它提供简单、熟悉的命令行选项，并通过调用实现不同编译阶段的一组工具来完成编译。

本指南中的 `nvcc` 命令行可以用于安装了 CUDA Toolkit 的 Linux 系统，也可以用于 Windows 命令行或 PowerShell，以及安装了 CUDA Toolkit 的 Windows Subsystem for Linux。指南中的 [NVCC 章节](./nvcc.html)介绍常见用法；完整文档见 [nvcc 用户手册](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)。

## 2.1.2 Kernel

正如 [CUDA 编程模型](../01-introduction/programming-model.html) 的引言所述，能够从 host 调用、并在 GPU 上执行的函数称为 **kernel**。kernel 被编写成由许多并行线程同时运行。

### 2.1.2.1 指定 Kernel（Specifying Kernels）

kernel 的代码使用 `__global__` 声明说明符指定。它告诉编译器：这个函数要编译为可以通过 kernel launch 调用的 GPU 函数。kernel launch 是启动 kernel 运行的操作，通常由 CPU 发起。kernel 函数的返回类型为 `void`。

```cpp
// Kernel definition
__global__ void vecAdd(float* A, float* B, float* C)
{

}
```

### 2.1.2.2 启动 Kernel（Launching Kernels）

kernel launch 的一部分会指定要并行执行该 kernel 的线程数量，这称为**执行配置（execution configuration）**。同一个 kernel 的不同次调用可以使用不同执行配置，例如不同的线程数或 thread block 数量。

从 CPU 代码启动 kernel 有两种方式：**三尖括号记法（triple chevron notation）**和 `cudaLaunchKernelEx`。这里介绍最常用的三尖括号记法；使用 `cudaLaunchKernelEx` 启动 kernel 的示例和详细讨论见[第 3.1.1 节](../03-advanced-cuda/advanced-apis-and-features.html)。

#### 2.1.2.2.1 三尖括号记法（Triple Chevron Notation）

三尖括号记法是 CUDA C++ 的一种[语言扩展](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)，用于启动 kernel。它使用三个尖括号字符 `<<< >>>` 包裹 kernel launch 的执行配置，所以称为“三尖括号”。执行配置参数以逗号分隔，写在尖括号内部，形式类似函数调用的参数。下面是启动 `vecAdd` kernel 的语法：

```cpp
__global__ void vecAdd(float* A, float* B, float* C)
{

}

int main()
{
    ...
    // Kernel invocation
    vecAdd<<<1, 256>>>(A, B, C);
    ...
}
```

三尖括号记法的前两个参数分别是 grid 维度和 thread block 维度。使用一维 thread block 或一维 grid 时，可以直接用整数指定维度。

上面的代码启动一个包含 256 个线程的 thread block。每个线程都会执行完全相同的 kernel 代码。在[线程与 Grid 索引内建函数](#2123-线程与-grid-索引内建函数-thread-and-grid-index-intrinsics)中，我们会展示每个线程如何利用自己在 thread block 和 grid 中的索引，改变它所操作的数据。

每个 block 的线程数存在上限，因为一个 block 中的所有线程都驻留在同一个流式多处理器（SM）上，并且必须共享该 SM 的资源。在当前 GPU 上，一个 thread block 最多可以包含 1024 个线程。如果资源允许，一个 SM 可以同时调度多个 thread block。

相对于调用它的 host 线程，kernel launch 是异步的。也就是说，kernel 会被设置为在 GPU 上执行，但 host 代码不会等待 kernel 在 GPU 上完成（甚至开始）后才继续执行。必须使用某种 CPU/GPU 同步机制，才能确定 kernel 已经完成。最基本的做法是让整个 GPU 完全同步，示例见[同步 CPU 与 GPU](#214-同步-cpu-与-gpu-synchronizing-cpu-and-gpu)。更复杂的同步方式见[异步执行](./asynchronous-execution.html)。

使用二维或三维 grid、thread block 时，要用 CUDA 类型 `dim3` 指定 grid 和 thread block 的维度。下面的代码片段启动一个 `MatAdd` kernel：grid 为 16×16 个 thread block，每个 thread block 为 8×8。

```cpp
int main()
{
    ...
    dim3 grid(16,16);
    dim3 block(8,8);
    MatAdd<<<grid, block>>>(A, B, C);
    ...
}
```

### 2.1.2.3 线程与 Grid 索引内建函数（Thread and Grid Index Intrinsics）

在 kernel 代码中，CUDA 提供了用于访问执行配置参数，以及线程或 block 索引的内建函数。

- `threadIdx` 给出线程在线程块内的索引。一个 thread block 中的每个线程都有不同的索引。
- `blockDim` 给出 thread block 的维度，也就是 kernel launch 执行配置中指定的维度。
- `blockIdx` 给出 thread block 在 grid 内的索引。每个 thread block 都有不同的索引。
- `gridDim` 给出 grid 的维度，也就是启动 kernel 时执行配置中指定的维度。

这些内建变量都是包含 `.x`、`.y`、`.z` 成员的三分量向量。启动配置没有指定的维度默认为 1。`threadIdx` 和 `blockIdx` 都从 0 开始索引，也就是说，`threadIdx.x` 的取值从 0 到 `blockDim.x-1`（含两端）；`.y` 和 `.z` 在各自维度上遵循同样规则。

同样，`blockIdx.x` 的取值从 0 到 `gridDim.x-1`（含两端），`.y` 和 `.z` 维度也遵循相同规则。

这些内建变量让单个线程能够确定自己应执行的工作。回到 `vecAdd` kernel：它接收三个参数，每个都是由浮点数组成的向量。kernel 对 `A` 和 `B` 执行逐元素相加，并把结果存储到 `C`。并行化方式是让每个线程执行一次加法；线程和 grid 索引决定该线程计算哪个元素。

```cpp
__global__ void vecAdd(float* A, float* B, float* C)
{
   // calculate which element this thread is responsible for computing
   int workIndex = threadIdx.x + blockDim.x * blockIdx.x

   // Perform computation
   C[workIndex] = A[workIndex] + B[workIndex];
}

int main()
{
    ...
    // A, B, and C are vectors of 1024 elements
    vecAdd<<<4, 256>>>(A, B, C);
    ...
}
```

在这个例子中，使用 4 个各含 256 个线程的 thread block，对包含 1024 个元素的向量求和。第一个 thread block 中，`blockIdx.x` 为 0，因此每个线程的 `workIndex` 就是它的 `threadIdx.x`。第二个 thread block 中，`blockIdx.x` 为 1，`blockDim.x * blockIdx.x` 就等于 `blockDim.x`，在本例中即 256；第二个 block 中每个线程的 `workIndex` 都是 `threadIdx.x + 256`。第三个 thread block 中，`workIndex` 为 `threadIdx.x + 512`。

这种计算 `workIndex` 的方式对于一维并行化非常常见。扩展到二维或三维时，通常在每个维度上重复相同的模式。

#### 2.1.2.3.1 边界检查（Bounds Checking）

前面的例子假定向量长度是 thread block 大小的倍数，本例中就是 256 的倍数。为了让 kernel 能处理任意向量长度，可以增加检查，确保内存访问不越过数组边界；然后启动一个包含部分非活跃线程的 thread block。

```cpp
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
     // calculate which element this thread is responsible for computing
     int workIndex = threadIdx.x + blockDim.x * blockIdx.x

     if(workIndex < vectorLength)
     {
         // Perform computation
         C[workIndex] = A[workIndex] + B[workIndex];
     }
}
```

有了上面的 kernel 代码，就可以启动多于实际需要的线程，而不会造成数组越界。当 `workIndex` 超过 `vectorLength` 时，线程退出并且不做任何工作。在 block 中启动一些不做工作的额外线程，开销并不大；但应避免启动完全没有线程执行工作的 thread block。现在这个 kernel 可以处理不是 block 大小倍数的向量长度。

所需 thread block 数量可以按以下方式计算：先把所需线程数（这里是向量长度）除以每个 block 的线程数，再向上取整。也就是对两者做整数除法并向上舍入。下面给出把这个过程表达成一次整数除法的常见写法。在整数除法前加上 `threads - 1`，就能实现类似 ceiling 的效果：只有当向量长度不能被每个 block 的线程数整除时，才增加一个 block。

```cpp
// vectorLength is an integer storing number of elements in the vector
int threads = 256;
int blocks = (vectorLength + threads-1)/threads;
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```

[CUDA Core Compute Library（CCCL）](https://nvidia.github.io/cccl/unstable/) 提供了方便的 `cuda::ceil_div` 工具，用于执行这个向上取整除法并计算 kernel launch 所需的 block 数量。包含头文件 `<cuda/cmath>` 后即可使用。

```cpp
// vectorLength is an integer storing number of elements in the vector
int threads = 256;
int blocks = cuda::ceil_div(vectorLength, threads);
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```

这里选择每个 block 256 个线程只是任意选择，但 256 往往是一个很好的起始值。

## 2.1.3 GPU 计算中的内存（Memory in GPU Computing）

要使用上面展示的 `vecAdd` kernel，数组 `A`、`B` 和 `C` 必须位于 GPU 可以访问的内存中。实现这一点有多种方法，本节会展示其中两种。其他方法将在后面的[统一内存](./unified-and-system-memory.html)章节中介绍。GPU 代码可访问的内存空间已经在[GPU 内存](../01-introduction/programming-model.html#123-gpu-内存-gpu-memory)中引入，并将在[GPU 设备内存空间](./writing-simt-kernels.html)中详细介绍。

### 2.1.3.1 统一内存（Unified Memory）

统一内存是 CUDA runtime 的一项功能，它让 NVIDIA Driver 管理 host 与 device 之间的数据移动。可以通过 `cudaMallocManaged` API 分配统一内存，也可以使用 `__managed__` 说明符声明变量。无论 GPU 还是 CPU 尝试访问该内存，NVIDIA Driver 都会确保相应处理器可以访问它。

下面的代码展示了一个完整函数：它使用统一内存为将在 GPU 上使用的输入、输出向量分配内存，并启动 `vecAdd` kernel。`cudaMallocManaged` 分配的缓冲区可以由 CPU 或 GPU 访问，最后使用 `cudaFree` 释放。

```cpp
void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

}
```

所有 CUDA 支持的操作系统和 GPU 都支持统一内存，但底层机制和性能可能根据系统架构而不同。[Unified Memory](./unified-and-system-memory.html) 会提供更多细节。在一些 Linux 系统上，例如具有[地址转换服务](./unified-and-system-memory.html)或[异构内存管理](./unified-and-system-memory.html)的系统，所有系统内存都会自动成为统一内存，不需要使用 `cudaMallocManaged` 或 `__managed__` 说明符。

### 2.1.3.2 显式内存管理（Explicit Memory Management）

显式管理不同内存空间之间的内存分配和数据迁移，可以帮助改善应用性能，但也会让代码更冗长。下面的代码使用 `cudaMalloc` 在 GPU 上显式分配内存；GPU 内存使用与上一个统一内存示例相同的 `cudaFree` API 释放。

```cpp
void explicitMemExample(int vectorLength)
{
    // Pointers for host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Pointers for device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    //Allocate Host Memory using cudaMallocHost API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // start-allocate-and-copy
    // Allocate memory on the GPU
    cudaMalloc(&devA, vectorLength*sizeof(float));
    cudaMalloc(&devB, vectorLength*sizeof(float));
    cudaMalloc(&devC, vectorLength*sizeof(float));

    // Copy data to the GPU
    cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength*sizeof(float));
    // end-allocate-and-copy

    // Launch the kernel
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    // wait for kernel execution to complete
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }

    // clean up
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}
```

CUDA API `cudaMemcpy` 用于把驻留在 CPU 的缓冲区复制到驻留在 GPU 的缓冲区。除了目标指针、源指针和字节数外，`cudaMemcpy` 的最后一个参数是 `cudaMemcpyKind_t`，可取的值包括：

- `cudaMemcpyHostToDevice`：从 CPU 复制到 GPU；
- `cudaMemcpyDeviceToHost`：从 GPU 复制到 CPU；
- `cudaMemcpyDeviceToDevice`：在同一 GPU 内或 GPU 之间复制。

本例把 `cudaMemcpyDefault` 作为 `cudaMemcpy` 的最后一个参数。CUDA 会根据源指针和目标指针的值确定要执行的复制类型。

`cudaMemcpy` API 是同步的，也就是说，复制完成前它不会返回。异步复制将在[在 CUDA stream 中启动内存传输](./asynchronous-execution.html)中介绍。

代码使用 `cudaMallocHost` 在 CPU 上分配内存。该 API 分配的是 host 上的[页锁定内存](./unified-and-system-memory.html)，它可以改善复制性能，也是执行[异步内存传输](./asynchronous-execution.html)的必要条件。一般来说，用于向 GPU 发送数据或从 GPU 接收数据的 CPU 缓冲区，最好使用页锁定内存。如果页锁定了过多 host 内存，一些系统上的性能可能下降。最佳实践是只对实际用于与 GPU 发送或接收数据的缓冲区进行页锁定。

### 2.1.3.3 内存管理与应用性能（Memory Management and Application Performance）

从上面的例子可以看出，显式内存管理更加冗长，因为程序员必须指定 host 与 device 之间的复制。显式内存管理的优点和缺点正是同一件事：它让程序员能够更精确地控制数据何时在 host 与 device 之间复制、内存驻留在哪里，以及具体分配哪一类内存。显式内存管理还可以通过控制内存传输并让传输与其他计算重叠，带来性能机会。

使用统一内存时，也有一些 CUDA API 可以为管理内存的 NVIDIA Driver 提供提示，这些 API 会在后面的[内存建议与预取](./unified-and-system-memory.html)中介绍；它们可以让统一内存获得部分显式内存管理的性能收益。

## 2.1.4 同步 CPU 与 GPU（Synchronizing CPU and GPU）

如[启动 Kernel](#2122-启动-kernel-launching-kernels)一节所述，相对于调用 kernel 的 CPU 线程，kernel launch 是异步的。这意味着 CPU 线程的控制流会在 kernel 完成之前继续执行，甚至可能在 kernel 启动之前就继续执行。为了保证 kernel 执行完成后 host 代码才继续运行，需要某种同步机制。

同步 GPU 与 host 线程最简单的方式是使用 `cudaDeviceSynchronize`。它会阻塞 host 线程，直到 GPU 上此前发出的所有工作都完成。本章示例只在 GPU 上执行单个操作，因此这种方式已经足够。在大型应用中，GPU 上可能有多个[stream](./asynchronous-execution.html)执行工作，而 `cudaDeviceSynchronize` 会等待所有 stream 中的工作完成。这类应用建议使用[Stream Synchronization](./asynchronous-execution.html) API，只与指定 stream 同步，或者使用 [CUDA Events](./asynchronous-execution.html)。这些内容会在[异步执行](./asynchronous-execution.html)章节中详细介绍。

## 2.1.5 综合示例（Putting it All Together）

下面的代码清单给出了本章介绍的简单向量加法 kernel 的完整代码，包括全部 host 代码和用于检查结果正确性的工具函数。示例默认向量长度为 1024，也可以通过可执行文件的命令行参数指定其他向量长度。

### 统一内存版本

```cpp
#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
}

int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    unifiedMemExample(vectorLength);
    return 0;
}
```

### 显式内存管理版本

```cpp
#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

void explicitMemExample(int vectorLength)
{
    // Pointers for host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Pointers for device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    //Allocate Host Memory using cudaMallocHost API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Allocate memory on the GPU
    cudaMalloc(&devA, vectorLength*sizeof(float));
    cudaMalloc(&devB, vectorLength*sizeof(float));
    cudaMalloc(&devC, vectorLength*sizeof(float));

    // Copy data to the GPU
    cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength*sizeof(float));

    // Launch the kernel
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    // wait for kernel execution to complete
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers do not match\n");
    }

    // clean up
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}

int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    explicitMemExample(vectorLength);
    return 0;
}
```

可以用下面的 `nvcc` 命令构建并运行两个示例：

```bash
$ nvcc vecAdd_unifiedMemory.cu -o vecAdd_unifiedMemory
$ ./vecAdd_unifiedMemory
Unified Memory: CPU and GPU answers match
$ ./vecAdd_unifiedMemory 4096
Unified Memory: CPU and GPU answers match
```

```bash
$ nvcc vecAdd_explicitMemory.cu -o vecAdd_explicitMemory
$ ./vecAdd_explicitMemory
Explicit Memory: CPU and GPU answers match
$ ./vecAdd_explicitMemory 4096
Explicit Memory: CPU and GPU answers match
```

在这些示例中，所有线程都执行相互独立的工作，不需要彼此协调或同步。但在很多情况下，线程需要与其他线程合作和通信才能完成工作。一个 block 内的线程可以通过[共享内存](./writing-simt-kernels.html)共享数据，并进行同步以协调内存访问。

block 级同步最基本的机制是 `__syncthreads()` 内建函数。它充当一个屏障：block 中所有线程都必须在屏障处等待，直到所有线程都到达后才允许任何线程继续。[Shared Memory](./writing-simt-kernels.html) 给出了使用 shared memory 的示例。

为了实现高效协作，shared memory 被设计为每个处理器核心附近的低延迟内存（很像 L1 cache），而 `__syncthreads()` 被设计为轻量操作。`__syncthreads()` 只同步单个 thread block 内的线程。

只有在特定情况下才支持 block 之间同步。例如，[thread block cluster](../01-introduction/programming-model.html#12221-线程块集群-thread-block-clusters) 允许 cluster 内的 block 同步；[Cooperative Groups](../04-cuda-features/cooperative-groups.html) API 提供了创建跨 block 同步域的机制。

通常，将同步限制在一个 thread block 内可以获得最佳性能。thread block 仍然可以使用[原子内存函数](./writing-simt-kernels.html)处理公共结果，后续章节会介绍这些函数。

[第 3.2.4 节](../03-advanced-cuda/advanced-kernel-programming.html)介绍 CUDA 同步原语，它们可以提供非常细粒度的控制，以最大化性能和资源利用率。

## 2.1.6 Runtime 初始化（Runtime Initialization）

CUDA runtime 会为系统中的每个设备创建一个 [CUDA context](../03-advanced-cuda/driver-api.html)。这个 context 是该设备的 primary context，并在该设备上第一次需要活动 context 的 runtime 函数调用时初始化。这个 context 由应用的所有 host 线程共享。

作为创建 context 的一部分，device code 必要时会执行[即时编译](../01-introduction/cuda-platform.html)，并被加载到 device memory。这些过程都会透明地发生。CUDA runtime 创建的 primary context 可以通过 Driver API 访问，从而实现互操作，详见 [Runtime API 与 Driver API 的互操作](../03-advanced-cuda/driver-api.html)。

从 CUDA 12.0 开始，调用 `cudaInitDevice` 和 `cudaSetDevice` 会初始化 runtime 以及与指定设备关联的 primary [context](../03-advanced-cuda/driver-api.html)。如果在这两个调用之前出现 runtime API 请求，runtime 会隐式使用设备 0，并按需自初始化来处理请求。这一点对于测量 runtime 函数调用时间，以及解释第一次 runtime 调用返回的错误代码非常重要。在 CUDA 12.0 之前，`cudaSetDevice` 不会初始化 runtime。

`cudaDeviceReset` 会销毁当前设备的 primary context。如果 primary context 被销毁后又调用 CUDA runtime API，该设备会重新创建一个 primary context。

::: tip 注意
CUDA 接口使用全局状态；该状态在 host 程序启动期间初始化，在 host 程序终止期间销毁。在程序启动或终止阶段（`main` 之后）隐式或显式使用这些接口，会导致未定义行为。

从 CUDA 12.0 开始，如果 runtime 尚未初始化，`cudaSetDevice` 会在改变 host 线程的当前设备后显式初始化 runtime。之前的版本会把新设备上的 runtime 初始化延迟到 `cudaSetDevice` 之后的第一次 runtime 调用。因此，检查 `cudaSetDevice` 的返回值以发现初始化错误非常重要。

参考手册中错误处理和版本管理章节的 runtime 函数不会初始化 runtime。
:::

## 2.1.7 CUDA 中的错误检查（Error Checking in CUDA）

每个 CUDA API 都会返回枚举类型 `cudaError_t` 的值。示例代码中经常省略错误检查；在生产应用中，最佳实践是始终检查并管理每次 CUDA API 调用的返回值。没有错误时，返回值为 `cudaSuccess`。许多应用会实现如下的工具宏：

```cpp
#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)
```

这个宏使用 `cudaGetErrorString` API，返回描述某个 `cudaError_t` 值含义的可读字符串。使用上面的宏时，应用可以把 CUDA runtime API 调用包在 `CUDA_CHECK(expression)` 宏中：

```cpp
CUDA_CHECK(cudaMalloc(&devA, vectorLength*sizeof(float)));
CUDA_CHECK(cudaMalloc(&devB, vectorLength*sizeof(float)));
CUDA_CHECK(cudaMalloc(&devC, vectorLength*sizeof(float)));
```

如果这些调用中的任意一个发现错误，宏就会把错误打印到 `stderr`。这个宏适合较小的项目；在更大的应用中，可以把它改造成日志系统或其他错误处理机制。

::: tip 注意
任何 CUDA API 调用返回的错误状态，也可能表示此前发出的异步操作发生了错误。[异步错误处理](./asynchronous-execution.html)会详细介绍这一点。
:::

### 2.1.7.1 错误状态（Error State）

CUDA runtime 为每个 host 线程维护一个 `cudaError_t` 状态。它默认是 `cudaSuccess`，发生错误时会被覆盖。`cudaGetLastError` 返回当前错误状态，然后把状态重置为 `cudaSuccess`；另一种选择是 `cudaPeekAtLastError`，它返回错误状态但不重置它。

使用[三尖括号记法](#21221-三尖括号记法-triple-chevron-notation)启动 kernel 时不会返回 `cudaError_t`。一个好的实践是在 kernel launch 后立即检查错误状态，以发现 kernel launch 的即时错误，或 kernel launch 之前已经存在的[异步错误](#2172-异步错误-asynchronous-errors)。在 kernel launch 后立即检查错误状态得到 `cudaSuccess`，并不意味着 kernel 已经成功执行，甚至不意味着 kernel 已经开始执行。它只能证明传给 runtime 的 kernel launch 参数和执行配置没有触发错误，并且在 kernel 开始前错误状态中没有遗留的先前错误或异步错误。

### 2.1.7.2 异步错误（Asynchronous Errors）

CUDA kernel launch 和许多 runtime API 都是异步的。CUDA 异步 runtime API 会在[异步执行](./asynchronous-execution.html)中详细讨论。CUDA 错误状态在每次发生错误时被设置并覆盖，因此异步操作执行期间发生的错误，只有在下一次检查错误状态时才会报告。这个检查可以是 `cudaGetLastError`、`cudaPeekAtLastError`，也可以是任何返回 `cudaError_t` 的 CUDA API 调用。

CUDA runtime API 函数返回错误时，不会清除错误状态。这意味着异步错误（例如 kernel 的非法内存访问）的错误代码会被每一个 CUDA runtime API 返回，直到调用 `cudaGetLastError` 清除错误状态。

```cpp
vecAdd<<<blocks, threads>>>(devA, devB, devC);
// check error state after kernel launch
CUDA_CHECK(cudaGetLastError());
// wait for kernel execution to complete
// The CUDA_CHECK will report errors that occurred during execution of the kernel
CUDA_CHECK(cudaDeviceSynchronize());
```

::: tip 注意
`cudaStreamQuery` 和 `cudaEventQuery` 可能返回的 `cudaError_t` 值 `cudaErrorNotReady` 不被视为错误，因此 `cudaPeekAtLastError` 或 `cudaGetLastError` 不会报告它。
:::

### 2.1.7.3 `CUDA_LOG_FILE`

识别 CUDA 错误的另一种好方法是使用 `CUDA_LOG_FILE` 环境变量。设置该环境变量后，CUDA driver 会把遇到的错误消息写入该变量指定路径的文件。

下面是不正确的 CUDA 代码：它尝试启动一个大于所有架构支持的最大 block 的 thread block。

```cpp
__global__ void k()
{ }

int main()
{
        k<<<8192, 4096>>>(); // Invalid block size
        CUDA_CHECK(cudaGetLastError());
        return 0;
}
```

构建并运行这段代码时，kernel launch 后的检查会使用[第 2.1.7 节](#217-cuda-中的错误检查-error-checking-in-cuda)中的宏检测并报告错误：

```bash
$ nvcc errorLogIllustration.cu -o errlog
$ ./errlog
CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument
```

但是，如果运行应用时把 `CUDA_LOG_FILE` 设置为文本文件，该文件会包含更多错误信息：

```bash
$ env CUDA_LOG_FILE=cudaLog.txt ./errlog
CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument
$ cat cudaLog.txt
[12:46:23.854][137216133754880][CUDA][E] One or more of block dimensions of (4096,1,1) exceeds corresponding maximum value of (1024,1024,64)
[12:46:23.854][137216133754880][CUDA][E] Returning 1 (CUDA_ERROR_INVALID_VALUE) from cuLaunchKernel
```

把 `CUDA_LOG_FILE` 设为 `stdout` 或 `stderr`，可以分别打印到标准输出和标准错误。即使应用没有正确检查 CUDA 返回值，使用 `CUDA_LOG_FILE` 也可以捕获并识别 CUDA 错误。这种方法对调试非常有力，但环境变量本身不能让应用在运行时处理和恢复 CUDA 错误。CUDA 的[错误日志管理](../04-cuda-features/error-log-management.html)功能还允许向 driver 注册回调函数，在检测到错误时调用；这可用于在运行时捕获和处理错误，也可以把 CUDA 错误日志无缝接入应用已有的日志系统。

[错误日志管理](../04-cuda-features/error-log-management.html)一节给出了更多示例。错误日志管理和 `CUDA_LOG_FILE` 需要 NVIDIA Driver `r570` 或更高版本。

## 2.1.8 Device 函数与 Host 函数（Device and Host Functions）

`__global__` 说明符表示 kernel 的入口点，也就是将在 GPU 上并行执行的函数。kernel 最常见的启动位置是 host，但也可以使用[动态并行](../04-cuda-features/cuda-dynamic-parallelism.html)，在另一个 kernel 内部启动 kernel。

`__device__` 说明符表示函数应编译为 GPU 函数，并且可由其他 `__device__` 或 `__global__` 函数调用。函数（包括类成员函数、函数对象和 lambda）可以同时指定为 `__device__` 和 `__host__`。

## 2.1.9 变量说明符（Variable Specifiers）

CUDA [说明符](../05-technical-appendices/cpp-language-extensions.html)可以用于静态变量声明，以控制变量的存放位置：

- `__device__`：变量存储在 [Global Memory](./writing-simt-kernels.html) 中；
- `__constant__`：变量存储在 [Constant Memory](./writing-simt-kernels.html) 中；
- `__managed__`：变量存储为 [Unified Memory](./unified-and-system-memory.html)；
- `__shared__`：变量存储在 [Shared Memory](./writing-simt-kernels.html) 中。

在 `__device__` 或 `__global__` 函数内部声明、但没有使用说明符的变量，在可能时会分配到寄存器；必要时会分配到[局部内存](./writing-simt-kernels.html)。在 `__device__` 或 `__global__` 函数外声明、且没有使用说明符的变量，会分配到系统内存。

### 2.1.9.1 检测 Device 编译（Detecting Device Compilation）

当函数同时指定 `__host__` 和 `__device__` 时，编译器会为该函数生成 GPU 代码和 CPU 代码。在这样的函数中，可以使用预处理器，为 GPU 版本或 CPU 版本指定不同代码。最常用的做法是检查是否定义了 `__CUDA_ARCH_`，下面的示例说明了这一点。

## 2.1.10 Thread Block Cluster

从计算能力 9.0 开始，CUDA 编程模型包含一个可选的层次：由 thread block 组成的 thread block cluster。类似于 thread block 内线程被保证共同调度到一个流式多处理器上，cluster 内 thread block 也被保证共同调度到 GPU 的图形处理集群（GPC）上。

和 thread block 类似，cluster 也会组织成一维、二维或三维的 thread block cluster grid，见[图 5](../figures.html#图-5-线程块集群)。

cluster 中 thread block 的数量由用户定义。在 CUDA 中，可移植的 cluster size 最大支持 8 个 thread block。需要注意的是，如果 GPU 硬件或 MIG 配置太小，无法支持 8 个多处理器，那么最大 cluster size 会相应减小。识别这些更小的配置，以及支持超过 8 个 thread block cluster size 的更大配置，是架构相关的；可以通过 `cudaOccupancyMaxPotentialClusterSize` API 查询。

cluster 中的所有 thread block 都被保证同时调度到单个 GPU Processing Cluster（GPC）上执行，并且可以使用 [Cooperative Groups](../04-cuda-features/cooperative-groups.html) API 通过硬件支持的 `cluster.sync()` 执行同步。Cluster group 还提供 `num_threads()` 和 `num_blocks()` API，分别用于查询 cluster group 中的线程数和 block 数；也可以用 `dim_threads()` 和 `dim_blocks()` API 分别查询线程或 block 在 cluster group 中的 rank。

属于 cluster 的 thread block 可以访问 **distributed shared memory**，也就是 cluster 中所有 thread block 的 shared memory 合并后的空间。cluster 内 thread block 可以读写 distributed shared memory 中的任意地址，也可以对任意地址执行原子操作。[Distributed Shared Memory](./writing-simt-kernels.html) 给出了使用 distributed shared memory 构建直方图的示例。

::: tip 注意
在启用了 cluster 支持的 kernel 中，为保持兼容性，`gridDim` 变量仍然以 thread block 数量表示 grid 的大小。block 在 cluster 中的 rank 可以通过 [Cooperative Groups](../04-cuda-features/cooperative-groups.html) API 获取。
:::

### 2.1.10.1 使用三尖括号记法启动 Cluster（Launching with Clusters in Triple Chevron Notation）

可以通过编译期 kernel 属性 `__cluster_dims__(X,Y,Z)`，或者 CUDA kernel launch API `cudaLaunchKernelEx`，为 kernel 启用 thread block cluster。下面的示例使用编译期 kernel 属性启动 cluster。

使用 kernel 属性指定的 cluster size 在编译期固定，然后可以使用传统的 `<<< >>>` 启动 kernel。如果 kernel 使用编译期 cluster size，那么 launch 时不能修改 cluster size。

```cpp
// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    // Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```

## 本节导航

- [2.2 CUDA Python 入门](./intro-to-cuda-python.html)
- [2.3 编写 SIMT Kernel](./writing-simt-kernels.html)
- [2.5 异步执行](./asynchronous-execution.html)
- [2.6 统一内存与系统内存](./unified-and-system-memory.html)
- [2.7 NVCC](./nvcc.html)
