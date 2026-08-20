---
title: '2.3 编写 SIMT Kernel（Writing SIMT Kernels）'
description: 完整翻译 NVIDIA CUDA Programming Guide 第 2.3 节 SIMT kernel 编写、内存和占用率
---

# 2.3 编写 SIMT Kernel（Writing SIMT Kernels）

对于给定问题，CUDA kernel 的编写方式在很大程度上与传统 CPU 代码相同。不过，GPU 具有一些可以用于提升性能的独特特性。此外，理解 GPU 上的线程如何调度、如何访问内存以及执行如何推进，有助于开发者编写出能够最大化利用可用计算资源的 kernel。

本节将进一步介绍如何使用 [SIMT 编程模型](../01-introduction/programming-model.html#编程模型中的-warp与-simt)编写 kernel，示例同时覆盖 C++ 和 Python。

## 2.3.1 SIMT 基础（Basics of SIMT）

从开发者的角度看，CUDA thread 是并行性的基本单位。[1.2.2.2 节](../01-introduction/programming-model.html#编程模型中的-warp与-simt)介绍了 GPU 执行的基本 SIMT 模型，[SIMT 执行模型](../03-advanced-cuda/advanced-kernel-programming.html#3221-simt-执行模型-simt-execution-model)则提供了更多细节。SIMT 模型允许每个线程维护自己的状态和控制流。从功能上说，每个线程都可以执行单独的代码路径。然而，如果注意尽量减少同一个 warp 中的线程采取分歧代码路径的情况，就可以获得显著的性能提升。

## 2.3.2 线程层次结构（Thread Hierarchy）

线程被组织成 thread block，thread block 再组织成 grid。Grid 可以是一维、二维或三维的；在 kernel 内部，可以通过内置变量 `gridDim` 查询 grid 的大小。Thread block 同样可以是一维、二维或三维的；可以通过内置变量 `blockDim` 查询 thread block 的大小；通过内置变量 `blockIdx` 查询 thread block 的索引；在 thread block 内部，通过内置变量 `threadIdx` 获得线程索引。这些内置变量用于为每个线程计算唯一的全局线程索引，使每个线程能够按需从 global memory 加载/存储特定数据并执行独特的代码路径。

### C++ 内置变量

- `gridDim.[x|y|z]`：分别表示 grid 在 `x`、`y`、`z` 维度上的大小。对于所有线程，这些值都相同，并且属于 kernel 启动时设置的执行配置。
- `blockDim.[x|y|z]`：分别表示 block 在 `x`、`y`、`z` 维度上的大小。对于所有线程，这些值都相同，并且属于 kernel 启动时设置的执行配置。
- `blockIdx.[x|y|z]`：分别表示 block 在 `x`、`y`、`z` 维度上的索引。不同线程的这些值可能不同，用于指示当前正在执行的是哪个 thread block。
- `threadIdx.[x|y|z]`：分别表示线程在 `x`、`y`、`z` 维度上的索引。不同线程的这些值不同，用于指示当前正在执行的是哪个线程。

### Python 内置变量

- `cuda.threadIdx.[xyz]`：分别表示 grid 在 `x`、`y`、`z` 维度上的大小。这些值对于所有线程都相同，并属于 kernel 启动时设置的执行配置。
- `cuda.blockDim.[xyz]`：分别表示 block 在 `x`、`y`、`z` 维度上的大小。这些值对于所有线程都相同，并属于 kernel 启动时设置的执行配置。
- `cuda.blockIdx.[xyz]`：分别表示 block 在 `x`、`y`、`z` 维度上的索引。这些值对于不同线程可能不同，用于指示当前正在执行的是哪个 thread block。
- `cuda.gridDim.[xyz]`：分别表示线程在 `x`、`y`、`z` 维度上的索引。这些值对于不同线程不同，用于指示当前正在执行的是哪个线程。

使用多维 thread block 和 grid 只是为了方便，并不会影响性能。一个 block 中的线程会以可预测的方式线性化：第一个索引 `x` 变化最快，其次是 `y`，最后是 `z`。这意味着在线程索引的线性化结果中，连续的 `threadIdx.x` 值表示连续的线程，`threadIdx.y` 的步长是 `blockDim.x`，`threadIdx.z` 的步长是 `blockDim.x * blockDim.y`。这会影响线程如何分配给 warp，详见[硬件多线程](../03-advanced-cuda/advanced-kernel-programming.html#3222-硬件多线程-hardware-multithreading)。

下图展示了一个简单的二维 grid 示例，其中 thread block 是一维的。

![图 11：Thread Block Grid](/images/chapter-02/grid-of-thread-blocks.png)

图 11 说明：整个 grid 由多个 thread block 组成；每个 block 内的线程可以通过 block 与 thread 的索引计算自己负责的全局数据位置。示例采用二维 grid，因此 block 的位置沿两个方向排列，而 block 内部仍是一维线程序列。

### 2.3.2.1 Thread Block 同步（Thread Block Synchronization）

前面展示的示例不需要同步 thread block 内的线程。当同一个 thread block 中的线程协作或访问相同的内存地址时，尤其是在下面介绍的 [shared memory](#2332-shared-memory) 中访问相同地址时，就必须进行同步，以避免竞争条件和内存危害。

block 内最基本的同步形式称为 `syncthreads`。

## 2.3.3 GPU Device Memory 空间（GPU Device Memory Spaces）

CUDA device 具有多个内存空间，kernel 中的 CUDA thread 可以访问这些空间。下表总结了常见内存类型、它们的线程作用域和生命周期；后续小节会分别详细说明这些内存类型。

| 内存类型 | 作用域 | 生命周期 | 所在位置 |
| --- | --- | --- | --- |
| Global | Grid | Application | Device |
| Constant | Grid | Application | Device |
| Shared | Block | Kernel | SM |
| Local | Thread | Kernel | Device |
| Register | Thread | Kernel | SM |

表 1：内存类型、作用域和生命周期。

### 2.3.3.1 Global Memory

Global memory（也称 device memory）是存储 kernel 中所有线程都可以访问的数据的主要内存空间。它类似于 CPU 系统中的 RAM。运行在 GPU 上的 kernel 可以直接访问 global memory，就像运行在 CPU 上的代码可以访问系统内存一样。

Global memory 是持久的。也就是说，在 global memory 中进行的分配以及其中存储的数据会一直存在，直到释放该分配或应用程序终止；`cudaDeviceReset` 也会释放所有分配。

Global memory 可以通过 `cudaMalloc`、`cudaMallocManaged` 等 CUDA API 调用分配。可以使用 `cudaMemcpy` 等 CUDA Runtime API 调用把 CPU memory 中的数据复制到 global memory。通过 CUDA API 创建的 global memory 分配使用 `cudaFree` 释放。

在 kernel 启动之前，应用通过 CUDA API 调用分配并初始化 global memory。kernel 执行期间，CUDA thread 可以从 global memory 读取数据，也可以把 CUDA thread 执行计算得到的结果写回 global memory。kernel 执行完毕后，它写入 global memory 的结果可以复制回 host，或者被 GPU 上的其他 kernel 使用。

由于 grid 中的所有线程都可以访问 global memory，因此必须小心避免线程之间的数据竞争。由 host 启动的 CUDA kernel 返回类型为 `void`，因此 kernel 计算出的数值结果要返回给 host，唯一方式就是将结果写入 global memory。

下面的 kernel 展示了 global memory 的简单用法。三个数组 `A`、`B`、`C` 都位于 global memory，并由这个向量加法 kernel 访问。

#### C++

```cuda
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}
```

#### Python

```python
from numba import cuda

@cuda.jit
def vecadd(A, B, C):
    work_index = cuda.grid(1)
    C[work_index] = A[work_index] + B[work_index]
```

### 2.3.3.2 Shared Memory

Shared memory 是同一个 thread block 中所有线程都可以访问的内存空间。它物理上位于每个 SM 中，并与 L1 cache（统一数据缓存）使用同一物理资源。Shared memory 中的数据在整个 kernel 执行期间保持有效。可以把 shared memory 看作 kernel 执行期间由用户管理的 scratchpad。与 global memory 相比，shared memory 容量较小，但由于它位于每个 SM 上，其带宽更高、访问延迟更低。

由于同一个 thread block 中的所有线程都可以访问 shared memory，必须注意避免同一 thread block 内线程之间的数据竞争。C++ 使用 `__syncthreads()` 函数实现同一 thread block 内的同步；Python 使用 `cuda.syncthreads()` 实现相同的 thread block 同步。该函数会阻塞 block 中的所有线程，直到所有线程都到达 `__syncthreads()` 或 `cuda.syncthreads()` 调用处。

#### C++ 示例

```cuda
// 假设 blockDim.x 为 128
__global__ void example_syncthreads(int* input_data, int* output_data)
{
    __shared__ int shared_data[128];
    shared_data[threadIdx.x] = input_data[blockDim.x*blockIdx.x + threadIdx.x];

    // 所有线程同步，保证对 shared_data 的所有写入都排在
    // 任何线程从 __syncthreads() 解除阻塞之前：
    __syncthreads();

    // 一个线程安全地读取 shared_data：
    if (threadIdx.x == 0) {
        float sum = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += shared_data[i];
        }
        output_data[blockIdx.x] = sum;
    }
}
```

#### Python 示例

```python
import numpy as np
from numba import cuda
import cupy as cp
```

```python
@cuda.jit
def example_syncthreads(input_data, output_data):
    shared_data = cuda.shared.array(shape=128, dtype=np.int32)

    shared_data[cuda.threadIdx.x] = input_data[cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x]
    cuda.syncthreads()

    if cuda.threadIdx.x == 0:
        sum = 0.0
        for x in shared_data:
            sum = sum + x
        output_data[cuda.blockIdx.x] = sum
```

Shared memory 的大小取决于所使用的 GPU 架构。由于 shared memory 与 L1 cache 共享同一物理空间，使用 shared memory 会减少某个 kernel 可使用的 L1 cache 大小。如果 kernel 不使用 shared memory，则整个物理空间都会被 L1 cache 使用。CUDA Runtime API 提供了按每个 SM 和每个 thread block 查询 shared memory 大小的能力：可以使用 `cudaGetDeviceProperties` 函数，并检查设备属性 `cudaDeviceProp.sharedMemPerMultiprocessor` 和 `cudaDeviceProp.sharedMemPerBlock`。

CUDA Runtime API 提供 `cudaFuncSetCacheConfig` 函数，让开发者告知 runtime 是希望为 shared memory 分配更多空间，还是为 L1 cache 分配更多空间。该函数向 runtime 指定的是偏好，并不保证一定会遵守；runtime 可以根据可用资源和 kernel 需求自行决策。

Shared memory 既可以静态分配，也可以动态分配。

#### 2.3.3.2.1 静态分配 Shared Memory（Static Allocation of Shared Memory）

要静态分配 shared memory，C++ 中必须在 kernel 内使用 `__shared__` 修饰符声明变量，Python 中使用 `cuda.shared.array()`。该数组会分配在 shared memory 中，并在 kernel 执行期间一直存在。以这种方式声明的 shared memory 大小必须在编译时确定。例如，下面位于 kernel 体内的代码声明了一个包含 1024 个 `float` 元素的 shared memory 数组。

```cuda
__shared__ float sharedArray[1024];
```

```python
from numba import cuda
import numpy as np

shared_array = cuda.shared.array(shape=1024, dtype=np.float32)
```

声明之后，thread block 中的所有线程都可以访问这个 shared memory 数组。

#### 2.3.3.2.2 动态分配 Shared Memory（Dynamic Allocation of Shared Memory）

在 C++ 中，可以在三尖括号表示法的 kernel 启动中，将每个 thread block 所需的 shared memory 字节数作为第三个（可选）参数传入：

```cpp
functionName<<<grid, block, sharedMemoryBytes>>>();
```

如果没有指定这个参数，其默认值为 0。

在 Python 中，必须使用 `cuda.core.launch()` 启动 kernel。`LaunchConfig` 参数接收一个 `cuda.core.LaunchConfig` 对象；该对象有一个名为 `shmem_size` 的字段，作用与 C++ 三尖括号表示法中的第三个参数相同。

在 kernel 内部，C++ 可以使用带空 `[]` 的 `extern __shared__` 修饰符声明一个在 kernel 启动时动态分配的变量。Python 中仍然使用静态分配时的 `cuda.shared.array` 方法，但将 `shape` 参数设为 0。

```cuda
extern __shared__ float sharedArray[];
```

```python
from numba import cuda
import numpy as np

# shape=0 表示该数组的大小由 kernel 启动时的执行配置动态决定
shared_array = cuda.shared.array(shape=0, type=np.float32)
```

需要注意，一个 kernel 只能拥有一个动态分配的 shared array。如果需要多个动态分配的 shared memory 数组，就必须分配一个足够大的动态 shared memory 数组，然后手动对其进行分区。例如，在 C++ 中，如果希望动态 shared memory 中具有下面这些数组：

```cpp
short array0[128];
float array1[64];
int   array2[256];
```

可以按如下方式声明和初始化：

```cpp
extern __shared__ float array[];

short* array0 = (short*)array;
float* array1 = (float*)&array0[128];
int*   array2 =   (int*)&array1[64];
```

指针必须按照其指向的类型对齐。例如，下面的代码不能工作，因为 `array1` 没有按 4 字节对齐：

```cpp
extern __shared__ float array[];
short* array0 = (short*)array;
float* array1 = (float*)&array0[127];
```

Python 中没有指针，因此不支持这种类型双关形式。

### 2.3.3.3 Registers

Register 位于 SM 上，作用域是单个线程。Register 的使用由编译器管理；kernel 执行期间，register 用于线程本地存储。可以使用 GPU 的设备属性 `regsPerMultiprocessor` 和 `regsPerBlock` 查询每个 SM 以及每个 thread block 的 register 数量。

编译 C++ 代码时，NVCC 允许开发者通过 `-maxrregcount` 选项[指定 kernel 可以使用的最大 register 数量](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#maxrregcount-amount-maxrregcount)。减少 kernel 可使用的 register 数量，可能使更多 thread block 能够同时调度到 SM 上；但也可能造成更多 register spilling。Register spilling 是指：当前保存在芯片上 register 中的值必须写出到 global memory，之后再读回，以便为其他值腾出空间。

### 2.3.3.4 Local Memory

Local memory 是类似 register 的线程本地存储，由 NVCC 管理，但它的物理位置在 global memory 空间中。“local”描述的是逻辑作用域，而不是物理位置。kernel 执行期间，local memory 用于线程本地存储。编译器可能放入 local memory 的自动变量包括：

- 编译器无法判断其索引是否为常量的数组；
- 会消耗过多 register 空间的大型结构或数组；
- 当 kernel 使用的 register 超过可用数量时，任何因 register spilling 而产生的变量。

由于 local memory 位于 device memory，其访问延迟和带宽与 global memory 相同，也受到[合并 global memory 访问](#2341-合并-global-memory-访问coalesced-global-memory-access)中所述的内存合并要求约束。不过，local memory 的组织方式使得连续的 32 位字由连续的线程 ID 访问。因此，只要一个 warp 中的所有线程访问相同的相对地址（例如数组变量中的相同索引，或结构体变量中的相同成员），访问就是完全合并的。

### 2.3.3.5 Constant Memory

Constant memory 的作用域是 grid，在整个应用程序生命周期内可访问。Constant memory 位于 device 上，并且 kernel 只能读取它。

C++ 中，在任何 kernel 或 function 之外使用 `__constant__` 修饰符声明 variable 或 array。

Python 中，在 kernel code **内部**使用 `const_array = numba.cuda.const.array_like(ary)` 创建 constant memory array；该数组包含 host array `ary` 中存储的数据。

Constant memory 意味着一个变量：

- 位于 constant memory 空间；
- 每个 device 都有一个独立对象；
- grid 中的所有线程都可以访问它；在 C++ 中，host 也可以通过 runtime library 访问它（`cudaGetSymbolAddress()`、`cudaGetSymbolSize()`、`cudaMemcpyToSymbol()`、`cudaMemcpyFromSymbol()`）。

C++ 中，constant memory 的生命周期与创建它的 context 相同。Python 中，constant memory 的生命周期与声明它的 kernel 相同。

可以通过设备属性元素 `totalConstMem` 查询 constant memory 的总大小。

Constant memory 适合存储少量、每个线程都会以只读方式使用的数据。与其他 memory 相比，constant memory 很小，通常每个 device 只有 64 KB。

下面是声明和使用 constant memory 的示例。

#### C++

```cpp
// 在 .cu 文件中
__constant__ float coeffs[4];

__global__ void compute(float *out) {
    int idx = threadIdx.x;
    out[idx] = coeffs[0] * idx + coeffs[1];
}

// 在 host code 中
float h_coeffs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
cudaMemcpyToSymbol(coeffs, h_coeffs, sizeof(h_coeffs));
compute<<<1, 10>>>(device_out);
```

#### Python

```python
from numba import cuda
import numpy as np

host_array = np.zeros(128, dtype=np.float32)
# 用其他数据填充 host_array

@cuda.jit
def kernel(args):
    ...

    const_array = cuda.const.array_like(a)

    # 现在这次访问会经过 constant memory
    a = const_array[cuda.threadIdx.x]
```

### 2.3.3.6 Cache

GPU device 具有多级 cache 结构，其中包括 L2 cache 和 L1 cache。

L2 cache 位于 device 上，由所有 SM 共享。可以通过 `cudaGetDeviceProperties` 返回的设备属性元素 `l2CacheSize` 查询 L2 cache 的大小。

如上面的 [Shared Memory](#2332-shared-memory) 小节所述，L1 cache 物理上位于每个 SM 上，并且与 shared memory 使用同一物理空间。如果 kernel 不使用 shared memory，那么整个物理空间都会由 L1 cache 使用。

可以通过相关函数控制 L2 和 L1 cache，让开发者指定各种 cache 行为。这些函数的细节见[配置 L1/Shared Memory 平衡](../03-advanced-cuda/advanced-kernel-programming.html)、[L2 Cache 控制](../04-cuda-features/l2-cache-control.html#advanced-kernels-l2-control)和[低级 Load/Store 函数](../05-technical-appendices/cpp-language-extensions.html#low-level-load-store-functions)。

如果不使用这些提示，编译器和 runtime 会尽力高效地利用 cache。

### 2.3.3.7 Texture Memory 与 Surface Memory

> **注意**
>
> 一些较旧的 CUDA 代码可能会使用 texture memory，因为在较早的 NVIDIA GPU 上，这样做在某些场景中可以带来性能收益。对于目前支持的所有 GPU，这些场景都可以使用直接的 load 和 store 指令处理；对于非纹理 load，使用 texture 和 surface memory 指令已经不再带来性能收益。

GPU 可能具有专用指令，用于从要在 3D rendering 中作为纹理使用的图像加载数据。CUDA 通过 [texture object API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html) 和 [surface object API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__SURFACE__OBJECT.html) 暴露这些指令及其使用机制。

在目前支持的任何 NVIDIA GPU 上，对于非图形应用，texture memory 和 surface memory 都不会为 CUDA 提供性能优势。不过，当应用需要为 rendering 读取 texture 或 surface 数据时，这些 API 仍然有用，例如为使用 CUDA 作为 shader language 的 [NVIDIA OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) 编写 hit shader。

对于仍然使用这些 API 进行非纹理 load 的已有代码库，相关 API 的说明仍可在旧版 [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#texture-and-surface-memory) 中找到。

### 2.3.3.8 Distributed Shared Memory

> **注意**
>
> Distributed shared memory 是使用 [thread block cluster](../01-introduction/programming-model.html#编程模型中的-thread-block-clusters) 时可用的特性。Thread block cluster 使用 [cooperative_groups API](../04-cuda-features/cooperative-groups.html#cooperative-groups)；这些 API 当前仅在 C++ 中可用。

[Thread Block Cluster](../01-introduction/programming-model.html#编程模型中的-thread-block-clusters) 在 compute capability 9.0 中引入，并由 [Cooperative Groups](../04-cuda-features/cooperative-groups.html#cooperative-groups) 提供支持。它允许 thread block cluster 中的线程访问该 cluster 中所有参与 thread block 的 shared memory。这种分区后的 shared memory 称为 *Distributed Shared Memory*，对应的地址空间称为 Distributed Shared Memory address space。属于某个 thread block cluster 的线程可以在分布式地址空间中进行读、写或 atomic 操作，不论目标地址属于本地 thread block 还是远程 thread block。无论 kernel 是否使用 distributed shared memory，静态或动态的 shared memory 大小规格仍然是按每个 thread block 指定的。Distributed shared memory 的大小等于 cluster 中的 thread block 数量乘以每个 thread block 的 shared memory 大小。

访问 distributed shared memory 要求所有 thread block 都已经存在。用户可以使用 [class `cluster_group`](../05-technical-appendices/device-callable-apis.html#cg-api-cluster-group) 中的 `cluster.sync()`，保证所有 thread block 都已开始执行。用户还必须保证所有 distributed shared memory 操作都发生在 thread block 退出之前；例如，如果远程 thread block 正在尝试读取某个 thread block 的 shared memory，程序就必须保证远程 thread block 完成读取后，被读取的 thread block 才能退出。

下面以一个简单的 histogram 计算为例，说明如何使用 thread block cluster 在 GPU 上进行优化。计算 histogram 的标准方式是在每个 thread block 的 shared memory 中计算，然后执行 global memory atomic。该方法的限制是 shared memory 容量。一旦 histogram bin 不再适合 shared memory，用户就需要直接在 global memory 中计算 histogram，并因此直接在 global memory 中执行 atomic。Distributed shared memory 提供了一个中间步骤：根据 histogram bin 的大小，可以在 shared memory、distributed shared memory 中计算，或者直接在 global memory 中计算。

下面的 CUDA kernel 示例根据 histogram bin 数量，在 shared memory 或 distributed shared memory 中计算 histogram。

```cpp
#include <cooperative_groups.h>

// Distributed Shared Memory histogram kernel
__global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block, const int *__restrict__ input,
                                    size_t array_size)
{
    extern __shared__ int smem[];
    namespace cg = cooperative_groups;
    int tid = cg::this_grid().thread_rank();

    // 初始化 cluster、读取大小并计算本地 bin 偏移量。
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;

    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
    {
        smem[i] = 0; // 将 shared memory histogram 初始化为 0
    }

    // cluster synchronization 保证 cluster 中所有 thread block 的 shared memory
    // 都初始化为 0，也保证所有 thread block 都已经开始执行并同时存在。
    cluster.sync();

    for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
    {
        int ldata = input[i];

        // 找到正确的 histogram bin。
        int binid = ldata;
        if (ldata < 0)
            binid = 0;
        else if (ldata >= nbins)
            binid = nbins - 1;

        // 找到用于计算 distributed shared memory histogram 的
        // 目标 block rank 和偏移量。
        int dst_block_rank = (int)(binid / bins_per_block);
        int dst_offset = binid % bins_per_block;

        // 指向目标 block shared memory 的指针。
        int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

        // 对 histogram bin 执行 atomic 更新。
        atomicAdd(dst_smem + dst_offset, 1);
    }

    // cluster synchronization 是必需的，它保证所有 distributed shared
    // memory 操作都已完成，并且没有 thread block 在其他 thread block
    // 仍访问 distributed shared memory 时退出。
    cluster.sync();

    // 使用本地 distributed memory histogram 执行 global memory histogram。
    int *lbins = bins + cluster.block_rank() * bins_per_block;
    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
    {
        atomicAdd(&lbins[i], smem[i]);
    }
}
```

上面的 kernel 可以在运行时根据所需 distributed shared memory 的数量，以不同 cluster size 启动。如果 histogram 足够小，可以只放进一个 block 的 shared memory，此时可以使用 cluster size 1 启动 kernel。下面的代码根据 shared memory 需求动态启动 cluster kernel。

```cpp
// 通过可扩展启动方式启动
{
    cudaLaunchConfig_t config = {0};
    config.gridDim = array_size / threads_per_block;
    config.blockDim = threads_per_block;

    // cluster_size 取决于 histogram 大小。
    // cluster_size == 1 表示不使用 distributed shared memory，
    // 只使用 thread block 本地 shared memory。
    int cluster_size = 2; // 这里以大小 2 为例
    int nbins_per_block = nbins / cluster_size;

    // 动态 shared memory 大小按每个 block 计算。
    // distributed shared memory 大小 = cluster_size * nbins_per_block * sizeof(int)
    config.dynamicSmemBytes = nbins_per_block * sizeof(int);

    CUDA_CHECK(::cudaFuncSetAttribute((void *)clusterHist_kernel,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       config.dynamicSmemBytes));

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attribute;

    cudaLaunchKernelEx(&config, clusterHist_kernel, bins, nbins, nbins_per_block, input, array_size);
}
```

## 2.3.4 内存性能（Memory Performance）

确保正确使用内存，是实现高性能 CUDA kernel 的关键。本节讨论在 CUDA kernel 中提高 global memory 和 shared memory 吞吐量的一些一般原则和示例。对于大多数 kernel，global memory 性能是首要性能考量。线程处理它们没有显式加载或创建的数据时，通常会使用 shared memory，因此理解 shared memory 的性能特征同样重要。

下面的小节将通过逐步改进一个矩阵转置 kernel 中对 global memory 和 shared memory 的访问，说明内存访问的重要方面。

### 2.3.4.1 合并 Global Memory 访问（Coalesced Global Memory Access）

Global memory 通过 32 字节的内存事务访问。当 CUDA 线程从 global memory 请求一个数据字时，相关 warp 会把该 warp 中所有线程的内存请求合并为满足请求所需数量的内存事务；事务数量取决于每个线程访问的数据字大小，以及这些内存地址在各线程之间的分布。例如，如果一个线程请求 4 字节数据字，则该 warp 实际向 global memory 发出的内存事务总量为 32 字节。为了最高效地使用内存系统，一个内存事务获取的数据应该全部被 warp 使用。也就是说，如果一个线程从 global memory 请求 4 字节数据，而事务大小为 32 字节，那么该 warp 中其他线程能够使用这 32 字节请求中的其他 4 字节数据，就可以最有效地利用内存系统。

如果 warp 中的连续线程请求内存中连续的 4 字节数据字，则该 warp 总共请求 128 字节内存，而这 128 字节会通过四个 32 字节内存事务获取。这样，warp 中线程对内存事务的利用率为 100%。下图展示了这种完全合并的内存访问。

![图 12：合并内存访问](/images/chapter-02/perfect_coalescing_32byte_segments.png)

图 12 说明：一个 warp 的线程访问连续的数据字，硬件只需发出覆盖这些数据的少量 32 字节事务；传输的数据都被线程使用，因此事务利用率最高。

相反，病态的最坏情况是：连续线程访问的元素在内存中相距 32 字节或更多。此时，warp 必须为每个线程发出一个 32 字节内存事务，总内存流量为 `32 字节 × 32 线程/warp = 1024 字节`。但实际使用的内存只有 128 字节（warp 中每个线程 4 字节），所以内存利用率只有 `128 / 1024 = 12.5%`，这是对内存系统非常低效的使用。下图展示了这种未合并内存访问。

![图 13：未合并内存访问](/images/chapter-02/no_coalescing_32byte_segments.png)

图 13 说明：线程请求分散到多个 32 字节段中，每个事务中只有一个线程实际使用数据，因此大量传输带宽被浪费。

实现合并内存访问最直接的方式，是让连续线程访问内存中的连续元素。例如，对于使用一维 thread block 启动的 kernel，前面展示的向量加法 kernel 就能实现合并内存访问。注意，一个线程会访问三个数组；而连续的线程（由连续的 `workIndex` 值表示）访问数组中的连续元素。

```cuda
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}
```

```python
# 定义执行 C = A + B 向量加法的 CUDA kernel
@cuda.jit
def vecadd(A, B, C):
    work_index = cuda.grid(1)
    C[work_index] = A[work_index] + B[work_index]
```

连续线程访问连续内存元素并不是实现合并访问的必要条件，只是实现合并的典型方式。当 warp 中不同线程以某种线性或置换方式访问同一批 32 字节内存段中的元素时，就会发生合并访问。换一种说法，实现合并访问的最佳方式是最大化“使用的字节数 / 传输的字节数”这一比例。

理解 global memory 合并的另一种等价方式，是考虑一条由单个 warp 发出的 load 指令请求 32 个地址时，需要多少个 global memory 事务才能满足请求。在最理想的情况下，一个 global memory 事务就能满足所有 load。对于完全合并的 4 字节数据元素访问，需要 4 个 global memory 事务。在最坏情况下，一条来自单个 warp 的 load 指令请求的地址可能需要 32 个 global memory 事务才能满足。一般而言，满足一次 load 所需的 global memory 事务越少，性能越好。

> **注意**
>
> 确保 global memory 访问正确合并，是编写高性能 CUDA kernel 时最重要的性能考量之一。应用必须尽可能高效地使用内存系统。

#### 2.3.4.1.1 使用 Global Memory 的矩阵转置示例（Matrix Transpose Example Using Global Memory）

考虑一个简单的 out-of-place 矩阵转置 kernel：它将大小为 `N × N` 的 32 位 float 方阵从矩阵 `a` 转置到矩阵 `c`。该示例使用二维 grid，并假定启动二维 thread block，每个 block 包含 `32 × 32` 个线程，即 `blockDim.x = 32`、`blockDim.y = 32`；因此，每个二维 thread block 处理矩阵的一个 `32 × 32` tile。每个线程处理矩阵中的一个唯一元素，所以不需要显式同步线程。下图说明了这一矩阵转置操作，图后给出 kernel 源代码。

![图 14：使用 Global Memory 的矩阵转置](/images/chapter-02/global_transpose.png)

图 14 说明：矩阵上方和左侧的标签是二维 thread block 索引，也可以看作 tile 索引；每个小方格表示由一个二维 thread block 处理的矩阵 tile。本例 tile 大小为 `32 × 32`，因此每个小方格代表一个 `32 × 32` tile。绿色方格表示一个示例 tile 在转置前后的位置。

```cpp
/* 使用行主序的二维索引访问一维内存数组的宏 */
/* ld 是 leading dimension，即矩阵的列数 */

#define INDX( row, col, ld ) ( ( (row) * (ld) ) + (col) )

/* CUDA 矩阵转置 kernel */

__global__ void cuda_transpose(int m, float *a, float *c )
{
    int myCol = blockDim.x * blockIdx.x + threadIdx.x;
    int myRow = blockDim.y * blockIdx.y + threadIdx.y;

    if( myRow < m && myCol < m )
    {
        c[INDX( myCol, myRow, m )] = a[INDX( myRow, myCol, m )];
    } /* end if */
    return;
} /* end cuda_transpose */
```

```python
import numpy as np
from numba import cuda
import cupy as cp

# 矩阵转置 kernel：每个线程处理一个矩阵元素，
# 在二维 grid 上使用二维 thread block 以匹配矩阵大小
@cuda.jit
def transpose(a, c):
    col = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    row = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    c[(col,row)] = a[(row,col)]
```

要判断这个 kernel 是否实现了合并 global memory 访问，需要判断连续线程是否访问连续的内存元素。在二维 thread block 中，`x` 索引变化最快，因此连续的 `threadIdx.x` 值应该访问连续的内存元素。`threadIdx.x` 出现在 `myCol` 中；当 `myCol` 作为 `INDX` 宏的第二个参数时，可以看到连续线程正在读取 `a` 中连续的值，因此对 `a` 的读取是完全合并的。

但是，对 `c` 的写入不是合并的，因为连续的 `threadIdx.x` 值（再次检查 `myCol`）写入 `c` 中相互间隔 `ld`（leading dimension）个元素的位置。这是因为此时 `myCol` 是 `INDX` 宏的第一个参数，而第一个参数每增加 1，内存位置就变化 `ld`。当 `ld` 大于 32（也就是矩阵大小大于 32）时，这等价于图 13 所示的病态情况。

为了缓解这种未合并写入，可以使用 shared memory；下一小节将介绍这种方法。

### 2.3.4.2 Shared Memory 访问模式（Shared Memory Access Patterns）

Shared memory 有 32 个 bank，连续的 32 位数据字会映射到连续的 bank。每个 bank 每个时钟周期的带宽为 32 位。

当同一个 warp 中的多个线程尝试访问同一个 bank 中的不同元素时，就会发生 bank conflict。此时，对该 bank 中数据的访问会被串行化，直到所有请求该数据的线程都取得数据为止。这种访问串行化会带来性能惩罚。

有两个例外：同一个 warp 中的多个线程访问同一个 shared memory 位置（无论读取还是写入）。对于读访问，该数据字会广播给请求它的线程；对于写访问，每个 shared memory 地址只会由其中一个线程写入（究竟由哪个线程写入是不确定的）。

下面展示了一些步长访问示例。bank 内部的红色方框表示 shared memory 中的一个唯一位置。

![图 15：步长 Shared Memory 访问](/images/chapter-02/examples-of-strided-shared-memory-accesses.png)

图 15 说明：左图是步长为一个 32 位字的线性寻址，不产生 bank conflict；中图是步长为两个 32 位字的线性寻址，产生二路 bank conflict；右图是步长为三个 32 位字的线性寻址，不产生 bank conflict。

下面展示了一些涉及 broadcast 机制的内存读取示例。bank 内部的红色方框表示 shared memory 中的一个唯一位置；如果多条箭头指向同一个位置，数据就会广播给请求它的所有线程。

![图 16：不规则 Shared Memory 访问](/images/chapter-02/examples-of-irregular-shared-memory-accesses.png)

图 16 说明：左图通过随机置换实现无冲突访问；中图中线程 3、4、6、7、9 访问 bank 5 中的同一个字，因此无冲突；右图是冲突自由的 broadcast 访问，多个线程访问一个 bank 中的同一个字。

> **注意**
>
> 对于使用 shared memory 的高性能 CUDA kernel，避免 bank conflict 是重要的性能考量。

#### 2.3.4.2.1 使用 Shared Memory 的矩阵转置示例（Matrix Transpose Example Using Shared Memory）

在前面的[使用 Global Memory 的矩阵转置示例](#23411-使用-global-memory-的矩阵转置示例matrix-transpose-example-using-global-memory)中，矩阵转置功能正确，但对 global memory 的使用没有优化，因为矩阵 `c` 的写入没有正确合并。本例把 shared memory 当作由用户管理的 cache，用于暂存从 global memory 读取和写入的数据，从而让读取和写回都能实现合并 global memory 访问。

```cpp
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

/* 使用列主序的二维索引访问一维内存数组的宏 */
/* ld 是 leading dimension，即矩阵的行数 */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

/* 使用 shared memory 的 CUDA 矩阵转置 kernel */
__global__ void smem_transpose(int m,
                               float *a,
                               float *c )
{
    /* 声明静态分配的 shared memory 数组 */
    __shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y];

    /* 为边界检查代码确定我的行、列索引 */
    const int myRow = blockDim.x * blockIdx.x + threadIdx.x;
    const int myCol = blockDim.y * blockIdx.y + threadIdx.y;

    /* 确定我的行 tile 和列 tile 索引 */
    const int tileX = blockDim.x * blockIdx.x;
    const int tileY = blockDim.y * blockIdx.y;

    if( myRow < m && myCol < m )
    {
        /* 从 global memory 读取到 shared memory 数组 */
        smemArray[threadIdx.x][threadIdx.y] = a[INDX( tileX + threadIdx.x, tileY + threadIdx.y, m )];
    } /* end if */

    /* 同步 thread block 中的线程 */
    __syncthreads();

    if( myRow < m && myCol < m )
    {
        /* 将结果从 shared memory 写回 global memory */
        c[INDX( tileY + threadIdx.x, tileX + threadIdx.y, m )] = smemArray[threadIdx.y][threadIdx.x];
    } /* end if */
    return;
}
```

```python
import numpy as np
from numba import cuda
import cupy as cp

# 矩阵转置 kernel：每个线程处理一个矩阵元素，
# 在二维 grid 上使用二维 thread block 以匹配矩阵大小
# 将输入暂存到 shared memory
@cuda.jit
def smem_transpose(a, c):
    smemArray = cuda.shared.array(shape=(32, 32), dtype=np.float32)

    tile_col = cuda.blockDim.x * cuda.blockIdx.x
    tile_row = cuda.blockDim.y * cuda.blockIdx.y

    smemArray[(cuda.threadIdx.x, cuda.threadIdx.y)] = a[(tile_row + cuda.threadIdx.y, tile_col + cuda.threadIdx.x)]

    cuda.syncthreads()

    c[(tile_col + cuda.threadIdx.y, tile_row + cuda.threadIdx.x)] = smemArray[(cuda.threadIdx.y, cuda.threadIdx.x)]
```

这个示例展示的基本性能优化，是确保访问 global memory 时内存访问正确合并。在执行复制之前，每个线程都会计算自己的 `tileRow` 和 `tileCol` 索引。这些索引表示将要操作的具体 tile，并且由当前执行的 thread block 决定。同一个 thread block 中的每个线程具有相同的 `tileRow` 和 `tileCol` 值，因此可以把它们看作该 thread block 将要处理的 tile 的起始位置。

随后，每个 thread block 使用下面的语句将矩阵的一个 `32 × 32` tile 从 global memory 复制到 shared memory。由于一个 warp 有 32 个线程，这个复制操作会由 32 个 warp 执行，而且不同 warp 之间没有保证的执行顺序。

```cpp
smemArray[threadIdx.x][threadIdx.y] = a[INDX( tileRow + threadIdx.y, tileCol + threadIdx.x, m )];
```

```python
smemArray[(cuda.threadIdx.x, cuda.threadIdx.y)] = a[(tile_row + cuda.threadIdx.y, tile_col + cuda.threadIdx.x)]
```

由于 `threadIdx.x` 出现在 `INDX` 的第二个参数中，连续线程访问连续的内存元素，因此对 `a` 的读取是完全合并的。在 Python 中，将 `cuda.threadIdx.x` 作为 `a` 的索引元组中的最后一个索引具有相同效果，也就是访问 `a` 时实现完全合并。

kernel 的下一步是调用 `__syncthreads()` / `cuda.syncthreads()`。这保证 thread block 中的所有线程在继续之前完成前面代码的执行，从而保证把 `a` 写入 shared memory 的操作在下一步开始前完成。这一点非常重要，因为下一步会从 shared memory 读取数据。如果没有 `__syncthreads()` / `cuda.syncthreads()` 调用，就不能保证 thread block 中所有 warp 都已完成把 `a` 写入 shared memory，就可能有一些 warp 已经开始从 shared memory 数组读取数据。当线程处理或存储它没有加载的数据时，必须进行同步，以确保该元素的加载操作已经完成后再访问它。

此时，对于每个 thread block，shared memory 数组中都有一个与原矩阵顺序相同的 `32 × 32` tile。为了正确转置 tile 内的元素，从 `smemArray` 读取时需要交换 `threadIdx.x` 和 `threadIdx.y`。为了将整个 tile 放到 `c` 中的正确位置，写入 `c` 时也要交换 `tileRow` 和 `tileCol`。为确保访问合并，`INDX` 的第二个参数使用 `threadIdx.x`，如下面的语句所示；Python 再次通过在矩阵索引元组的最后一个索引中使用 `cuda.threadIdx.x` 实现同样效果。

```cpp
c[INDX( tileCol + threadIdx.y, tileRow + threadIdx.x, m )] = smemArray[threadIdx.y][threadIdx.x];
```

```python
c[(tile_col + cuda.threadIdx.y, tile_row + cuda.threadIdx.x)] = smemArray[(cuda.threadIdx.y, cuda.threadIdx.x)]
```

这个 kernel 展示了 shared memory 的两种常见用途：

- 使用 shared memory 暂存 global memory 中的数据，从而确保对 global memory 的读写都正确合并；
- 使用 shared memory 让同一个 thread block 中的线程共享数据。

#### 2.3.4.2.2 Shared Memory Bank Conflict

在[Shared Memory 访问模式](#2342-shared-memory-访问模式shared-memory-access-patterns)中介绍了 shared memory 的 bank 结构。前面的矩阵转置示例实现了对 global memory 的正确合并访问，但没有考虑 shared memory 是否存在 bank conflict。考虑下面的二维 shared memory 声明：

```cpp
__shared__ float smemArray[32][32];
```

```python
from numba import cuda
import numpy as np

smemArray = cuda.shared.array(shape=(32,32), dtype=np.float32)
```

假设 kernel 预期使用 `32 × 32` 个线程的二维 thread block 启动。由于一个 warp 有 32 个线程，某个 warp 中每个线程的 `threadIdx.y` 都是固定值，而 `threadIdx.x` 满足 `0 <= threadIdx.x < 32`。

下图左侧展示 warp 中线程访问 `smemArray` 一列数据的情况。Warp 0 访问 `smemArray[0][0]` 到 `smemArray[31][0]`（Python 中为 `smemArray[(0,0)]` 到 `smemArray[(31,0)]`）。在 C++ 和 Python 中，多维数组都是最后一个索引变化最快，因此 warp 0 中的连续线程访问相距 32 个元素的内存位置。如图所示，颜色表示 bank；warp 0 沿整列访问会产生 32 路 bank conflict。

下图右侧展示 warp 中线程跨一行访问 `smemArray` 的情况。Warp 0 访问 `smemArray[0][0]` 到 `smemArray[0][31]`（Python 中为 `smemArray[(0,0)]` 到 `smemArray[(0,31)]`）。此时，warp 0 中的连续线程访问相邻的内存位置。如图所示，颜色表示 bank；warp 0 横跨整行访问不会产生 bank conflict。理想情况是 warp 中每个线程访问不同颜色的 shared memory 位置。

![图 17：Shared Memory 中的 Bank 结构](/images/chapter-02/bank-conflicts-shared-mem.png)

图 17 说明：方框中的数字表示 warp 索引，颜色表示对应 shared memory 位置所属的 bank。

回到[使用 Shared Memory 的矩阵转置示例](#23421-使用-shared-memory-的矩阵转置示例matrix-transpose-example-using-shared-memory)，可以检查 shared memory 的使用以确定是否存在 bank conflict。第一次使用 shared memory 是将 global memory 中的数据写入 shared memory：

```cpp
smemArray[threadIdx.x][threadIdx.y] = a[INDX( tileRow + threadIdx.y, tileCol + threadIdx.x, m )];
```

```python
smemArray[(cuda.threadIdx.x, cuda.threadIdx.y)] = a[(tile_row + cuda.threadIdx.y, tile_col + cuda.threadIdx.x)]
```

由于数组以行主序存储，同一个 warp 中的连续线程（由连续的 `threadIdx.x` 值表示）以 32 个元素的步长访问 `smemArray`，因为 `threadIdx.x` 是数组的第一个索引。这会产生 32 路 bank conflict，对应图 17 左侧。

第二次使用 shared memory 是将数据从 shared memory 写回 global memory：

```cpp
c[INDX( tileCol + threadIdx.y, tileRow + threadIdx.x, m )] = smemArray[threadIdx.y][threadIdx.x];
```

```python
c[(tile_col + cuda.threadIdx.y, tile_row + cuda.threadIdx.x)] = smemArray[(cuda.threadIdx.y, cuda.threadIdx.x)]
```

此时，`threadIdx.x` 是 `smemArray` 数组的第二个索引，因此同一个 warp 中的连续线程以 1 个元素的步长访问 `smemArray`。这不会产生 bank conflict，对应图 17 右侧。

如图 17 所示的矩阵转置 kernel，一次 shared memory 访问没有 bank conflict，另一次访问产生 32 路 bank conflict。避免 bank conflict 的常见修复方式，是给 shared memory 数组的列维度增加 1 个元素进行填充：

```cpp
__shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y+1];
```

```python
smemArray = cuda.shared.array(shape=(32, 32 + 1), dtype=np.float32)
```

对 `smemArray` 声明做这个小调整，就可以消除 bank conflict。下图展示了大小为 `32 × 33` 的 shared memory 数组：无论同一 warp 的线程沿整列还是跨整行访问 shared memory 数组，bank conflict 都已消除，也就是同一 warp 中的线程访问不同颜色的位置。

![图 18：无 Bank Conflict 的 Shared Memory 结构](/images/chapter-02/no-bank-conflicts-shared-mem.png)

图 18 说明：方框中的数字表示 warp 索引，颜色表示对应 shared memory 位置所属的 bank。将列维度从 32 填充为 33 后，连续行的起始位置不再映射到同一个 bank。

## 2.3.5 Atomic 操作（Atomics）

高性能 CUDA kernel 依赖于尽可能多地表达算法并行性。GPU kernel 执行具有异步性，因此要求线程尽可能独立运行。不过，线程完全独立并不总是可能；正如 [Shared Memory](#2332-shared-memory) 中所述，CUDA 提供了让同一个 thread block 中的线程交换数据并同步的机制。

在整个 grid 的层面，没有同步 grid 中所有线程的机制。但可以使用 atomic function，对 global memory 位置提供同步访问。Atomic function 允许线程锁定一个 global memory 位置，并对该位置执行读-修改-写操作。在锁被持有期间，其他线程不能访问同一位置。

### 2.3.5.1 类似 C++ `std::atomic` 的 Atomic（C++ std::atomic-like Atomics）

在 C++ 中，CUDA 为名称相近的 C++ 标准库 atomic 提供了类似的语法和行为，包括 `cuda::std::atomic` 和 `cuda::std::atomic_ref`。CUDA 还提供扩展 C++ atomic：`cuda::atomic` 和 `cuda::atomic_ref`，允许用户指定 atomic 操作的[线程作用域](../03-advanced-cuda/advanced-kernel-programming.html#323-线程作用域-thread-scopes)。Atomic function 的细节见[Atomic Function](../05-technical-appendices/cpp-language-extensions.html#atomic-functions)。

下面是使用 `cuda::atomic_ref` 执行 device-wide atomic 加法的示例。这里 `array` 是 float 数组，`result` 是一个指向 global memory 位置的 float pointer，该位置用于保存数组元素之和。

```cpp
__global__ void sumReduction(int n, float *array, float *result) {
    ...
    tid = threadIdx.x + blockIdx.x * blockDim.x;

    cuda::atomic_ref<float, cuda::thread_scope_device> result_ref(result);
    result_ref.fetch_add(array[tid]);
    ...
}
```

应谨慎使用 atomic function，因为它们会强制线程同步，可能影响性能。

### 2.3.5.2 Python 中的 Memory Atomic

在 Python 中，atomic memory operation 由 `numba.cuda.atomic` 命名空间中提供给 GPU code 的函数实现。常见操作包括 `add`、`sub`、`max`、`min` 和 `compare_and_swap`。支持的 atomic operation 完整列表见 [Numba CUDA 文档](https://numba.pydata.org/numba-doc/dev/cuda/intrinsics.html)。

下面的代码展示了一个使用 atomic memory access 计算数组所有值之和的 kernel。每个 thread block 将数组的一部分加载到 shared memory；每个 thread block 中的一个线程计算本地和，并对结果数组 `s` 执行 atomic add。由于数据位于靠近 SM 计算资源的 shared memory 中，通常让一个线程执行求和仍具有合理的性能。

```python
import numpy as np
from numba import cuda
import cupy as cp

@cuda.jit
def sum_reduce(a, s):
    # 创建一个 shared array，支持最多 512 个线程的 block，
    # 虽然本例会使用更少的线程
    shared_staging = cuda.shared.array(shape=512, dtype=np.float32)

    # 将值加载到 shared memory，然后同步以确保所有加载完成
    shared_staging[cuda.threadIdx.x] = a[cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x]
    cuda.syncthreads()

    # 每个 block 只有线程 0 执行本地加法，
    # 随后每个 thread block 只执行一次 atomic operation
    local_sum = float(0.0)
    if cuda.threadIdx.x == 0:
        for i in range(cuda.blockDim.x):
            local_sum = local_sum + shared_staging[i]
        cuda.atomic.add(s, 0, local_sum)


array_length = 2**18

a = cp.ones(array_length)
s = cp.zeros(1, dtype=np.float32)

block_size = 256
grid_size = int(array_length/block_size)
sum_reduce[grid_size, block_size](a, s)

s_host = cp.asnumpy(s)
print(f"Sum is {int(s_host[0])}, expected {array_length}")
```

在这个简单示例中，输入数组全为 1，因此正确的和等于 `array_length`。

如果将

```python
cuda.atomic.add(s, 0, local_sum)
```

替换为非 atomic 加法：

```python
s[0] = s[0] + local_sum
```

对 `s[0]` 的访问就不是 atomic 的，`s[0]` 的最终值会小于 `array_length`。此外，该值可能随着每次运行而变化，也可能随着运行 GPU 的 SM 数量不同而变化。这说明，在这段代码中，atomic memory access 对保证正确性是必需的。

> **注意**
>
> 这个示例虽然功能正确，但并不是为了展示如何在 GPU 上编写峰值性能的 reduction operation。[CUDA Core Compute Libraries（CCCL）](https://github.com/nvidia/cccl) 提供了包括 reduction 在内的许多高性能 primitive。为了兼顾生产力和性能，开发者应尽可能优先使用这些经过高度调优的实现，而不是重新实现相同算法。这些 primitive 可通过 [`cuda.coop` package](https://nvidia.github.io/cccl/unstable/python/coop.html) 在 Python 中使用。

C++ 中也提供类似的 atomic，相关内容在[旧式 Atomic Function](../05-technical-appendices/cpp-language-extensions.html#legacy-atomic-functions)中讨论；不过，在 CUDA C++ 中推荐使用类似 `std::atomic` 的 atomic，并将其视为最佳实践。

## 2.3.6 Cooperative Groups

[Cooperative Groups](../04-cuda-features/cooperative-groups.html#cooperative-groups) 是 CUDA C++ 中提供的软件工具，允许应用程序定义能够彼此同步的线程组，即使该线程组跨越多个 thread block、单个 GPU 上的多个 grid，甚至跨越多个 GPU。一般的 CUDA 编程模型允许 thread block 内或 thread block cluster 内的线程高效同步，但不提供定义小于一个 thread block 或 cluster 的线程组的机制。同样，CUDA 编程模型不提供支持 thread block 之间同步的机制或保证。

Cooperative Groups 通过软件提供这两种能力。它允许应用程序创建跨越 thread block 和 cluster 边界的线程组，但这会带来语义限制和性能影响；详情见 [Cooperative Groups 特性章节](../04-cuda-features/cooperative-groups.html#cooperative-groups)。

## 2.3.7 Kernel 启动与 Occupancy（Kernel Launch and Occupancy）

启动 CUDA kernel 时，CUDA 会根据 kernel 启动时指定的执行配置，把线程组织成 thread block 和 grid。kernel 启动后，调度器将 thread block 分配给 SM。应用程序无法控制或查询哪些 thread block 被调度到哪些 SM，也不能依赖调度器提供的特定顺序或策略保证；因此，程序不能依赖具体的调度顺序或方案来保证正确执行。

一个 SM 能调度的 block 数量，取决于给定 thread block 所需的硬件资源以及该 SM 上可用的硬件资源。kernel 首次启动时，调度器开始为 SM 分配 thread block。只要 SM 仍有未被其他 thread block 占用的足够硬件资源，调度器就会继续向 SM 分配 thread block。如果某个时刻所有 SM 都没有容量接受新的 thread block，调度器就会等待，直到 SM 完成之前分配的 thread block。完成后，SM 可以接受更多工作，调度器会向它们分配新的 thread block。这个过程持续到所有 thread block 都已调度并执行完成。

`cudaGetDeviceProperties` 函数允许应用程序通过[设备属性](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp)查询每个 SM 的限制。注意，存在按 SM 计算的限制，也存在按 thread block 计算的限制：

- `maxBlocksPerMultiProcessor`：每个 SM 的最大 resident block 数量；
- `sharedMemPerMultiprocessor`：每个 SM 可用的 shared memory 字节数；
- `regsPerMultiprocessor`：每个 SM 可用的 32 位 register 数量；
- `maxThreadsPerMultiProcessor`：每个 SM 的最大 resident thread 数量；
- `sharedMemPerBlock`：一个 thread block 可分配的最大 shared memory 字节数；
- `regsPerBlock`：一个 thread block 可分配的最大 32 位 register 数量；
- `maxThreadsPerBlock`：每个 thread block 的最大线程数。

CUDA kernel 的 occupancy 是 active warp 数量与 SM 支持的最大 active warp 数量之比。一般来说，较高的 occupancy 是好的实践，因为它可以隐藏延迟并提高性能。

要计算 occupancy，需要知道前面描述的 SM 资源限制，以及目标 CUDA kernel 所需的资源。要确定每个 kernel 的资源使用量，可以在程序编译时向 `nvcc` 使用 [`--resource-usage`](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#resource-usage-res-usage) 选项；它会显示 kernel 所需的 register 和 shared memory 数量。

下面考虑一个 compute capability 10.0 的 device，其设备属性如下表所示。

| 资源 | 值 |
| --- | ---: |
| `maxBlocksPerMultiProcessor` | 32 |
| `sharedMemPerMultiprocessor` | 233472 |
| `regsPerMultiprocessor` | 65536 |
| `maxThreadsPerMultiProcessor` | 2048 |
| `sharedMemPerBlock` | 49152 |
| `regsPerBlock` | 65536 |
| `maxThreadsPerBlock` | 1024 |

表 2：SM 资源示例。

如果 kernel 以 `testKernel<<<512, 768>>>()` 启动，也就是每个 block 有 768 个线程，那么每个 SM 同时只能执行 2 个 thread block。由于 `maxThreadsPerMultiProcessor` 为 2048，调度器不能在每个 SM 上分配超过 2 个 thread block。因此 occupancy 为 `(768 * 2) / 2048`，即 75%。

如果 kernel 以 `testKernel<<<512, 32>>>()` 启动，也就是每个 block 有 32 个线程，每个 SM 不会受到 `maxThreadsPerMultiProcessor` 的限制；但由于 `maxBlocksPerMultiProcessor` 为 32，调度器每个 SM 只能分配 32 个 thread block。由于每个 block 有 32 个线程，SM 上 resident 的线程总数为 `32 blocks * 32 threads per block`，即 1024 个线程。compute capability 10.0 的 SM 每个 SM 最多支持 2048 个 resident thread，因此此时 occupancy 为 `1024 / 2048`，即 50%。

也可以对 shared memory 进行同样的分析。例如，如果一个 kernel 使用 100 KB shared memory，调度器每个 SM 只能分配 2 个 thread block，因为第三个 thread block 会再需要 100 KB shared memory，使总需求达到 300 KB，超过每个 SM 可用的 233472 字节。

每个 block 的线程数和每个 block 的 shared memory 使用量由程序员显式控制，可以调整它们以达到期望的 occupancy。程序员对 register 使用量的控制有限，因为编译器和 runtime 会尝试优化 register 使用。但程序员可以通过 `nvcc` 的 [`--maxrregcount`](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#maxrregcount-amount-maxrregcount) 选项指定每个 thread block 的最大 register 数量。如果 kernel 需要的 register 多于指定数量，就很可能溢出到 local memory，从而改变 kernel 的性能特征。在某些情况下，即使发生 spilling，限制 register 也能让更多 thread block 被调度，进而提高 occupancy，并最终带来性能净提升。
