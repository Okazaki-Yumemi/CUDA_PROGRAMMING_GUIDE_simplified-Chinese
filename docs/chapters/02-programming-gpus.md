---
title: 第二部分：用 CUDA 编程 GPU
description: CUDA C++、CUDA Python、SIMT kernel、tile kernel、异步执行和内存管理
---

# 第二部分：用 CUDA 编程 GPU

> 原文部分：**2. Programming GPUs in CUDA**。本页把官方目录中的七个主题串成可运行的学习路径，并保留关键 API/变量的英文写法。

## 2.1 CUDA C++

### 2.1.1 使用 NVCC 编译

CUDA C++ 源文件可以同时包含 host code 和 device code。`nvcc` 会识别 CUDA 扩展，把 device 部分编译为 GPU 代码，把 host 部分交给系统 C++ 编译器，再把二者链接到一起。常见文件扩展名是 `.cu`。

编译时需要明确：目标 compute capability、要包含的头文件、链接的 CUDA 库，以及是否需要保留 PTX 以支持后续 JIT。不要把“编译通过”当作“目标 GPU 上一定快”；架构选择、寄存器用量和内存访问仍需单独测量。

### 2.1.2 Kernel

用 `__global__` 声明一个可以从 host 启动、在 device 上执行的函数。kernel 没有普通 C++ 函数意义上的返回值；结果通常写入 device memory、managed memory 或通过其他输出参数传回。

```cpp
__global__ void scale(float* data, int count, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        data[i] *= factor;
    }
}
```

这里的索引公式把“线程在 block 内的位置”和“block 在 grid 内的位置”组合成一个全局线性索引。边界判断 `i < count` 不能省略，因为通常会用向上取整的 block 数量覆盖完整数组，最后一个 block 可能包含超出数组范围的线程。

### 2.1.3 启动 kernel 与三尖括号

CUDA C++ 使用 `<<<...>>>` 指定 kernel 的 execution configuration：

```cpp
int threads = 256;
int blocks = (count + threads - 1) / threads;
scale<<<blocks, threads>>>(device_data, count, 2.0f);
```

最基本的形式是 `kernel<<<grid, block>>>(args...)`。较新的 CUDA 功能还可以在执行配置中加入 cluster、stream 或 SM 配置。线程数不是越多越好：block 太小可能浪费调度机会，太大则可能受到寄存器或 shared memory 约束，降低同时驻留的 block 数量。

### 2.1.4 一个可运行的最小心智模型

下面的向量加法展示 host 分配、kernel launch、同步与释放的最小闭环。它是理解 API 生命周期的示例，不是所有生产程序都应采用的最佳性能方案。

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void vector_add(const float* a, const float* b,
                           float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    constexpr int n = 1 << 20;
    float *a = nullptr, *b = nullptr, *c = nullptr;
    cudaMallocManaged(&a, n * sizeof(float));
    cudaMallocManaged(&b, n * sizeof(float));
    cudaMallocManaged(&c, n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 1.0f;
    }

    vector_add<<<(n + 255) / 256, 256>>>(a, b, c, n);
    cudaDeviceSynchronize();

    std::printf("c[0] = %f\n", c[0]);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
```

这个例子使用 unified memory 是为了压缩演示中的复制步骤。它仍然需要 `cudaDeviceSynchronize()`：kernel launch 通常是异步的，host 线程在 GPU 完成计算前就可能继续执行。生产代码应在每个 CUDA API 调用和 kernel launch 后检查错误，而不是只在程序结束时检查。

### 2.1.5 线程与网格索引内建变量

常用内建变量包括：

| 变量 | 含义 |
| --- | --- |
| `threadIdx` | 当前线程在线程块内的坐标，支持 `.x/.y/.z` |
| `blockIdx` | 当前线程块在网格内的坐标 |
| `blockDim` | 当前线程块的尺寸 |
| `gridDim` | 当前网格的尺寸 |
| `warpSize` | 当前设备的 warp 宽度，常见值为 32 |

不要把 `threadIdx.x` 当成整个 kernel 的全局索引；只有组合 `blockIdx.x * blockDim.x + threadIdx.x`，才得到一维 grid 下的全局位置。二维图像、三维体数据和 batch 维度则应明确每一维的布局，避免把内存行列顺序与数学坐标顺序混在一起。

### 2.1.6 统一内存与显式内存管理

`cudaMallocManaged` 把分配和地址空间管理简化为统一指针；`cudaMalloc`/`cudaMemcpy` 则把 device allocation 与 host-device copy 暴露出来。显式管理适合需要精确控制数据驻留、传输时机和多设备路径的程序；统一内存适合先验证算法、减少样板代码或使用复杂指针结构的场景。

统一内存不是“没有复制”。页面可能在 CPU 和 GPU 之间迁移，访问模式不规则时会引入缺页和迁移开销。可以结合预取、访问提示和 kernel 边界同步减少意外迁移，但仍应通过 profiler 验证。

### 2.1.7 让 CPU 与 GPU 同步

常见同步边界有：

- `cudaDeviceSynchronize()`：等待设备上此前提交的工作完成；
- `cudaStreamSynchronize(stream)`：只等待某一个 stream；
- `cudaEventRecord` / `cudaEventSynchronize`：用 event 记录和等待特定时间点；
- kernel 内的 `__syncthreads()`：只同步同一个 thread block 的线程。

`__syncthreads()` 不是全局同步；不同 block 之间不能通过它交换结果。把它放在条件分支中也必须确保同一个 block 的所有线程都能以一致方式到达，否则可能死锁或产生未定义行为。

### 2.1.8 CUDA 中的错误检查

CUDA 错误有同步和异步两类。API 调用可能立刻返回参数或资源错误；kernel 内发生的错误可能要到后续同步点才暴露。开发阶段可使用如下辅助宏：

```cpp
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err__ = (call);                                   \
    if (err__ != cudaSuccess) {                                   \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,  \
                     cudaGetErrorString(err__));                  \
        std::exit(EXIT_FAILURE);                                  \
    }                                                             \
} while (0)

CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

`cudaGetLastError()` 适合检查最近一次 launch 的错误状态；再加上 `cudaDeviceSynchronize()` 才能把许多异步执行错误带回 host。不要在不理解语义的情况下随意清空错误状态，否则会丢失真正的失败位置。

## 2.2 CUDA Python

CUDA Python 不是单一产品，而是一组 Python 生态路径：Numba 的 CUDA kernel、CuPy 的 GPU array、PyTorch/JAX 等框架，以及对 CUDA 库的 Python 封装。选择时应先看数据结构和现有框架，再决定是写 Python kernel 还是调用成熟库。

### 2.2.1 生态与作用边界

- **数组编程**：适合把 NumPy 风格表达迁移到 GPU，减少手写索引；
- **JIT kernel**：需要自定义逐元素、归约或融合操作时，用 `@cuda.jit` 等接口表达 kernel；
- **框架扩展**：已有深度学习或科学计算工作流时，优先复用框架的 CUDA tensor 和 stream 语义；
- **CUDA libraries**：矩阵乘、卷积、FFT、随机数等成熟运算通常先找库。

### 2.2.2 Python 中的 SIMT kernel

下面是 Numba 风格的向量加法示意：

```python
from numba import cuda

@cuda.jit
def vector_add(a, b, c):
    i = cuda.grid(1)
    if i < c.size:
        c[i] = a[i] + b[i]

threads = 256
blocks = (n + threads - 1) // threads
vector_add[blocks, threads](a, b, c)
cuda.synchronize()
```

Python 的语法更短，但底层约束没有消失：线程索引、边界检查、内存驻留、同步和异步错误仍然存在。对于数据类型、数组布局或设备支持敏感的代码，要以所用 Python 库的版本文档为准。

### 2.2.3 Python 中的内存与同步

数组对象通常会记录自己所在的设备和内存空间。host 到 GPU 的 copy、GPU 到 host 的 copy 和 stream 调度可能是异步的；在需要读取结果或释放资源前，应使用库提供的同步语义。不要因为 Python 语句已经返回，就假定 GPU 工作已经完成。

## 2.3 编写 SIMT Kernel

### 2.3.1 线程层次

SIMT kernel 的第一步是确定工作项与线程的映射：一个线程负责一个元素、多个元素，还是一个 tile 中的一部分？映射决定索引公式、内存访问顺序和边界处理方式。

线程块是同步和 shared memory 的主要作用域；grid 是调度和总体任务规模的作用域。只要 block 间没有未处理的数据依赖，grid 就可以扩展到不同数量的 SM 上。

### 2.3.2 GPU device memory spaces

| 空间 | 典型作用 | 主要注意点 |
| --- | --- | --- |
| registers | 当前线程的私有局部变量 | 过多会降低每个 SM 能同时驻留的线程数 |
| local memory | 线程私有但实际可能落到 device memory 的变量 | 访问延迟高，通常由寄存器溢出或大局部数组引起 |
| shared memory | block/cluster 内协作和 tile 缓冲 | 容量有限，需要同步，并要检查 bank conflict |
| global memory | 大容量、跨 kernel 的数据 | 关注合并访问、对齐、带宽和访问复用 |
| constant memory | 只读且适合广播的参数 | 访问模式和容量有特殊约束 |
| texture/read-only cache | 特定只读或空间局部性访问 | 应根据目标架构与 API 支持选择 |

### 2.3.3 合并的全局内存访问

一个 warp 的线程如果访问连续且对齐的全局内存地址，硬件可以用较少的内存事务满足整个 warp 的请求，这通常称为 coalesced access。反之，跨步或随机访问可能需要更多事务，即使总共读取的字节数相同，也会降低有效带宽。

常见的优化步骤是：

1. 先画出一个 warp 的线程到地址的映射；
2. 确认相邻 lane 是否访问相邻元素；
3. 检查行主序/列主序和对齐；
4. 如果算法需要转置或重排，用 shared memory 把不规则访问隔离在局部阶段；
5. 用 Nsight Compute 等工具确认理论分析是否对应实际事务和吞吐。

### 2.3.4 Shared memory 与 bank conflict

shared memory 被划分为多个 bank。一个 warp 同时访问不同 bank 时可以并行服务；多个 lane 访问同一 bank 的不同地址时可能发生 bank conflict。广播同一个地址、以及某些设备支持的访问模式，不能简单按“同 bank 必冲突”判断，应该结合目标架构文档。

矩阵转置中常见的技巧是把 `32 x 32` 的 tile 改成 `32 x 33`，用额外一列打破行与 bank 的周期对齐。PDF 的[图 15](../figures.html#图-15-共享内存中的步长访问)和[图 18](../figures.html#图-18-带-padding-的共享内存)把这个现象画成了地址到 bank 的映射。

### 2.3.5 原子操作、协作组和占用率

原子操作保证某个内存位置上的更新不会被并发线程以不一致方式覆盖，但它不等于全局排序。热点地址上的大量原子操作会产生串行化，需要考虑分层归约、局部计数和最终合并。

Cooperative Groups 把线程组和同步关系表达得更明确；block、warp、tile、cluster 可以成为不同协作作用域。占用率（occupancy）描述一个 SM 上实际驻留的线程数相对于硬件上限的比例，但高占用率不自动等于高性能。寄存器、shared memory、指令级并行、内存延迟和实际吞吐应一起分析。

## 2.4 编写 Tile Kernel

tile kernel 把“线程如何协作加载/计算/存储”提升为一等概念。相比逐元素 kernel，它更容易利用数据复用和矩阵运算硬件，但也更依赖 tile 形状、边界、共享内存容量、对齐以及同步点。

### 2.4.1 数组与 tile

先定义逻辑数组和 tile 的坐标系统。例如二维矩阵 `A[row, col]` 可以按 `tile_rows x tile_cols` 分块，每个 block 负责一个 tile。边界 tile 可能只有部分元素有效，加载和写回都应带有效谓词，不能让越界线程访问数组。

### 2.4.2 Tile 空间与数据移动

tile-space load 把一个 tile 从 global memory 搬入寄存器或 shared memory；tile-space store 则把结果写回。较新的硬件上，某些 tile-space loads 可以由 Tensor Memory Accelerator（TMA）降低为更高效的搬运路径，但这需要支持的硬件、描述符、对齐和正确的同步语义。

### 2.4.3 Tile 上的操作

常见 tile 操作包括逐元素加法、归约、矩阵乘、转置、排序或布局重排。tile 内的结果交换可能依赖 block/cluster 同步；当数据需要跨 block 共享时，必须改变算法分阶段执行，或使用明确支持该作用域的机制。

### 2.4.4 Tile 与 SIMT 的关系

tile 是对一组数据和协作的抽象，SIMT 是线程执行模型。tile API 可以隐藏一部分线程索引细节，但不会消除线程发散、内存带宽、同步或资源限制。调试时应先理解 tile API 生成的线程和内存动作，再进行性能推理。

## 2.5 异步执行

CUDA 的异步接口让 host 线程提交工作后立即返回。异步工作可以包括 kernel launch、host-device copy、device-device copy、stream 操作以及 CUDA Graph 执行。并发的前提不是“调用没有阻塞”这么简单，而是：工作之间没有依赖，使用的 stream/事件关系正确，且硬件有足够的复制引擎和执行资源。

### 2.5.1 Stream 与 Event

stream 是有序的工作队列。同一个 stream 内的操作按顺序执行；不同 stream 之间只有在显式依赖或隐式资源约束下才存在顺序关系。event 可以记录一个 stream 中的时间点，并让另一个 stream 等待它。

一个典型的流水线是：

`copy tile A → kernel(tile A) → copy result A` 与 `copy tile B → kernel(tile B) → copy result B` 在不同 stream 中交错，让复制和计算重叠。但前提是 host buffer、device buffer 和生命周期都正确；过早复用 buffer 会造成数据竞争。

### 2.5.2 CUDA Graph

CUDA Graph 把 kernel、复制、事件和依赖关系表示为一个可实例化的图。定义或 capture 阶段描述工作，instantiate 阶段把图准备为可执行形式，launch 阶段重复提交。对大量重复的小 kernel，图可以减少每次提交的 CPU 开销；对结构动态变化的工作流，构图/更新成本也必须纳入考虑。

图的正确性取决于依赖边。没有显式边的节点不应被当作有先后关系；使用 stream capture 时，还要注意哪些操作允许被捕获，哪些 API、同步或外部副作用不适合出现在捕获区域。

### 2.5.3 异步复制与流水线

异步数据复制把数据搬运和计算拆开。`cudaMemcpyAsync`、shared memory 的异步 pipeline、TMA 等路径都要求明确数据何时可读、何时可覆写。可以使用双缓冲或多缓冲：当前 buffer 在计算，下一 buffer 同时搬运；切换时用 barrier、event 或 pipeline API 保证可见性。

## 2.6 统一内存与系统内存

### 2.6.1 统一内存

统一内存把 host 与 device 的访问组织到统一虚拟地址空间中。它可以通过按需迁移、预取和访问策略，在正确性层面简化指针管理。性能层面仍要考虑：

- CPU 和 GPU 是否同时访问同一页；
- 页面是否在频繁抖动；
- GPU 是否支持并发 managed access；
- host/device 之间的互连带宽与一致性能力；
- 是否需要 `cudaMemPrefetchAsync` 或访问提示。

### 2.6.2 系统内存与可移植性

不同操作系统、WSL、Tegra、NVLink C2C 和 PCIe 系统的统一内存支持可能不同。应用启动时可以查询设备属性，选择统一内存、显式复制或其他路径。不要只在开发机上验证，然后假定所有部署设备拥有相同的一致性和迁移能力。

## 2.7 NVCC：NVIDIA CUDA 编译器

NVCC 的角色是协调 host 编译器与 device 编译流程。实用的构建检查清单包括：

- `.cu` 文件中的 host/device 函数是否使用正确的声明符；
- `-arch` / `-gencode` 是否覆盖部署设备；
- 是否需要在 fatbin 中保留 PTX；
- host 编译器版本、C++ 标准和 ABI 是否匹配；
- 调试构建是否保留行号、设备调试信息和较少的优化；
- Release 构建是否检查寄存器、shared memory、代码大小和启动配置。

编译器诊断、device linking、host/device 函数限制、预处理宏和扩展语法都是版本敏感内容。遇到编译器报错时，优先查看对应 CUDA Toolkit 版本的 nvcc 文档，而不要根据旧版博客猜测行为。

## 本部分小结

第二部分把第一部分的抽象落成了一个工程循环：

`选择数据布局 → 设计线程/块映射 → 选择内存空间 → 编写并启动 kernel → 在正确的作用域同步 → 检查异步错误 → 用 profiler 验证性能`

下一步可进入[第三部分：高级 CUDA](./03-advanced-cuda.html)，或者先通过[PDF 图版](../figures.html)把访存和调度示意图对照代码看一遍。
