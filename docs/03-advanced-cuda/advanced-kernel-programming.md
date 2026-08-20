---
title: '3.2 高级 Kernel 编程（Advanced Kernel Programming）'
description: NVIDIA CUDA Programming Guide 第 3.2 节：PTX、SIMT 硬件、多线程、线程作用域、同步原语和异步数据复制
---

# 3.2 高级 Kernel 编程（Advanced Kernel Programming）

本节首先深入介绍 NVIDIA GPU 的硬件模型，然后介绍 CUDA kernel 代码中用于提升性能的一些高级特性。这些内容会引入线程作用域、异步执行及其同步原语等概念，为理解 kernel 代码中可用的高级性能特性打下基础。

其中一些特性的完整说明位于本指南后面的 CUDA 功能章节：本节介绍的[高级同步原语](#324-高级同步原语-advanced-synchronization-primitives)会在[第 4.9 节：异步屏障](../04-cuda-features/asynchronous-barriers.html)和[第 4.10 节：Pipeline](../04-cuda-features/pipelines.html)中完整介绍；本节介绍的[异步数据复制](#325-异步数据复制-asynchronous-data-copies)，包括 Tensor Memory Accelerator（TMA），会在[第 4.11 节：异步数据复制](../04-cuda-features/asynchronous-data-copies.html)中完整介绍。

## 3.2.1 使用 PTX（Using PTX）

*Parallel Thread Execution*（PTX）是 CUDA 用来抽象底层硬件 ISA 的虚拟机指令集架构，在[第 1.3 节 CUDA 平台](../01-introduction/cuda-platform.html)中已经介绍。直接编写 PTX 是一种非常高级的优化技术，大多数开发者都不需要使用，应当把它视为最后手段。不过，在某些场景下，直接编写 PTX 所提供的细粒度控制确实可以提升特定应用的性能。这类场景通常位于对性能极其敏感的代码路径中，即使性能提升不到一个百分点也可能带来显著收益。所有可用的 PTX 指令都记录在 [PTX ISA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)中。

### `cuda::ptx` 命名空间

直接使用 PTX 的一种方式是使用 [libcu++](https://nvidia.github.io/cccl/unstable/libcudacxx/)中的 `cuda::ptx` 命名空间。该命名空间提供了与 PTX 指令直接对应的 C++ 函数，简化了在 C++ 应用中使用 PTX 的过程。更多信息请参阅 [`cuda::ptx` 命名空间](https://nvidia.github.io/cccl/unstable/libcudacxx/ptx_api.html)文档。

### Inline PTX

把 PTX 嵌入代码的另一种方式是使用 inline PTX。该方法在 [Inline PTX Assembly 文档](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)中有详细说明，使用方式与在 CPU 上编写汇编代码非常相似。

## 3.2.2 硬件实现（Hardware Implementation）

流式多处理器（Streaming Multiprocessor，SM；参见[第 1.2 节中的 GPU 硬件模型](../01-introduction/programming-model.html)）被设计为同时执行数百个线程。为了管理如此大量的线程，SM 使用一种称为 *Single-Instruction, Multiple-Thread*（SIMT，单指令多线程）的并行计算模型，详见[SIMT 执行模型](#3221-simt-执行模型-simt-execution-model)。指令采用流水线执行：单个线程内部利用指令级并行，同时通过硬件多线程实现大量线程级并行，详见[硬件多线程](#3222-硬件多线程-hardware-multithreading)。与 CPU 核心不同，SM 按顺序发射指令，不进行分支预测或推测执行。

[SIMT 执行模型](#3221-simt-执行模型-simt-execution-model)和[硬件多线程](#3222-硬件多线程-hardware-multithreading)介绍所有设备共有的 SM 架构特性；不同 compute capability 设备的具体参数见[计算能力](../05-technical-appendices/compute-capabilities.html)。

NVIDIA GPU 架构使用小端序表示。

### 3.2.2.1 SIMT 执行模型（SIMT Execution Model）

每个 SM 会创建、管理、调度并执行由 32 个并行线程组成的线程组，这种线程组称为 *warp*。组成一个 warp 的线程从同一个程序地址同时开始执行，但每个线程都有自己的指令地址计数器和寄存器状态，因此可以独立分支和执行。*Warp* 一词源自织造业，最早的并行线程技术使用了这一术语。*Half-warp* 是一个 warp 的前半部分或后半部分；*quarter-warp* 是一个 warp 的四个四分之一中的任意一个。

一个 warp 每次执行一条公共指令；当 warp 中的 32 个线程都选择相同的执行路径时，效率最高。如果线程因为数据相关的条件分支而发生分歧，warp 会依次执行每条被选择的分支路径，并禁用不在当前路径上的线程。分支分歧只发生在 warp 内部；不同 warp 无论执行相同还是互不相同的代码路径，都可以彼此独立执行。

SIMT 架构与 SIMD（Single Instruction, Multiple Data，单指令多数据）向量组织相似，都是用一条指令控制多个处理单元。关键区别在于：SIMD 向量组织把 SIMD 宽度暴露给软件，而 SIMT 指令描述的是单个线程的执行和分支行为。与 SIMD 向量机相比，SIMT 允许程序员为独立的标量线程编写线程级并行代码，也允许为协同线程编写数据并行代码。

从正确性的角度看，程序员基本可以忽略 SIMT 的具体行为；但如果让同一个 warp 中的线程尽量少走分歧路径，通常可以获得显著的性能提升。实际使用时，这类似于缓存行：设计正确性时可以安全地忽略缓存行大小，但要追求峰值性能，就必须把它考虑到代码结构中。另一方面，向量架构要求软件手动把 load 合并成向量并管理分歧。

#### 3.2.2.1.1 独立线程调度（Independent Thread Scheduling）

在 compute capability 小于 7.0 的 GPU 上，一个 warp 的 32 个线程共享同一个程序计数器，同时使用 active mask 指示 warp 中哪些线程处于活动状态。因此，处于分歧区域或不同执行状态的同一 warp 线程不能相互发信号或交换数据；依赖锁或互斥量进行细粒度数据共享的算法，可能根据竞争线程来自哪个 warp 而发生死锁。

在 compute capability 7.0 及更高版本的 GPU 上，*独立线程调度*（independent thread scheduling）允许线程之间实现与 warp 无关的完全并发。GPU 会为每个线程维护执行状态，包括程序计数器和调用栈，并可以在线程粒度让出执行权：既可以更好地利用执行资源，也可以让一个线程等待另一个线程产生数据。调度优化器会把同一个 warp 中的活动线程重新组织为 SIMT 单元。这样既保留了此前 NVIDIA GPU 的高吞吐 SIMT 执行，又提供了更大的灵活性：线程可以在小于一个 warp 的粒度上分歧和重新汇合。

独立线程调度可能破坏依赖旧 GPU 架构隐式 warp 同步行为的代码。*Warp-synchronous* 代码假设同一 warp 的线程在每条指令上都以锁步方式执行；但独立线程调度可以在子 warp 粒度分歧和重新汇合，这会使这种假设失效，并可能导致实际参与代码执行的线程集合与预期不同。为 CC 7.0 之前 GPU 编写的 warp-synchronous 代码（例如不使用同步的 warp 内归约）都应重新检查兼容性。应使用 `__syncwarp()` 显式同步此类代码，以确保在所有 GPU 代际上行为正确。

> **注意：** 当前指令中参与执行的 warp 线程称为 *active threads*，未处于当前指令上的线程称为 *inactive*（disabled）。线程可能因为早于同一 warp 的其他线程退出、选择了当前 warp 正在执行的另一条分支路径，或者位于线程数不是 warp size 整数倍的 block 尾部而处于 inactive 状态。
>
> 如果 warp 执行的非 atomic 指令由多个线程向 global 或 shared memory 的同一位置写入，发生的串行写入次数可能因设备的 compute capability 而不同。但是对于所有 compute capability，最终执行写入的线程都是未定义的。
>
> 如果 warp 执行的 atomic 指令由多个线程对 global memory 的同一位置执行读、改、写，则每次读/改/写都会发生并且这些操作会串行化，但它们发生的顺序未定义。

### 3.2.2.2 硬件多线程（Hardware Multithreading）

当 SM 获得一个或多个待执行的 thread block 时，会把 block 划分为 warp，并由 *warp scheduler* 调度每个 warp。block 划分为 warp 的方式始终相同：每个 warp 包含连续递增的 thread ID，且第一个 warp 从线程 0 开始。[线程层次结构](../02-programming-gpus/writing-simt-kernels.html)介绍 thread ID 与 block 内线程索引之间的关系。

一个 block 中的 warp 总数定义为：

\[
\operatorname{ceil}\left(\frac{T}{W_{size}}\right)
\]

- `T` 是每个 block 的线程数；
- `Wsize` 是 warp size，等于 32；
- `ceil(x, y)` 表示将 `x` 向上取整到 `y` 的最小整数倍。

![图 22：一个 thread block 被划分为 32 线程的 warp](/images/chapter-03/warps-in-a-block.png)

图 22 说明：thread block 按连续的线程 ID 划分为多个 warp，每个 warp 包含最多 32 个线程；当 block 的线程数不是 32 的整数倍时，最后一个 warp 可能包含 inactive 线程。

SM 处理的每个 warp 的执行上下文（程序计数器、寄存器等）在整个 warp 生命周期内都保存在片上。因此，在 warp 之间切换不会产生开销。在每个指令发射周期，warp scheduler 会选择一个线程已经准备好执行下一条指令的 warp（即该 warp 的 active threads），并向这些线程发射指令。

每个 SM 都有一组 32 位寄存器，这些寄存器在 warp 之间分配；SM 还拥有由 thread block 划分使用的 shared memory。对于给定 kernel，SM 能同时驻留并处理的 block 和 warp 数量，取决于 kernel 使用的寄存器和 shared memory 数量，以及 SM 上可用的寄存器和 shared memory 数量。每个 SM 还会受到驻留 block 数和 warp 数上限的限制。这些上限以及 SM 上可用的寄存器、shared memory 数量取决于设备的 compute capability，具体见[计算能力](../05-technical-appendices/compute-capabilities.html)。如果每个 SM 都没有足够资源处理至少一个 block，kernel 将启动失败。确定每个 block 分配的寄存器和 shared memory 总量有多种方式，详见[启动与 occupancy](../02-programming-gpus/writing-simt-kernels.html)。

### 3.2.2.3 异步执行特性（Asynchronous Execution Features）

近代 NVIDIA GPU 增加了异步执行能力，使数据移动、计算和同步可以在 GPU 内部更好地重叠。这些能力允许由 GPU 代码调用的某些操作相对于同一 thread block 中的其他 GPU 代码异步执行。这里的异步执行不要与[第 2.5 节异步执行](../02-programming-gpus/asynchronous-execution.html)混淆：后者讨论的是 kernel launch 或内存操作彼此之间，或相对于 CPU 的异步性。

Compute capability 8.0（NVIDIA Ampere 架构）引入了从 global memory 到 shared memory 的硬件加速异步数据复制和异步屏障，详见 [NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)。

Compute capability 9.0（NVIDIA Hopper 架构）进一步扩展了异步执行特性，加入 [Tensor Memory Accelerator（TMA）](#325-异步数据复制-asynchronous-data-copies)单元、异步事务屏障和异步矩阵乘加操作。TMA 可以在 global memory 与 shared memory 之间传输大块数据和多维 tensor；更多细节见 [Hopper Architecture in Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)。

CUDA 提供了可以由 device code 中的线程调用的 API，以使用这些特性。异步编程模型定义了异步操作相对于 CUDA 线程的行为。

异步操作由一个 CUDA 线程发起，但会像由另一个线程执行一样异步执行；这里把后者称为 *async thread*。在格式正确的程序中，一个或多个 CUDA 线程会与该异步操作同步。发起异步操作的 CUDA 线程不一定属于参与同步的线程集合。Async thread 始终与发起操作的 CUDA 线程关联。

异步操作使用同步对象来发出完成信号，这个同步对象可以是 barrier 或 pipeline。同步对象会在[高级同步原语](#324-高级同步原语-advanced-synchronization-primitives)中详细说明，[异步数据复制](#325-异步数据复制-asynchronous-data-copies)则展示如何使用它们完成异步内存操作。

#### 3.2.2.3.1 Async Thread 与 Async Proxy

异步操作访问内存的方式可能与普通操作不同。为区分这些内存访问方法，CUDA 引入了 *async thread*、*generic proxy* 和 *async proxy* 的概念。普通操作（load 和 store）通过 generic proxy 执行。某些异步指令，例如 [LDGSTS](../04-cuda-features/asynchronous-data-copies.html#ldgsts) 和 [STAS/REDAS](../04-cuda-features/asynchronous-data-copies.html#stas-redas)，被建模为在 generic proxy 中运行的 async thread。另一些异步指令，例如使用 TMA 的 bulk-asynchronous copy 和部分 tensor core 操作（`tcgen05.*`、`wgmma.mma_async.*`），则被建模为在 async proxy 中运行的 async thread。

**在 generic proxy 中运行的 async thread。** 发起异步操作时，操作会关联到一个不同于发起 CUDA 线程的 async thread。对同一地址的、*先于*该异步操作发生的 generic proxy（普通）load/store，保证排在异步操作之前；但是，后续对同一地址的普通 load/store 不保证保持该顺序，在 async thread 完成之前可能产生竞争条件。

**在 async proxy 中运行的 async thread。** 发起异步操作时，操作会关联到一个不同于发起 CUDA 线程的 async thread。对同一地址的、*先于或后于*该异步操作发生的普通 load/store，都不保证保持顺序。必须使用 proxy fence 在不同 proxy 之间进行同步，以保证正确的内存顺序。使用 TMA 执行异步复制时，需要 proxy fence 的示例见 [使用 Tensor Memory Accelerator（TMA）](../04-cuda-features/asynchronous-data-copies.html#using-the-tensor-memory-accelerator-tma)。

更多细节请参阅 [PTX ISA 中关于 proxy 的说明](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=proxy#proxies)。

## 3.2.3 线程作用域（Thread Scopes）

CUDA thread 组成[线程层次结构](../02-programming-gpus/writing-simt-kernels.html)，利用这一层次结构是编写正确且高性能 CUDA kernel 的基础。在这个层次结构中，内存操作的可见范围和同步范围可能不同。为了描述这种非均匀性，CUDA 编程模型引入了 *thread scope* 概念。线程作用域定义哪些线程可以观察到某个线程的 load/store，也定义哪些线程可以通过 atomic operation、barrier 等同步原语相互同步。每个作用域都对应内存层次结构中的一个一致性点（point of coherency）。

线程作用域暴露在 [CUDA PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=thread%2520scopes#scope)中，也作为 [libcu++](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#thread-scopes)的扩展提供。下表列出了可用的线程作用域：

| CUDA C++ 线程作用域 | CUDA PTX 线程作用域 | 说明 | 内存层次结构中的一致性点 |
| --- | --- | --- | --- |
| `cuda::thread_scope_thread` | — | 内存操作仅对本地线程可见。 | — |
| `cuda::thread_scope_block` | `.cta` | 内存操作对同一 thread block 中的其他线程可见。 | L1 |
| — | `.cluster` | 内存操作对同一 thread block cluster 中的其他线程可见。 | L2 |
| `cuda::thread_scope_device` | `.gpu` | 内存操作对同一 GPU device 上的其他线程可见。 | L2 |
| `cuda::thread_scope_system` | `.sys` | 内存操作对同一系统中的其他线程可见（CPU、其他 GPU）。 | L2 + 相连 cache |

[高级同步原语](#324-高级同步原语-advanced-synchronization-primitives)和[异步数据复制](#325-异步数据复制-asynchronous-data-copies)会展示线程作用域的使用方式。

## 3.2.4 高级同步原语（Advanced Synchronization Primitives）

本节介绍三类同步原语：

- [作用域 Atomic](#3241-作用域-atomic-scoped-atomics)：把 C++ 内存顺序与 CUDA 线程作用域结合起来，在 block、cluster、device 或 system 作用域安全地在线程间通信。
- [异步屏障](#3242-异步屏障-asynchronous-barriers)：把同步拆分为 arrive 和 wait 阶段，也可以用来跟踪异步操作的进度。
- [Pipeline](#3243-pipeline-pipelines)：对工作分阶段，并协调多缓冲 producer-consumer 模式，常用于把计算与[异步数据复制](#325-异步数据复制-asynchronous-data-copies)重叠起来。

### 3.2.4.1 作用域 Atomic（Scoped Atomics）

[第 5.4.5 节](../05-technical-appendices/cpp-language-extensions.html#atomic-functions)概述了 CUDA 提供的 atomic function。本节聚焦支持 [C++ 标准 atomic 内存语义](https://en.cppreference.com/w/cpp/atomic/memory_order.html)的*作用域 atomic*，它们可以通过 [libcu++](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives.html)或编译器内置函数使用。作用域 atomic 提供了在 CUDA 线程层次结构的适当层级上高效同步的工具，有助于在复杂并行算法中同时保证正确性和性能。

#### 3.2.4.1.1 线程作用域与内存顺序（Thread Scope and Memory Ordering）

作用域 atomic 结合了两个关键概念：

- **线程作用域（Thread Scope）**：定义哪些线程可以观察到 atomic 操作的效果，见[线程作用域](#323-线程作用域-thread-scopes)。
- **内存顺序（Memory Ordering）**：定义相对于其他内存操作的顺序约束，见 [C++ 标准 atomic 内存语义](https://en.cppreference.com/w/cpp/atomic/memory_order.html)。

下面的两个版本都实现了一个 block 作用域的 atomic counter。

#### CUDA C++ `cuda::atomic`

```cpp
#include <cuda/atomic>

__global__ void block_scoped_counter() {
    // 只在本 block 内可见的 shared atomic counter
    __shared__ cuda::atomic<int, cuda::thread_scope_block> counter;

    // 初始化 counter（只能由一个线程完成）
    if (threadIdx.x == 0) {
        counter.store(0, cuda::memory_order_relaxed);
    }
    __syncthreads();

    // block 中的所有线程执行 atomic increment
    int old_value = counter.fetch_add(1, cuda::memory_order_relaxed);

    // 使用 old_value……
}
```

#### 编译器内置 Atomic Function

```cpp
__global__ void block_scoped_counter() {
    // 只在本 block 内可见的 counter
    __shared__ int counter;

    // 初始化 counter（只能由一个线程完成）
    if (threadIdx.x == 0) {
        __nv_atomic_store_n(&counter, 0,
                            __NV_ATOMIC_RELAXED,
                            __NV_THREAD_SCOPE_BLOCK);
    }
    __syncthreads();

    // block 中的所有线程执行 atomic increment
    int old_value = __nv_atomic_fetch_add(&counter, 1,
                                          __NV_ATOMIC_RELAXED,
                                          __NV_THREAD_SCOPE_BLOCK);

    // 使用 old_value……
}
```

这个例子实现了一个 *block 作用域 atomic counter*，体现了作用域 atomic 的基本概念：

- **Shared variable**：使用 `__shared__` memory，让 block 中的所有线程共享同一个 counter。
- **Atomic 类型声明**：`cuda::atomic<int, cuda::thread_scope_block>` 创建一个具有 block 级可见性的 atomic integer。
- **单次初始化**：只有线程 0 初始化 counter，避免设置阶段出现竞争条件。
- **Block 同步**：`__syncthreads()` 确保所有线程在继续执行前都能看到已初始化的 counter。
- **Atomic increment**：每个线程以 atomic 方式递增 counter，并获得递增前的值。

这里选择 `cuda::memory_order_relaxed`，是因为我们只需要 atomicity（不可分割的读-改-写），不需要不同内存位置之间的顺序约束。对于简单计数操作，increment 的顺序不会影响正确性。

对于 producer-consumer 模式，需要使用 acquire-release 语义保证正确的顺序：

#### CUDA C++ `cuda::atomic`

```cpp
__global__ void producer_consumer() {
    __shared__ int data;
    __shared__ cuda::atomic<bool, cuda::thread_scope_block> ready;

    if (threadIdx.x == 0) {
        // Producer：先写 data，再发出 ready 信号
        data = 42;
        ready.store(true, cuda::memory_order_release);  // Release 确保 data 写入可见
    } else {
        // Consumer：等待 ready 信号，再读取 data
        while (!ready.load(cuda::memory_order_acquire)) {  // Acquire 确保读到 data 的写入
            // spin wait
        }
        int value = data;
        // 处理 value……
    }
}
```

#### 编译器内置 Atomic Function

```cpp
__global__ void producer_consumer() {
    __shared__ int data;
    __shared__ bool ready; // 只有 ready flag 需要 atomic 操作

    if (threadIdx.x == 0) {
        // Producer：先写 data，再发出 ready 信号
        data = 42;
        __nv_atomic_store_n(&ready, true,
                            __NV_ATOMIC_RELEASE,
                            __NV_THREAD_SCOPE_BLOCK);  // Release 确保 data 写入可见
    } else {
        // Consumer：等待 ready 信号，再读取 data
        while (!__nv_atomic_load_n(&ready,
                                   __NV_ATOMIC_ACQUIRE,
                                   __NV_THREAD_SCOPE_BLOCK)) {  // Acquire 确保读到 data 的写入
            // spin wait
        }
        int value = data;
        // 处理 value……
    }
}
```

#### 3.2.4.1.2 性能注意事项（Performance Considerations）

- **使用尽可能窄的作用域**：block 作用域 atomic 远快于 system 作用域 atomic。
- **优先使用较弱的内存顺序**：只有在正确性确实需要时才使用更强的顺序约束。
- **考虑内存位置**：shared memory atomic 比 global memory atomic 更快。

### 3.2.4.2 异步屏障（Asynchronous Barriers）

异步屏障与典型的单阶段屏障（`__syncthreads()`）不同：线程到达屏障的通知（*arrive*）与等待其他线程到达屏障的操作（*wait*）是分开的。这个拆分提升了执行效率，因为线程可以执行与屏障无关的额外工作，更有效地利用等待时间。异步屏障可以实现 CUDA 线程之间的 producer-consumer 模式，也可以让异步数据复制在完成时向屏障发出 arrive 信号，从而支持内存层次结构中的异步数据复制。

compute capability 7.0 及更高版本的设备支持异步屏障。compute capability 8.0 及更高版本的设备为 shared memory 中的异步屏障提供硬件加速，并且同步粒度有了显著提升：硬件可以加速 block 内任意线程子集的同步。更早的架构只在整个 warp（`__syncwarp()`）或整个 block（`__syncthreads()`）级别加速同步。

CUDA 编程模型通过 `cuda::std::barrier` 提供异步屏障；这是 [libcu++](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/barrier.html)提供的符合 ISO C++ 的 barrier。除了实现 [`std::barrier`](https://en.cppreference.com/w/cpp/thread/barrier.html)，该库还提供 CUDA 专用扩展，用于选择 barrier 的线程作用域以改善性能，并暴露较低层级的 [`cuda::ptx`](https://nvidia.github.io/cccl/unstable/libcudacxx/ptx_api.html) API。`cuda::barrier` 可以通过友元函数 `cuda::device::barrier_native_handle()` 与 `cuda::ptx` 互操作：先取得 barrier 的 native handle，再传给 `cuda::ptx` 函数。CUDA 还为 shared memory、block 线程作用域的异步屏障提供了[底层 primitives API](../05-technical-appendices/device-callable-apis.html)。

下表概括了不同线程作用域下可用的异步屏障：

| 线程作用域 | 内存位置 | Arrive on Barrier | Wait on Barrier | 硬件加速 | CUDA API |
| --- | --- | --- | --- | --- | --- |
| block | local shared memory | 支持 | 支持 | 支持（8.0+） | `cuda::barrier`、`cuda::ptx`、primitives |
| cluster | local shared memory | 支持 | 支持 | 支持（9.0+） | `cuda::barrier`、`cuda::ptx` |
| cluster | remote shared memory | 支持 | 不支持 | 支持（9.0+） | `cuda::barrier`、`cuda::ptx` |
| device | global memory | 支持 | 支持 | 不支持 | `cuda::barrier` |
| system | global/unified memory | 支持 | 支持 | 不支持 | `cuda::barrier` |

#### 时序拆分的同步（Temporal Splitting of Synchronization）

如果没有异步 arrive-wait barrier，thread block 内的同步通常使用 `__syncthreads()`，或者使用 [Cooperative Groups](../04-cuda-features/cooperative-groups.html)时使用 `block.sync()`：

```cpp
#include <cooperative_groups.h>

__global__ void simple_sync(int iteration_count) {
    auto block = cooperative_groups::this_thread_block();

    for (int i = 0; i < iteration_count; ++i) {
        /* code before arrive */

        // 等待所有线程到达这里。
        block.sync();

        /* code after wait */
    }
}
```

线程会在同步点（`block.sync()`）处阻塞，直到所有线程都到达该同步点。此外，同步点之前发生的内存更新保证在同步点之后对 block 中的所有线程可见。

这个模式有三个阶段：

1. 同步前的代码执行会在同步后读取的内存更新。
2. 同步点。
3. 同步后的代码可以看到同步前发生的内存更新。

使用异步屏障时，时序拆分的同步模式如下。三个代码版本展示了相同的 arrive/wait 逻辑。

##### CUDA C++ `cuda::barrier`

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

__device__ void compute(float *data, int iteration);

__global__ void split_arrive_wait(int iteration_count, float *data)
{
  using barrier_t = cuda::barrier<cuda::thread_scope_block>;
  __shared__ barrier_t bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    // 使用预期 arrive 次数初始化 barrier。
    init(&bar, block.size());
  }
  block.sync();

  for (int i = 0; i < iteration_count; ++i)
  {
    /* code before arrive */

    // 当前线程 arrive；arrive 不会阻塞线程。
    barrier_t::arrival_token token = bar.arrive();
    compute(data, i);

    // 等待所有参与 barrier 的线程完成预期次数的 bar.arrive()。
    bar.wait(std::move(token));
    /* code after wait */
  }
}
```

##### CUDA C++ `cuda::ptx`

```cpp
#include <cuda/ptx>
#include <cooperative_groups.h>

__device__ void compute(float *data, int iteration);

__global__ void split_arrive_wait(int iteration_count, float *data)
{
  __shared__ uint64_t bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    // 使用预期 arrive 次数初始化 barrier。
    cuda::ptx::mbarrier_init(&bar, block.size());
  }
  block.sync();

  for (int i = 0; i < iteration_count; ++i)
  {
    /* code before arrive */

    // 当前线程 arrive；arrive 不会阻塞线程。
    uint64_t token = cuda::ptx::mbarrier_arrive(&bar);
    compute(data, i);

    // 等待所有参与 barrier 的线程完成 mbarrier_arrive()。
    while (!cuda::ptx::mbarrier_try_wait(&bar, token)) {}
    /* code after wait */
  }
}
```

##### CUDA C primitives

```cpp
#include <cuda_awbarrier_primitives.h>
#include <cooperative_groups.h>

__device__ void compute(float *data, int iteration);

__global__ void split_arrive_wait(int iteration_count, float *data)
{
  __shared__ __mbarrier_t bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    // 使用预期 arrive 次数初始化 barrier。
    __mbarrier_init(&bar, block.size());
  }
  block.sync();

  for (int i = 0; i < iteration_count; ++i)
  {
    /* code before arrive */

    // 当前线程 arrive；arrive 不会阻塞线程。
    __mbarrier_token_t token = __mbarrier_arrive(&bar);
    compute(data, i);

    // 等待所有参与 barrier 的线程完成 __mbarrier_arrive()。
    while (!__mbarrier_try_wait(&bar, token, 1000)) {}
    /* code after wait */
  }
}
```

在这种模式中，同步点被拆分为 arrive 点（`bar.arrive()`）和 wait 点（`bar.wait(std::move(token))`）。线程第一次调用 `bar.arrive()` 时开始参与一个 `cuda::barrier`。调用 `bar.wait(std::move(token))` 的线程会一直阻塞，直到参与线程完成了预期次数的 `bar.arrive()`；预期次数由传给 `init()` 的 expected arrival count 参数决定。参与线程在调用 `bar.arrive()` 之前发生的内存更新，保证在它们调用 `bar.wait(std::move(token))` 后对参与线程可见。注意，`bar.arrive()` 不会阻塞线程；线程可以继续做不依赖其他参与线程在 `bar.arrive()` 前产生的内存更新的工作。

*Arrive and wait* 模式有五个阶段：

1. arrive 前的代码执行将在 wait 后读取的内存更新。
2. Arrive 点带有隐式 memory fence（等价于 `cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block)`）。
3. arrive 与 wait 之间的代码。
4. Wait 点。
5. wait 后的代码可以看到 arrive 前执行的更新。

关于异步屏障的完整使用指南，参阅[第 4.9 节：异步屏障](../04-cuda-features/asynchronous-barriers.html)。

### 3.2.4.3 Pipeline（Pipelines）

CUDA 编程模型提供 pipeline 同步对象，用于把异步内存复制排列成多个阶段，从而实现双缓冲或多缓冲的 producer-consumer 模式。Pipeline 是一个带有 *head* 和 *tail* 的双端队列，按先进先出（FIFO）顺序处理工作。Producer 线程把工作提交到 pipeline 的 head，consumer 线程从 pipeline 的 tail 取出工作。

Pipeline 通过 [libcu++](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/pipeline.html)中的 `cuda::pipeline` API 暴露，也通过[底层 primitives API](../05-technical-appendices/device-callable-apis.html)暴露。两个 API 的主要功能如下。

| `cuda::pipeline` API | 说明 |
| --- | --- |
| `producer_acquire` | 获取 pipeline 内部队列中的一个可用 stage。 |
| `producer_commit` | 提交在 `producer_acquire` 后、当前获取 stage 上发起的异步操作。 |
| `consumer_wait` | 等待 pipeline 最旧 stage 中的异步操作完成。 |
| `consumer_release` | 将 pipeline 最旧 stage 释放回 pipeline 对象以便复用；释放的 stage 随后可以被 producer 获取。 |

| Primitives API | 说明 |
| --- | --- |
| `__pipeline_memcpy_async` | 请求把数据从 global memory 复制到 shared memory，以便异步执行。 |
| `__pipeline_commit` | 提交在调用前、当前 pipeline stage 上发起的异步操作。 |
| `__pipeline_wait_prior(N)` | 等待 pipeline 中除最后 `N` 次 commit 之外的所有异步操作完成。 |

`cuda::pipeline` API 的接口更丰富，限制更少；primitives API 只支持跟踪从 global memory 到 shared memory 的异步复制，并且复制大小和对齐方式受到特定要求。Primitives API 相当于一个使用 `cuda::thread_scope_thread` 的 `cuda::pipeline` 对象。

详细的使用模式和示例见[第 4.10 节：Pipeline](../04-cuda-features/pipelines.html)。

## 3.2.5 异步数据复制（Asynchronous Data Copies）

在内存层次结构中高效移动数据，是 GPU 计算获得高性能的基础。传统同步内存操作会让线程在数据传输期间空闲等待。GPU 通常通过并行性隐藏内存延迟：在内存操作完成时，SM 切换去执行另一个 warp。即使可以通过并行性隐藏延迟，内存延迟仍可能成为内存带宽利用率和计算资源效率的瓶颈。为解决这些瓶颈，现代 GPU 架构提供了硬件加速的异步数据复制机制，使内存传输可以独立进行，同时线程继续执行其他工作。

异步数据复制把内存传输的发起与等待完成解耦，从而可以将计算与数据移动重叠。这样，线程可以在内存延迟期间执行有用工作，提升总体吞吐和资源利用率。

> **注意：** 本节的概念与前面的[第 2.5 节异步执行](../02-programming-gpus/asynchronous-execution.html)相似，但第 2.5 节讨论的是 `cudaMemcpyAsync` 等 API 发起的 kernel 和内存传输异步执行，可以理解为应用不同组成部分之间的异步性。
>
> 本节讨论的是：在不阻塞 GPU 线程的情况下，把数据从 GPU DRAM（即 global memory）传输到 SM 上的内存（例如 shared memory 或 tensor memory），或者反向传输。这是单次 kernel launch 内部执行过程中的异步性。

为了理解异步复制如何提升性能，可以看一个常见的 GPU 计算模式。CUDA 应用经常采用 *copy and compute* 模式：

- 从 global memory 获取数据；
- 把数据存储到 shared memory；
- 使用 shared memory 中的数据执行计算，并可能把结果写回 global memory。

这一模式中的 *copy* 阶段通常写成 `shared[local_idx] = global[global_idx]`。编译器会把 global 到 shared 的复制展开为：先从 global memory 读入 register，再从 register 写入 shared memory。

当这个模式出现在迭代算法中时，每个 thread block 都需要在 `shared[local_idx] = global[global_idx]` 之后同步，确保对 shared memory 的所有写入完成后再开始计算。计算阶段之后也需要再次同步，避免在所有线程完成计算前覆盖 shared memory。下面的代码展示了这种模式。

```cpp
#include <cooperative_groups.h>

__device__ void compute(int* global_out, int const* shared_in) {
    // 使用 shared memory 中当前 batch 的所有值计算。
    // 把当前线程的结果写回 global memory。
}

__global__ void without_async_copy(int* global_out, int const* global_in,
                                   size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size()); // 示例假设输入大小适配 batch_sz * grid_size

  extern __shared__ int shared[]; // block.size() * sizeof(int) 字节
  size_t local_idx = block.thread_rank();

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    // 计算该 block 在 global memory 中当前 batch 的索引。
    size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
    size_t global_idx = block_batch_idx + threadIdx.x;
    shared[local_idx] = global_in[global_idx];

    // 等待所有复制完成。
    block.sync();

    // 计算并把结果写回 global memory。
    compute(global_out + block_batch_idx, shared);

    // 等待所有使用 shared memory 的计算完成。
    block.sync();
  }
}
```

使用异步数据复制时，可以让 global memory 到 shared memory 的数据移动异步进行，从而更高效地利用 SM 等待数据到达期间的资源：

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

__device__ void compute(int* global_out, int const* shared_in) {
    // 使用 shared memory 中当前 batch 的所有值计算。
    // 把当前线程的结果写回 global memory。
}

__global__ void with_async_copy(int* global_out, int const* global_in,
                                size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size()); // 示例假设输入大小适配 batch_sz * grid_size

  extern __shared__ int shared[]; // block.size() * sizeof(int) 字节

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    // 计算该 block 在 global memory 中当前 batch 的索引。
    size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;

    // 整个线程组协同把整个 batch 复制到 shared memory。
    cooperative_groups::memcpy_async(block, shared, global_in + block_batch_idx,
                                      block.size());

    // 在等待复制时计算其他数据。

    // 等待所有复制完成。
    cooperative_groups::wait(block);

    // 计算并把结果写回 global memory。
    compute(global_out + block_batch_idx, shared);

    // 等待所有使用 shared memory 的计算完成。
    block.sync();
  }
}
```

[`cooperative_groups::memcpy_async`](../05-technical-appendices/device-callable-apis.html)函数把 `block.size()` 个元素从 global memory 复制到 `shared` 数据中。该操作表现为由另一个线程执行；复制完成后，这个异步操作会与当前线程调用的 [`cooperative_groups::wait`](../05-technical-appendices/device-callable-apis.html)同步。在复制完成之前修改 global data，或读取/写入 shared data，都会引入数据竞争。

这个例子体现了所有异步复制的基本概念：把内存传输的发起与完成解耦，让线程在数据后台移动时执行其他工作。CUDA 编程模型提供了多种访问这些能力的 API，包括 [Cooperative Groups](../04-cuda-features/cooperative-groups.html)和 [libcu++](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html)中的 `memcpy_async`，以及较低层的 `cuda::ptx` 和 primitives API。这些 API 具有相似的语义：把对象从源复制到目标，效果如同由另一个线程执行；复制完成后，再使用不同的完成机制进行同步。

现代 GPU 架构提供了多种异步数据移动硬件机制：

- **LDGSTS（compute capability 8.0+）**：支持从 global memory 到 shared memory 的高效小规模异步传输。
- **Tensor Memory Accelerator（TMA，compute capability 9.0+）**：扩展了这些能力，提供针对大型多维数据传输优化的 bulk-asynchronous copy。
- **STAS 指令（compute capability 9.0+）**：支持在 cluster 内从 registers 到 distributed shared memory 的小规模异步传输。

这些机制具有不同的数据路径、传输大小和对齐要求，开发者应根据具体数据访问模式选择合适的机制。下表概括了 GPU 内部异步复制支持的数据路径：

| 源 | 目标 | 异步复制 | Bulk-asynchronous copy |
| --- | --- | --- | --- |
| global | global | — | — |
| `shared::cta` | global | — | 支持（TMA，9.0+） |
| global | `shared::cta` | 支持（LDGSTS，8.0+） | 支持（TMA，9.0+） |
| global | `shared::cluster` | — | 支持（TMA，9.0+） |
| `shared::cluster` | `shared::cta` | — | 支持（TMA，9.0+） |
| `shared::cta` | `shared::cta` | — | — |
| registers | `shared::cluster` | 支持（STAS，9.0+） | — |

各机制的详细说明见[使用 LDGSTS](../04-cuda-features/asynchronous-data-copies.html#ldgsts)、[使用 Tensor Memory Accelerator（TMA）](../04-cuda-features/asynchronous-data-copies.html#using-the-tensor-memory-accelerator-tma)和[使用 STAS](../04-cuda-features/asynchronous-data-copies.html#stas-redas)。

## 3.2.6 配置 L1/Shared Memory 平衡（Configuring L1/Shared Memory Balance）

正如[第 2.3 节的 L1 data cache](../02-programming-gpus/writing-simt-kernels.html)所述，SM 上的 L1 与 shared memory 使用同一物理资源，这个资源称为 unified data cache。在大多数架构上，如果 kernel 很少使用或完全不使用 shared memory，那么可以把 unified data cache 配置为该架构允许的最大 L1 cache 大小。

为 shared memory 预留的 unified data cache 容量可以按 kernel 配置。应用可以在 kernel launch 前调用 [`cudaFuncSetAttribute`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g317e77d2657abf915fd9ed03e75f3eb0)，把 `carveout`（首选的 shared memory 容量）设置为架构支持的最大 shared memory 容量的整数百分比：

```cpp
cudaFuncSetAttribute(kernel_name,
                     cudaFuncAttributePreferredSharedMemoryCarveout,
                     carveout);
```

除了整数百分比，还提供了三个方便使用的 carveout 枚举值：

- `cudaSharedmemCarveoutDefault`
- `cudaSharedmemCarveoutMaxL1`
- `cudaSharedmemCarveoutMaxShared`

每种架构支持的最大 shared memory 和 carveout 大小不同，详见[各 compute capability 的 Shared Memory Capacity](../05-technical-appendices/compute-capabilities.html)。如果所选整数百分比不能精确对应某个受支持的 shared memory 容量，系统会使用下一个更大的容量。例如，compute capability 12.0 设备的最大 shared memory 容量为 100 KB，支持的大小为 0、8、16、32、64 和 100 KB；把 carveout 设置为 50% 时，实际使用的是 64 KB，而不是 50 KB。

传给 `cudaFuncSetAttribute` 的函数必须使用 `__global__` 说明符声明。`cudaFuncSetAttribute` 对 driver 来说是一个提示；如果执行 kernel 需要，driver 可以选择不同的 carveout 大小。

> **注意：** 另一个 CUDA API `cudaFuncSetCacheConfig` 也允许应用调整 kernel 的 L1/shared memory 平衡。但是该 API 对 shared/L1 平衡设置的是硬性要求。因此，交错执行具有不同 shared memory 配置的 kernel，可能会因为重新配置 shared memory 而不必要地串行化 launch。相比之下，推荐使用 `cudaFuncSetAttribute`，因为 driver 可以在执行函数所需或避免频繁切换配置时选择不同配置。

依赖每个 block 使用超过 48 KB shared memory 的 kernel 与具体架构相关。因此，这些 kernel 必须使用[动态 shared memory](../02-programming-gpus/writing-simt-kernels.html)，而不能使用静态大小数组，并且必须显式调用 `cudaFuncSetAttribute` opt-in：

```cpp
// Device code
__global__ void MyKernel(...)
{
  extern __shared__ float buffer[];
  ...
}

// Host code
int maxbytes = 98304; // 96 KB
cudaFuncSetAttribute(MyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
MyKernel<<<gridDim, blockDim, maxbytes>>>(...);
```
