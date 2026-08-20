---
title: 4.11 异步数据复制
description: LDGSTS、Tensor Memory Accelerator（TMA）和 STAS 的异步数据移动方法
---

# 4.11 异步数据复制（Asynchronous Data Copies）

本节在[第 3.2.5 节](../03-advanced-cuda/advanced-kernel-programming.html)的基础上，详细介绍 GPU 内存层次结构中的异步数据移动，并给出示例。内容包括用于逐元素复制的 LDGSTS、用于批量（一维和多维）传输的 Tensor Memory Accelerator（TMA），以及用于将寄存器复制到分布式共享内存的 STAS；同时说明这些机制如何与[异步 barrier](asynchronous-barriers.html)和[pipeline](pipelines.html)结合使用。

## 4.11.1 使用 LDGSTS（Using LDGSTS）

许多 CUDA 应用需要频繁地在全局内存和共享内存之间移动数据。这通常涉及复制较小的数据元素，或执行不规则的内存访问。LDGSTS（计算能力 8.0 及更高版本，参见 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy)）的主要目标，是为较小的逐元素数据传输提供高效的、从全局内存到共享内存的异步传输机制，并通过重叠执行提高计算资源利用率。

**维度（Dimensions）**。LDGSTS 支持复制 4、8 或 16 字节。复制 4 或 8 字节时始终使用所谓的 L1 ACCESS 模式，数据也会缓存到 L1；复制 16 字节时启用 L1 BYPASS 模式，不会污染 L1。

**源与目标（Source and destination）**。LDGSTS 异步复制只支持从全局内存到共享内存。根据复制数据的大小，指针必须分别按 4、8 或 16 字节对齐。当共享内存和全局内存都按 128 字节对齐时，可以获得最佳性能。

**异步性（Asynchronicity）**。使用 LDGSTS 的数据传输是[异步的](../03-advanced-cuda/advanced-kernel-programming.html)，被建模为异步线程操作（参见 Async Thread 和 Async Proxy）。这允许发起线程继续计算，同时由硬件异步复制数据。*实际数据传输是否异步由硬件实现决定，未来可能发生变化*。

LDGSTS 必须在操作完成时提供信号。它可以使用共享内存 barrier 或 pipeline 作为完成信号机制。默认情况下，每个线程只等待自己的 LDGSTS 复制。因此，如果使用 LDGSTS 预取将由其他线程共享的数据，在与 LDGSTS 完成机制同步后，还必须执行 `__syncthreads()`。

**表 18：使用 LDGSTS 时可能的源/目标内存空间及完成机制。空单元表示该源-目标组合不受支持。**

| 源 | 目标 | 完成机制 | API |
| --- | --- | --- | --- |
| global | global |  |  |
| shared::cta | global |  |  |
| global | shared::cta | shared memory barrier、pipeline | `cuda::memcpy_async`、`cooperative_groups::memcpy_async`、`__pipeline_memcpy_async` |
| global | shared::cluster |  |  |
| shared::cluster | shared::cta |  |  |
| shared::cta | shared::cta |  |  |

下面通过示例演示 LDGSTS 的用法，并解释不同 API 之间的差异。

### 4.11.1.1 在条件代码中批量加载（Batching Loads in Conditional Code）

在下面的 stencil 示例中，thread block 的第一个 warp 负责协作加载所需的全部数据，包括中心区域以及左、右 halo。对于同步复制，由于代码包含条件分支，编译器可能生成一系列“从全局加载（LDG）、写入共享（STS）”的指令，而不是先执行 3 个 LDG 再执行 3 个 STS。后者才是隐藏全局内存延迟的最佳加载方式。

```cpp
__global__ void stencil_kernel(const float *left, const float *center, const float *right)
{
    // 左 halo（8 个元素）- 中心（32 个元素）- 右 halo（8 个元素）
    __shared__ float buffer[8 + 32 + 8];
    const int tid = threadIdx.x;

    if (tid < 8) {
        buffer[tid] = left[tid]; // 左 halo
    } else if (tid >= 32 - 8) {
        buffer[tid + 16] = right[tid]; // 右 halo
    }
    if (tid < 32) {
        buffer[tid + 8] = center[tid]; // 中心
    }
    __syncthreads();

    // 计算 stencil
}
```

为了确保数据以最佳方式加载，可以把同步内存复制替换为异步复制，直接将数据从全局内存加载到共享内存。这不仅通过直接写入共享内存减少寄存器使用，还能确保所有全局内存加载都处于进行中。

**CUDA C++：`cuda::memcpy_async`**

```cpp
#include <cooperative_groups.h>
#include <cuda/barrier>

__global__ void stencil_kernel(const float *left, const float *center, const float *right)
{
    auto block = cooperative_groups::this_thread_block();
    auto thread = cooperative_groups::this_thread();
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
    __shared__ barrier_t barrier;
    __shared__ float buffer[8 + 32 + 8];

    if (block.thread_rank() == 0) {
        init(&barrier, block.size());
    }
    __syncthreads();

    // 版本 1：由各个线程单独发起复制。
    if (tid < 8) {
        cuda::memcpy_async(buffer + tid, left + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier);
        // 或：cuda::memcpy_async(thread, buffer + tid, left + tid,
        //       cuda::aligned_size_t<4>(sizeof(float)), barrier);
    } else if (tid >= 32 - 8) {
        cuda::memcpy_async(buffer + tid + 16, right + tid,
                           cuda::aligned_size_t<4>(sizeof(float)), barrier);
        // 或：cuda::memcpy_async(thread, buffer + tid + 16, right + tid,
        //       cuda::aligned_size_t<4>(sizeof(float)), barrier);
    }
    if (tid < 32) {
        cuda::memcpy_async(buffer + tid + 8, center + tid,
                           cuda::aligned_size_t<4>(sizeof(float)), barrier);
        // 或：cuda::memcpy_async(thread, buffer + tid + 8, center + tid,
        //       cuda::aligned_size_t<4>(sizeof(float)), barrier);
    }

    // 版本 2：在所有线程之间协作发起复制。
    cuda::memcpy_async(block, buffer, left,
                       cuda::aligned_size_t<4>(8 * sizeof(float)), barrier);
    cuda::memcpy_async(block, buffer + 8, center,
                       cuda::aligned_size_t<4>(32 * sizeof(float)), barrier);
    cuda::memcpy_async(block, buffer + 40, right,
                       cuda::aligned_size_t<4>(8 * sizeof(float)), barrier);

    barrier.arrive_and_wait();
    __syncthreads();

    // 计算 stencil
}
```

**CUDA C++：`cooperative_groups::memcpy_async`**

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

__global__ void stencil_kernel(const float *left, const float *center, const float *right)
{
    cg::thread_block block = cg::this_thread_block();
    __shared__ float buffer[8 + 32 + 8];

    cg::memcpy_async(block, buffer, left, 8 * sizeof(float));
    cg::memcpy_async(block, buffer + 8, center, 32 * sizeof(float));
    cg::memcpy_async(block, buffer + 40, right, 8 * sizeof(float));
    cg::wait(block);
    __syncthreads();

    // 计算 stencil。
}
```

**CUDA C 原语**

```cpp
#include <cuda_pipeline.h>

__global__ void stencil_kernel(const float *left, const float *center, const float *right)
{
    __shared__ float buffer[8 + 32 + 8];
    const int tid = threadIdx.x;

    if (tid < 8) {
        __pipeline_memcpy_async(buffer + tid, left + tid, sizeof(float));
    } else if (tid >= 32 - 8) {
        __pipeline_memcpy_async(buffer + tid + 16, right + tid, sizeof(float));
    }
    if (tid < 32) {
        __pipeline_memcpy_async(buffer + tid + 8, center + tid, sizeof(float));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // 计算 stencil。
}
```

`cuda::barrier` 的 `cuda::memcpy_async` 重载可以使用异步 barrier 同步异步数据传输。该重载把复制操作视为由绑定到 barrier 的另一个线程执行：创建复制时增加当前阶段的预期计数，复制完成时减少该计数。因此，只有当参与 barrier 的所有线程都已到达，且绑定到当前阶段的所有 `memcpy_async` 都已完成时，`barrier` 才会进入下一阶段。这里使用 block 级 barrier，block 中的所有线程都参与，并使用 `arrive_and_wait` 合并到达和等待，因为两个阶段之间没有需要执行的工作。

线程级复制（版本 1）和协作复制（版本 2）都可以获得相同结果。版本 2 的 API 会自动处理底层复制方式。在两个版本中，都使用 `cuda::aligned_size_t<4>()` 告知编译器数据按 4 字节对齐且复制大小是 4 的倍数，从而允许使用 LDGSTS。为了与 `cuda::barrier` 互操作，这里使用来自 `cuda/barrier` header 的 `cuda::memcpy_async`。

`cooperative_groups::memcpy_async` 会协调 block 中所有线程的内存传输，但使用 `cg::wait(block)` 同步完成，而不是显式操作 barrier。

基于底层原语的实现使用 `__pipeline_memcpy_async()` 发起逐元素传输，使用 `__pipeline_commit()` 提交复制批次，并使用 `__pipeline_wait_prior(0)` 等待 pipeline 中的所有操作完成。这种方式提供最直接的控制，但代码比高级 API 更冗长；它还确保底层会使用 LDGSTS，而高级 API 不保证这一点。

::: note
`cooperative_groups::memcpy_async` 在此示例中比其他 API 效率低，因为它会在每个复制操作启动后立即自动提交，阻止了其他 API 可以实现的“在单次提交前批量提交多个复制”的优化。
:::

### 4.11.1.2 预取数据（Prefetching Data）

本示例展示如何使用异步数据复制，将数据从全局内存预取到共享内存。在迭代式复制-计算模式中，这可以用当前迭代的计算隐藏未来迭代的数据传输延迟，并可能增加在途字节数。

**CUDA C++：`cuda::memcpy_async`**

```cpp
#include <cooperative_groups.h>
#include <cuda/pipeline>

template <size_t num_stages = 2>
__global__ void prefetch_kernel(int* global_out, int const* global_in,
                                size_t size, size_t batch_size) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(size == batch_size * grid.size());

    extern __shared__ int shared[];
    size_t shared_offset[num_stages];
    for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();

    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
    auto block_batch = [&](size_t batch) -> int {
        return block.group_index().x * block.size() + grid.size() * batch;
    };

    for (int s = 0; s < num_stages; ++s) {
        pipeline.producer_acquire();
        cuda::memcpy_async(shared + shared_offset[s] + tid,
                           global_in + block_batch(s) + tid,
                           cuda::aligned_size_t<4>(sizeof(int)), pipeline);
        pipeline.producer_commit();
    }

    int stage = 0;
    for (size_t compute_batch = 0, fetch_batch = num_stages;
         compute_batch < batch_size; ++compute_batch, ++fetch_batch) {
        constexpr size_t pending_batches = num_stages - 1;
        cuda::pipeline_consumer_wait_prior<pending_batches>(pipeline);
        __syncthreads();
        compute(global_out + block_batch(compute_batch) + tid,
                shared + shared_offset[stage] + tid);
        pipeline.consumer_release();
        __syncthreads();
        pipeline.producer_acquire();
        if (fetch_batch < batch_size) {
            cuda::memcpy_async(shared + shared_offset[stage] + tid,
                               global_in + block_batch(fetch_batch) + tid,
                               cuda::aligned_size_t<4>(sizeof(int)), pipeline);
        }
        pipeline.producer_commit();
        stage = (stage + 1) % num_stages;
    }
}
```

**CUDA C++：`cooperative_groups::memcpy_async`**

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

template <size_t num_stages = 2>
__global__ void prefetch_kernel(int* global_out, int const* global_in,
                                size_t size, size_t batch_size) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    assert(size == batch_size * grid.size());
    extern __shared__ int shared[];
    size_t shared_offset[num_stages];
    for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();

    for (int s = 0; s < num_stages; ++s) {
        cg::memcpy_async(block, shared + shared_offset[s],
                         global_in + block_batch(s),
                         cuda::aligned_size_t<4>(sizeof(int)) * block.size());
    }
    int stage = 0;
    for (size_t compute_batch = 0, fetch_batch = num_stages;
         compute_batch < batch_size; ++compute_batch, ++fetch_batch) {
        size_t pending_batches =
            (fetch_batch < batch_size - num_stages) ? num_stages - 1
                                                   : batch_size - fetch_batch - 1;
        cg::wait_prior(pending_batches);
        __syncthreads();
        compute(global_out + block_batch(compute_batch) + tid,
                shared + shared_offset[stage] + tid);
        __syncthreads();
        if (fetch_batch < batch_size) {
            cg::memcpy_async(block, shared + shared_offset[stage],
                             global_in + block_batch(fetch_batch),
                             cuda::aligned_size_t<4>(sizeof(int)) * block.size());
        }
        stage = (stage + 1) % num_stages;
    }
}
```

**CUDA C 原语**

```cpp
#include <cooperative_groups.h>
#include <cuda_awbarrier_primitives.h>

template <size_t num_stages = 2>
__global__ void prefetch_kernel(int* global_out, int const* global_in,
                                size_t size, size_t batch_size) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    extern __shared__ int shared[];
    size_t shared_offset[num_stages];
    for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();

    auto block_batch = [&](size_t batch) -> int {
        return block.group_index().x * block.size() + grid.size() * batch;
    };
    for (int s = 0; s < num_stages; ++s) {
        __pipeline_memcpy_async(shared + shared_offset[s] + tid,
                                global_in + block_batch(s) + tid,
                                cuda::aligned_size_t<4>(sizeof(int)));
        __pipeline_commit();
    }
    int stage = 0;
    for (size_t compute_batch = 0, fetch_batch = num_stages;
         compute_batch < batch_size; ++compute_batch, ++fetch_batch) {
        constexpr size_t pending_batches = num_stages - 1;
        __pipeline_wait_prior<pending_batches>();
        __syncthreads();
        compute(global_out + block_batch(compute_batch) + tid,
                shared + shared_offset[stage] + tid);
        __syncthreads();
        if (fetch_batch < batch_size) {
            __pipeline_memcpy_async(shared + shared_offset[stage] + tid,
                                    global_in + block_batch(fetch_batch) + tid,
                                    cuda::aligned_size_t<4>(sizeof(int)));
        }
        __pipeline_commit();
        stage = (stage + 1) % num_stages;
    }
}
```

`cuda::memcpy_async` 的实现使用线程本地 `cuda::pipeline` 执行多阶段预取：先安排 `num_stages` 个复制操作填充 pipeline，然后遍历所有 batch，等待当前 batch 完成、执行计算，并在存在后续 batch 时安排下一次复制。`cooperative_groups::memcpy_async` 的实现不显式使用 pipeline 对象，而是依赖 API 在底层分阶段安排传输。CUDA C 原语实现以类似方式使用低级原语。

为了生成高效代码，即使已经没有更多 batch 要取，也要让 pipeline 保持 `num_stages` 个 batch。这通过在没有更多数据时仍提交 pipeline（`pipeline.producer_commit()` 或 `__pipeline_commit()`）实现。由于无法访问内部 pipeline，cooperative groups API 无法实现这一点。

### 4.11.1.3 通过 Warp Specialization 实现生产者-消费者模式（Producer-Consumer Pattern Through Warp Specialization）

本示例展示如何实现生产者-消费者模式：一个 warp 专门充当生产者，执行从全局内存到共享内存的异步数据复制，其余 warp 从共享内存消费数据并执行计算。为了让生产者线程与消费者线程并发，使用共享内存双缓冲。当消费者 warp 处理一个 buffer 时，生产者 warp 异步将下一批数据取入另一个 buffer。

**CUDA C++：`cuda::memcpy_async`**

```cpp
#include <cooperative_groups.h>
#include <cuda/pipeline>

using pipeline = cuda::pipeline<cuda::thread_scope_block>;

__device__ void produce(pipeline &pipe, int num_stages, int stage,
                        int num_batches, int batch, float *buffer,
                        int buffer_len, float *in, int N) {
  if (batch < num_batches) {
    pipe.producer_acquire();
    cuda::memcpy_async(buffer + stage * buffer_len + threadIdx.x,
                       in + batch * buffer_len + threadIdx.x,
                       cuda::aligned_size_t<4>(sizeof(float)), pipe);
    pipe.producer_commit();
  }
}

__device__ void consume(pipeline &pipe, int num_stages, int stage,
                        int num_batches, int batch, float *buffer,
                        int buffer_len, float *out, int N) {
  pipe.consumer_wait();
  // 消费 buffer(stage) 并更新 out(batch)
  pipe.consumer_release();
}

__global__ void producer_consumer_pattern(float *in, float *out,
                                          int N, int buffer_len) {
  auto block = cooperative_groups::this_thread_block();
  constexpr int warpSize = 32;
  __shared__ extern float buffer[]; // 大小为 2 * buffer_len
  const int num_batches = N / buffer_len;

  constexpr auto scope = cuda::thread_scope_block;
  constexpr int num_stages = 2;
  cuda::std::size_t producer_count = warpSize;
  __shared__ cuda::pipeline_shared_state<scope, num_stages> shared_state;
  pipeline pipe = cuda::make_pipeline(block, &shared_state, producer_count);

  if (block.thread_rank() < producer_count)
    for (int s = 0; s < num_stages; ++s)
      produce(pipe, num_stages, s, num_batches, s, buffer, buffer_len, in, N);

  int stage = 0;
  for (size_t b = 0; b < num_batches; ++b) {
    if (block.thread_rank() < producer_count)
      produce(pipe, num_stages, stage, num_batches, b + num_stages,
              buffer, buffer_len, in, N);
    else
      consume(pipe, num_stages, stage, num_batches, b,
              buffer, buffer_len, out, N);
    stage = (stage + 1) % num_stages;
  }
}
```

**CUDA C 原语**

```cpp
#include <cooperative_groups.h>
#include <cuda_awbarrier_primitives.h>

__device__ void produce(__mbarrier_t ready[], __mbarrier_t filled[],
                        float *buffer, int buffer_len, float *in, int N) {
  for (int i = 0; i < N / buffer_len; ++i) {
    __mbarrier_token_t token = __mbarrier_arrive(&ready[i % 2]);
    while (!__mbarrier_try_wait(&ready[i % 2], token, 1000)) {}
    __pipeline_memcpy_async(buffer + i * buffer_len + threadIdx.x,
                            in + i * buffer_len + threadIdx.x,
                            cuda::aligned_size_t<4>(sizeof(float)));
    __pipeline_arrive_on(filled[i % 2]);
    __mbarrier_arrive(filled[i % 2]);
  }
}

__device__ void consume(__mbarrier_t ready[], __mbarrier_t filled[],
                        float *buffer, int buffer_len, float *out, int N) {
  __mbarrier_arrive(&ready[0]);
  __mbarrier_arrive(&ready[1]);
  for (int i = 0; i < N / buffer_len; ++i) {
    __mbarrier_token_t token = __mbarrier_arrive(&filled[i % 2]);
    while (!__mbarrier_try_wait(&filled[i % 2], token, 1000)) {}
    __mbarrier_arrive(&ready[i % 2]);
  }
}

__global__ void producer_consumer_pattern(int N, float *in, float *out,
                                          int buffer_len) {
  __shared__ extern float buffer[];
  __shared__ __mbarrier_t bar[4];
  auto block = cooperative_groups::this_thread_block();
  if (block.thread_rank() < 4)
    __mbarrier_init(bar + block.thread_rank(), block.size());
  __syncthreads();
  if (block.thread_rank() < warpSize)
    produce(bar, bar + 2, buffer, buffer_len, in, N);
  else
    consume(bar, bar + 2, buffer, buffer_len, out, N);
}
```

`cuda::memcpy_async` 实现使用抽象级别最高的 `cuda::memcpy_async` 和双阶段 `cuda::pipeline`。它使用分区 pipeline：第一个 warp 作为生产者，其余 warp 作为消费者。生产者先填充两个 pipeline 阶段；主循环中，消费者处理当前 batch 的同时，生产者获取未来 batch，从而维持稳定的工作流。

基于原语的 CUDA C 实现将 `__pipeline_memcpy_async()` 与共享内存 barrier 结合为完成机制。`__pipeline_arrive_on()` 将内存复制关联到 barrier：它使 barrier 到达计数增加 1；当它之前排序的所有异步操作完成时，到达计数自动减少 1，所以对计数的净影响为 0。因此还需要显式调用 `__mbarrier_arrive()` 到达 barrier。

## 4.11.2 使用 Tensor Memory Accelerator（TMA）（Using the Tensor Memory Accelerator）

许多应用需要从全局内存移动大量数据，数据通常以多维数组形式布局，并具有非连续的访问模式。为了减少全局内存访问，会在计算前把这类数组的子 tile 复制到共享内存。加载和存储涉及容易出错且重复的地址计算。计算能力 9.0（Hopper）及更高版本提供了 Tensor Memory Accelerator（TMA，参见 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk)）来卸载这些计算。TMA 的主要目标是为多维数组提供高效的全局内存到共享内存数据传输机制。

**命名（Naming）**。Tensor Memory Accelerator（TMA）是对本节功能的统称。为了前向兼容并减少与 PTX ISA 的差异，本节根据复制类型将 TMA 操作称为 *bulk-asynchronous copy* 或 *bulk-tensor asynchronous copy*。“bulk”用于与上一节描述的异步内存操作区分。

**维度（Dimensions）**。TMA 支持一维和多维（最多 5 维）数组复制。一维连续数组的 bulk-asynchronous copy 编程模型不同于多维数组的 bulk-tensor asynchronous copy。执行多维 bulk-tensor asynchronous copy 时，硬件需要一个 tensor map，用来描述多维数组在全局内存和共享内存中的布局。tensor map 通常由主机使用 `cuTensorMapEncode` API 创建，然后作为带有 `__grid_constant__` 标注的 `const` kernel 参数从主机传到设备。设备可以使用它在共享内存和全局内存之间复制 tile。相反，连续一维数组的 bulk-asynchronous copy 不需要 tensor map，可以在设备端通过指针和大小参数执行。

**源与目标（Source and destination）**。TMA 操作的源和目标地址可以位于共享内存或全局内存。操作可以从全局内存读到共享内存、从共享内存写到全局内存，也可以从共享内存复制到同一 cluster 中另一个 block 的分布式共享内存。在 cluster 中，还可以将 bulk-asynchronous tensor 操作指定为 *multicast*，把数据从全局内存传到 cluster 内多个 block 的共享内存。multicast 针对 `sm_90a` 优化，在其他目标架构上性能可能显著降低，因此建议只配合 `sm_90a` 使用。

**异步性（Asynchronicity）**。TMA 数据传输是异步的，并被建模为 async proxy 操作。发起线程可以继续计算，同时硬件异步复制数据。*实际数据传输是否异步由硬件实现决定，未来可能变化*。bulk-asynchronous 操作有多种完成机制。当操作从全局内存读取到共享内存时，block 中任意线程都可以通过等待共享内存 barrier 等待数据可读。当操作从共享内存写入全局内存或分布式共享内存时，只有发起线程可以等待操作完成，这通过 bulk async-group 完成机制实现。

**表 19：使用 TMA 时可能的源/目标内存空间及完成机制。**

| 源 | 目标 | 完成机制 |
| --- | --- | --- |
| global | global |  |
| shared::cta | global | bulk async-group |
| global | shared::cta | shared memory barrier |
| global | shared::cluster | shared memory barrier（multicast） |
| shared::cta | shared::cluster | shared memory barrier |
| shared::cta | shared::cta |  |

### 4.11.2.1 使用 TMA 传输一维数组（Using TMA to transfer one-dimensional arrays）

下表总结 bulk-asynchronous TMA 的源/目标内存空间、完成机制以及对应 API。

**表 20：使用 bulk-asynchronous TMA 时可能的源/目标内存空间及完成机制。**

| 源 | 目标 | 完成机制 | API |
| --- | --- | --- | --- |
| global | global |  |  |
| shared::cta | global | bulk async-group | `cuda::ptx::cp_async_bulk` |
| global | shared::cta | shared memory barrier | `cuda::memcpy_async`、`cuda::device::memcpy_async_tx`、`cuda::ptx::cp_async_bulk` |
| global | shared::cluster | shared memory barrier | `cuda::ptx::cp_async_bulk` |
| shared::cta | shared::cluster | shared memory barrier | `cuda::ptx::cp_async_bulk` |
| shared::cta | shared::cta |  |  |

某些功能需要 inline PTX，目前通过 CUDA Standard C++ 库 `cuda::ptx` 命名空间提供。可以用以下代码检查 wrapper 是否可用：

```cpp
#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Device code is being compiled with older architectures that are incompatible with TMA.");
#endif // __CUDA_MINIMUM_ARCH__
```

如果源地址和目标地址按 16 字节对齐且大小是 16 的倍数，`cuda::memcpy_async` 会使用 TMA；否则会回退到同步复制。另一方面，`cuda::device::memcpy_async_tx` 和 `cuda::ptx::cp_async_bulk` 始终使用 TMA，不满足要求时会产生未定义行为。

下面通过读-修改-写一维数组的示例演示 bulk-asynchronous copy。kernel 依次执行：初始化共享内存 barrier；发起全局到共享的复制；到达并等待 barrier；增加共享内存 buffer 中的值；使用 proxy fence 让共享内存写入对后续 bulk-asynchronous copy 可见；发起共享到全局的复制；等待复制完成读取共享内存。

```cpp
#include <cuda/barrier>
#include <cuda/ptx>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;
static constexpr size_t buf_len = 1024;

__device__ inline bool is_elected() {
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0);
    return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF));
}

__global__ void add_one_kernel(int* data, size_t offset) {
  __shared__ alignas(16) int smem_data[buf_len];
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) init(&bar, blockDim.x);
  __syncthreads();

  if (is_elected()) {
    cuda::memcpy_async(smem_data, data + offset,
                       cuda::aligned_size_t<16>(sizeof(smem_data)), bar);
    // 也可以使用 cuda::device::memcpy_async_tx 或 cuda::ptx::cp_async_bulk。
  }

  barrier::arrival_token token = bar.arrive();
  bar.wait(std::move(token));

  for (int i = threadIdx.x; i < buf_len; i += blockDim.x)
    smem_data[i] += 1;

  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();

  if (is_elected()) {
    ptx::cp_async_bulk(ptx::space_global, ptx::space_shared,
                       data + offset, smem_data, sizeof(smem_data));
    ptx::cp_async_bulk_commit_group();
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }
}
```

**Barrier 初始化**。barrier 使用参与 block 的线程数初始化，因此只有所有线程都到达后 barrier 才会翻转。共享内存 barrier 的详细说明见异步 barrier 的“跟踪异步内存操作”部分。

**TMA 读取**。bulk-asynchronous copy 指令让硬件把大块数据复制到共享内存，并在读取完成后更新共享内存 barrier 的 transaction count。通常，尽可能少地发起更大的复制可以获得最佳性能，因为硬件可以异步执行复制，不需要将它拆成更小的块。

发起 bulk-asynchronous copy 的线程还要告诉 barrier 预期到达多少 transaction；此处按字节计数。`cuda::memcpy_async` 会自动完成这一点，但 `cuda::device::memcpy_async_tx` 和 `cuda::ptx::cp_async_bulk` 不会，需要显式调用 `cuda::ptx::mbarrier_expect_tx`。如果多个线程更新 transaction count，预期计数是这些更新的总和。只有所有线程都已到达且所有字节都已到达，barrier 才会翻转；翻转后，线程和后续 bulk-asynchronous copy 都可以安全读取共享内存中的字节。

**Barrier 等待**。使用 `bar.wait()` 和 token 等待 barrier 翻转。也可以显式跟踪 barrier 阶段，以获得更高效率。

**共享内存写入与同步**。增加 buffer 值会读写共享内存。使用 `cuda::ptx::fence_proxy_async` 让这些写入在后续由 async proxy 读取之前完成排序。每个线程先通过 fence 排序自己对共享内存对象的写入，再通过 `__syncthreads()` 将所有线程的操作排在由线程 0 发起的异步操作之前。

**TMA 写入与同步**。共享到全局的写入再次由单个线程发起，不使用共享内存 barrier 跟踪，而使用线程本地机制。多个写入可以批量组成 bulk async-group；随后线程可以等待该组完成从共享内存读取，或者完成写入全局内存并使写入对发起线程可见。bulk 和非 bulk 复制使用不同 async-group，分别对应 `cp.async.bulk.wait_group` 和 `cp.async.wait_group`。

::: note
建议由 block 中的单个线程发起 TMA 操作。仅使用 `if (threadIdx.x == 0)` 似乎足够，但编译器无法验证确实只有一个线程发起复制，可能插入遍历所有活动线程的 peeling loop，导致 warp 串行化和性能下降。这里使用 `is_elected()`，通过 `cuda::ptx::elect_sync` 从 warp 0 选出一个编译器可识别的线程；也可以使用 `cooperative_groups::invoke_one` 达到相同效果。
:::

**表 21：一维 bulk-asynchronous 操作的对齐要求。**

| 地址/大小 | 对齐要求 |
| --- | --- |
| 全局内存地址 | 必须按 16 字节对齐 |
| 共享内存地址 | 必须按 16 字节对齐 |
| 共享内存 barrier 地址 | 必须按 8 字节对齐（`cuda::barrier` 保证） |
| 传输大小 | 必须是 16 字节的倍数 |

#### 4.11.2.1.1 预取数据（Prefetching Data）

本示例展示如何使用 TMA 将数据从全局内存预取到共享内存。在迭代式复制-计算模式中，可以用当前迭代的计算隐藏未来迭代的数据传输延迟。

```cpp
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/ptx>

namespace ptx = cuda::ptx;
namespace cg = cooperative_groups;

__device__ inline bool is_elected() {
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0);
    return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF));
}

template <int block_size, int num_stages>
__global__ void prefetch_kernel(int* global_out, int const* global_in,
                                size_t size, size_t batch_size) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    const int tid = threadIdx.x;
    extern __shared__ int shared[];
    size_t shared_offset[num_stages];
    for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();
    auto block_batch = [&](size_t batch) -> int {
        return block.group_index().x * block.size() + grid.size() * batch;
    };

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> bar[num_stages];
    if (tid == 0) {
        #pragma unroll num_stages
        for (int i = 0; i < num_stages; ++i) init(&bar[i], 1);
    }
    __syncthreads();

    if (is_elected()) {
        size_t num_bytes = block_size * sizeof(int);
        #pragma unroll num_stages
        for (int s = 0; s < num_stages; ++s) {
            cuda::device::memcpy_async_tx(&shared[shared_offset[s]],
                &global_in[block_batch(s)],
                cuda::aligned_size_t<16>(num_bytes), bar[s]);
            (void)cuda::device::barrier_arrive_tx(bar[s], 1, num_bytes);
        }
    }

    int stage = 0;
    uint32_t parity = 0;
    for (size_t compute_batch = 0, fetch_batch = num_stages;
         compute_batch < batch_size; ++compute_batch, ++fetch_batch) {
        while (!ptx::mbarrier_try_wait_parity(
                   ptx::sem_acquire, ptx::scope_cta,
                   cuda::device::barrier_native_handle(bar[stage]), parity)) {}
        compute(global_out + block_batch(compute_batch) + tid,
                shared + shared_offset[stage] + tid);
        __syncthreads();
        if (is_elected() && fetch_batch < batch_size) {
            size_t num_bytes = block_size * sizeof(int);
            cuda::device::memcpy_async_tx(&shared[shared_offset[stage]],
                &global_in[block_batch(fetch_batch)],
                cuda::aligned_size_t<16>(num_bytes), bar[stage]);
            (void)cuda::device::barrier_arrive_tx(bar[stage], 1, num_bytes);
        }
        stage++;
        if (stage == num_stages) { stage = 0; parity ^= 1; }
    }
}
```

该示例使用 `cuda::device::memcpy_async_tx` 执行 TMA 复制，并使用带显式阶段跟踪的共享内存 barrier 同步：初始化阶段为每个阶段建立 barrier，并将前 `num_stages` 个 batch 预加载到不同共享内存区域；主循环依次等待当前 batch、计算、预取未来 batch，并循环管理阶段和 barrier parity。

### 4.11.2.2 使用 TMA 传输多维数组（Using TMA to transfer multi-dimensional arrays）

本节聚焦多维 TMA 复制。一维和多维情况的主要区别是：必须在主机上创建 tensor map，并将其传递给 CUDA kernel。

**表 22：使用 bulk-tensor asynchronous TMA 时可能的源/目标内存空间及完成机制。**

| 源 | 目标 | 完成机制 | API |
| --- | --- | --- | --- |
| global | global |  |  |
| shared::cta | global | bulk async-group | `cuda::ptx::cp_async_bulk_tensor` |
| global | shared::cta | shared memory barrier | `cuda::ptx::cp_async_bulk_tensor` |
| global | shared::cluster | shared memory barrier | `cuda::ptx::cp_async_bulk_tensor` |
| shared::cta | shared::cluster | shared memory barrier | `cuda::ptx::cp_async_bulk_tensor` |
| shared::cta | shared::cta |  |  |

所有功能都需要 inline PTX，目前由 CUDA Standard C++ 库的 `cuda::ptx` 命名空间提供。下面介绍如何使用 CUDA Driver API 创建 tensor map、把它传到设备端，并在设备端使用它。

**Driver API**。使用 `cuTensorMapEncodeTiled` Driver API 创建 tensor map。可以直接链接 driver（`-lcuda`），也可以使用 `cudaGetDriverEntryPointByVersion` API。下面是获取 `cuTensorMapEncodeTiled` 函数指针的示例：

```cpp
#include <cudaTypedefs.h>

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  CUDA_CHECK(cudaGetDriverEntryPointByVersion(
      "cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000,
      cudaEnableDefault, &driver_status));
  assert(driver_status == cudaDriverEntryPointSuccess);
  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}
```

**创建（Creation）**。创建 tensor map 需要许多参数，包括全局内存数组的基地址、数组大小（元素数）、从一行到下一行的 stride（字节数），以及共享内存 buffer 的大小（元素数）。下面创建一个 `GMEM_HEIGHT x GMEM_WIDTH` 的二维 row-major 数组描述符。参数顺序按最快变化维度在前排列。

```cpp
CUtensorMap tensor_map{};
constexpr uint32_t rank = 2;
uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
uint32_t elem_stride[rank] = {1, 1};

auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
CUresult res = cuTensorMapEncodeTiled(
  &tensor_map,
  CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
  rank, tensor_ptr, size, stride, box_size, elem_stride,
  CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
  CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
  CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
  CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
```

**主机到设备传输（Host-to-device transfer）**。有三种方式让设备代码访问 tensor map。推荐方法是把 tensor map 作为 `const __grid_constant__` 参数传给 kernel。其他方法包括使用 `cudaMemcpyToSymbol` 将 tensor map 复制到设备 `__constant__` 内存，或通过全局内存访问它。某些 GCC C++ 编译器版本会在传递 64 字节对齐参数时发出 ABI 警告，该警告可以忽略。

```cpp
#include <cuda.h>

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map) {
   // 在这里使用 tensor_map。
}
int main() {
  CUtensorMap map;
  // 初始化 map
  kernel<<<1, 1>>>(map);
}
```

也可以使用全局 `__constant__` 变量：

```cpp
#include <cuda.h>

__constant__ CUtensorMap global_tensor_map;
__global__ void kernel() {
  // 在这里使用 global_tensor_map。
}
int main() {
  CUtensorMap local_tensor_map;
  // 初始化 map
  cudaMemcpyToSymbol(global_tensor_map, &local_tensor_map, sizeof(CUtensorMap));
  kernel<<<1, 1>>>();
}
```

最后，也可以把 tensor map 复制到全局内存。使用位于设备全局内存的 tensor map 指针时，任何 thread block 在使用更新后的 tensor map 之前，都必须执行 fence；除非 tensor map 再次被修改，否则同一 block 后续使用它不需要重复 fence。此方法可能比前两种方式慢。

```cpp
#include <cuda.h>
#include <cuda/ptx>
namespace ptx = cuda::ptx;

__device__ CUtensorMap global_tensor_map;
__global__ void kernel(CUtensorMap *tensor_map) {
  ptx::n32_t<128> size_bytes;
  ptx::fence_proxy_tensormap_generic(
      ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes);
  // fence 后可以安全使用 tensor_map。
}
```

**使用（Use）**。下面的 kernel 从较大的二维数组加载一个大小为 `SMEM_HEIGHT x SMEM_WIDTH` 的二维 tile。tile 左上角由 `x` 和 `y` 指定；tile 被加载到共享内存，修改后再写回全局内存。

```cpp
#include <cuda.h>
#include <cuda/barrier>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

__device__ inline bool is_elected() {
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0);
    return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF));
}

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map,
                       int x, int y) {
  __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) init(&bar, blockDim.x);
  __syncthreads();

  barrier::arrival_token token;
  if (is_elected()) {
    int32_t tensor_coords[2] = {x, y};
    ptx::cp_async_bulk_tensor(
        ptx::space_shared, ptx::space_global, &smem_buffer,
        &tensor_map, tensor_coords, cuda::device::barrier_native_handle(bar));
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    token = bar.arrive();
  }
  bar.wait(std::move(token));

  smem_buffer[0][threadIdx.x] += threadIdx.x;
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();

  if (is_elected()) {
    int32_t tensor_coords[2] = {x, y};
    ptx::cp_async_bulk_tensor(
        ptx::space_global, ptx::space_shared, &tensor_map,
        tensor_coords, &smem_buffer);
    ptx::cp_async_bulk_commit_group();
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }
  if (threadIdx.x == 0) (&bar)->~barrier();
}
```

**负索引和越界（Negative indices and out of bounds）**。从全局内存读取到共享内存时，如果 tile 的一部分越界，对应的共享内存区域会填充为 0；tile 左上角索引也可以为负数。从共享内存写入全局内存时，tile 的部分区域可以越界，但左上角索引不能为负数。

**大小和 stride（Size and stride）**。tensor 的 size 是某一维上的元素数，所有 size 必须大于 1。stride 是同一维相邻元素之间的字节数。例如，4 x 4 的整数矩阵的 size 为 4 和 4；每个元素 4 字节，因此 stride 为 4 和 16 字节。由于对齐要求，4 x 3 的 row-major 整数矩阵也必须使用 4 和 16 字节的 stride；每行会填充额外 4 字节，以确保下一行起始地址按 16 字节对齐。

**表 23：多维 bulk-tensor 异步复制的对齐要求。**

| 地址/大小 | 对齐要求 |
| --- | --- |
| 全局内存地址 | 必须按 16 字节对齐 |
| 全局内存 size | 必须大于等于 1，不必是 16 字节的倍数 |
| 全局内存 stride | 必须是 16 字节的倍数 |
| 共享内存地址 | 必须按 128 字节对齐 |
| 共享内存 barrier 地址 | 必须按 8 字节对齐（`cuda::barrier` 保证） |
| 传输大小 | 必须是 16 字节的倍数 |

#### 4.11.2.2.1 在设备端编码 Tensor Map（Encoding a Tensor Map on Device）

前文介绍了如何使用 CUDA Driver API 在主机上创建 tensor map。本节介绍如何在设备端编码 tiled-type tensor map。当典型的 `const __grid_constant__` kernel 参数传递方式不合适时，这很有用，例如在单次 kernel 启动中处理大小不同的一批 tensor。

推荐模式如下：

1. 在主机上使用 Driver API 创建 tensor map“模板” `template_tensor_map`。
2. 在设备 kernel 中复制并修改 `template_tensor_map`，将副本存入全局内存，并正确执行 fence。
3. 在另一个 kernel 中使用 tensor map，并执行适当的 fence。

高层代码结构如下：

```cpp
// 初始化设备上下文
CUDA_CHECK(cudaDeviceSynchronize());

// 使用 cuTensorMapEncodeTiled Driver API 创建 tensor map 模板
CUtensorMap template_tensor_map = make_tensormap_template();

// 在全局内存中分配 tensor map 和 tensor
CUtensorMap* global_tensor_map;
CUDA_CHECK(cudaMalloc(&global_tensor_map, sizeof(CUtensorMap)));
char* global_buf;
CUDA_CHECK(cudaMalloc(&global_buf, 8 * 256));

// 使用数据填充全局 buffer
fill_global_buf<<<1, 1>>>(global_buf);

// 定义要在设备端创建的 tensor map 参数
tensormap_params p{};
p.global_address    = global_buf;
p.rank              = 2;
p.box_dim[0]        = 128;
p.box_dim[1]        = 4;
p.global_dim[0]     = 256;
p.global_dim[1]     = 8;
p.global_stride[0]  = 256;
p.element_stride[0] = 1;
p.element_stride[1] = 1;

// 在设备端编码 global_tensor_map
encode_tensor_map<<<1, 32>>>(template_tensor_map, p, global_tensor_map);

// 在另一个 kernel 中使用
consume_tensor_map<<<1, 1>>>(global_tensor_map);

// 检查错误
CUDA_CHECK(cudaDeviceSynchronize());
```

在后续示例中，`tensormap_params` 结构体包含要更新的字段值：

```cpp
struct tensormap_params {
  void* global_address;
  int rank;
  uint32_t box_dim[5];
  uint64_t global_dim[5];
  size_t global_stride[4];
  uint32_t element_stride[5];
};

// 初始化设备上下文、创建模板、分配 tensor map 和数据，然后：
encode_tensor_map<<<1, 32>>>(template_tensor_map, p, global_tensor_map);
consume_tensor_map<<<1, 1>>>(global_tensor_map);
CUDA_CHECK(cudaDeviceSynchronize());
```

#### 4.11.2.2.2 在设备端编码和修改 Tensor Map（Device-side Encoding and Modification of a Tensor Map）

在全局内存中编码 tensor map 的推荐流程如下：

1. 将已有的 `template_tensor_map` 传给 kernel。与在 `cp.async.bulk.tensor` 指令中使用 tensor map 的 kernel 不同，这里可以通过任意方式传递，例如全局内存指针、kernel 参数或 `__constant__` 变量。
2. 使用 `template_tensor_map` 的值复制初始化共享内存中的 tensor map。
3. 使用 `cuda::ptx::tensormap_replace` 函数修改共享内存中的 tensor map。这些函数封装 `tensormap.replace` PTX 指令，可以修改 tiled-type tensor map 的任意字段，包括基地址、大小和 stride。
4. 使用 `cuda::ptx::tensormap_copy_fenceproxy` 将修改后的 tensor map 从共享内存复制到全局内存，并执行所需 fence。

下面的 kernel 遵循上述步骤。为完整起见，它修改 tensor map 的所有字段；实际 kernel 通常只修改其中几个字段。

::: note
Tensor map 格式可能随时间变化。因此，`cuda::ptx::tensormap_replace` 函数及相应的 `tensormap.replace.tile` PTX 指令标记为 `sm_90a` 专用。使用时应通过 `nvcc -arch sm_90a ...` 编译。
:::

::: tip
在 `sm_90a` 上，共享内存中全零初始化的 buffer 也可以用作初始 tensor map 值，从而完全在设备端编码 tensor map，而不使用 Driver API 编码 `template_tensor_map`。
:::

::: note
设备端修改只支持 tiled-type tensor map；其他类型的 tensor map 不能在设备端修改。
:::

```cpp
#include <cuda/ptx>
namespace ptx = cuda::ptx;

__launch_bounds__(32)
__global__ void encode_tensor_map(
    const __grid_constant__ CUtensorMap template_tensor_map,
    tensormap_params p, CUtensorMap* out) {
  __shared__ alignas(128) CUtensorMap smem_tmap;
  if (threadIdx.x == 0) {
    smem_tmap = template_tensor_map;
    const auto space_shared = ptx::space_shared;
    ptx::tensormap_replace_global_address(space_shared, &smem_tmap, p.global_address);
    ptx::tensormap_replace_rank(space_shared, &smem_tmap, p.rank - 1);
    if (0 < p.rank) ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.box_dim[0]);
    if (1 < p.rank) ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.box_dim[1]);
    if (2 < p.rank) ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.box_dim[2]);
    if (3 < p.rank) ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.box_dim[3]);
    if (4 < p.rank) ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.box_dim[4]);
    if (0 < p.rank) ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, (uint32_t)p.global_dim[0]);
    if (1 < p.rank) ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, (uint32_t)p.global_dim[1]);
    if (2 < p.rank) ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, (uint32_t)p.global_dim[2]);
    if (3 < p.rank) ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, (uint32_t)p.global_dim[3]);
    if (4 < p.rank) ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, (uint32_t)p.global_dim[4]);
    if (1 < p.rank) ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.global_stride[0]);
    if (2 < p.rank) ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.global_stride[1]);
    if (3 < p.rank) ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.global_stride[2]);
    if (4 < p.rank) ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.global_stride[3]);
    if (0 < p.rank) ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.element_stride[0]);
    if (1 < p.rank) ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.element_stride[1]);
    if (2 < p.rank) ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.element_stride[2]);
    if (3 < p.rank) ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.element_stride[3]);
    if (4 < p.rank) ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.element_stride[4]);
    auto zero = ptx::n32_t<0>{};
    ptx::tensormap_replace_elemtype(space_shared, &smem_tmap, zero);
    ptx::tensormap_replace_interleave_layout(space_shared, &smem_tmap, zero);
    ptx::tensormap_replace_swizzle_mode(space_shared, &smem_tmap, zero);
    ptx::tensormap_replace_fill_mode(space_shared, &smem_tmap, zero);
  }
  __syncwarp();
  ptx::n32_t<128> bytes_128;
  ptx::tensormap_cp_fenceproxy(ptx::sem_release, ptx::scope_gpu,
                               out, &smem_tmap, bytes_128);
}
```

#### 4.11.2.2.3 使用已修改的 Tensor Map（Usage of a Modified Tensor Map）

与把 tensor map 作为 `const __grid_constant__` kernel 参数传递不同，在全局内存中使用 tensor map 时，必须在修改 tensor map 的线程和使用它的线程之间，显式建立 tensor map proxy 的 release-acquire 模式。

release 部分使用上一节的 `cuda::ptx::tensormap_cp_fenceproxy` 完成。acquire 部分使用 `cuda::ptx::fence_proxy_tensormap_generic`，它封装 `fence.proxy.tensormap::generic.acquire` 指令。同一设备上的线程使用 `.gpu` 作用域即可；不同设备上的线程必须使用 `.sys` 作用域。某个线程 acquire tensor map 后，同一 block 的其他线程在充分同步（例如 `__syncthreads()`）后可以使用它。使用 tensor map 的线程和执行 fence 的线程必须位于同一 block；不同 block、不同 cluster 或不同 kernel 之间的 cluster/grid 同步或 stream 顺序同步不足以建立 tensor map 更新的排序。

```cpp
__global__ void consume_tensor_map(CUtensorMap* tensor_map) {
  ptx::n32_t<128> size_bytes;
  ptx::fence_proxy_tensormap_generic(
      ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes);
  // fence 后可以安全使用 tensor_map。

  __shared__ uint64_t bar;
  __shared__ alignas(128) char smem_buf[4][128];
  if (threadIdx.x == 0) {
    ptx::mbarrier_init(&bar, 1);
    ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global,
                              smem_buf, tensor_map, {0, 0}, &bar);
    ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta,
                                   ptx::space_shared, &bar, sizeof(smem_buf));
  }
  const int parity = 0;
  while (!ptx::mbarrier_try_wait_parity(&bar, parity)) {}
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 128; ++i) {
      printf("%3d ", smem_buf[j][i]);
      if (i % 32 == 31) printf("\n");
    }
    printf("\n");
  }
}
```

#### 4.11.2.2.4 使用 Driver API 创建 Tensor Map 模板值（Creating a Template Tensor Map Value Using the Driver API）

下面的代码创建一个最小的 tiled-type tensor map，之后可以在设备端修改。

```cpp
CUtensorMap make_tensormap_template() {
  CUtensorMap template_tensor_map{};
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
  uint32_t dims_32 = 16;
  uint64_t dims_strides_64 = 16;
  uint32_t elem_strides = 1;
  CUresult res = cuTensorMapEncodeTiled(
    &template_tensor_map,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
    1, nullptr, &dims_strides_64, &dims_strides_64,
    &dims_32, &elem_strides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CU_CHECK(res);
  return template_tensor_map;
}
```

#### 4.11.2.2.5 共享内存 Bank Swizzling（Shared-Memory Bank Swizzling）

默认情况下，TMA engine 按照数据在全局内存中的布局顺序将数据加载到共享内存。但是，对于某些共享内存访问模式，这种布局可能不是最优的，并可能引起共享内存 bank conflict。为了提高性能、减少 bank conflict，可以应用“swizzle pattern”改变共享内存布局。

共享内存有 32 个 bank，连续的 32 位字映射到连续 bank；每个 bank 每个时钟周期的带宽为 32 位。当一次事务中多次使用同一个 bank 时，加载和存储会出现 bank conflict，从而降低带宽。

为了让用户代码以避免 bank conflict 的方式布局共享内存，TMA engine 可以在把数据写入共享内存前对其执行 swizzle，并在从共享内存复制回全局内存时执行 unswizzle。tensor map 中编码了使用的 swizzle mode。

**示例：矩阵转置**。数据按 row-major 存储在全局内存中，但希望在共享内存中按列访问，这会产生 bank conflict。使用 128 字节 swizzle mode 和新的共享内存索引后，可以消除冲突。示例加载一个 row-major 的 8x8 `int4` 矩阵；每 8 个线程从共享内存 buffer 加载一行并将其存储到单独转置 buffer 的一列，普通布局会在存储时产生 8 路 bank conflict。使用 `CU_TENSOR_MAP_SWIZZLE_128B` 后，布局匹配 128 字节行宽，使按行和按列访问时每次事务不再使用相同 bank。

![没有 swizzle 的共享内存布局](/images/chapter-04/swizzle-example1.png)

*图 51：没有 swizzle 时，共享内存索引等同于全局内存索引。每条加载指令读取一行并存储到转置 buffer 的一列；同一列的矩阵元素落在同一个 bank，存储必须串行化为 8 次事务，形成每列 8 路 bank conflict。*

![使用 CU_TENSOR_MAP_SWIZZLE_128B 的共享内存布局](/images/chapter-04/swizzle-example2.png)

*图 52：使用 `CU_TENSOR_MAP_SWIZZLE_128B` swizzle 后，一行被存储到一列，但行和列中的每个矩阵元素都来自不同 bank，因此没有 bank conflict。*

```cpp
__global__ void kernel_tma(const __grid_constant__ CUtensorMap tensor_map) {
   __shared__ alignas(1024) int4 smem_buffer[8][8];
   __shared__ alignas(1024) int4 smem_buffer_tr[8][8];
   #pragma nv_diag_suppress static_var_with_dynamic_init
   __shared__ barrier bar;
   if (threadIdx.x == 0) init(&bar, blockDim.x);
   __syncthreads();

   barrier::arrival_token token;
   if (is_elected()) {
     int32_t tensor_coords[2] = {0, 0};
     ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                               &smem_buffer, &tensor_map, tensor_coords,
                               cuda::device::barrier_native_handle(bar));
     token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
   } else token = bar.arrive();
   bar.wait(std::move(token));

   for (int sidx_j = threadIdx.x; sidx_j < 8; sidx_j += blockDim.x) {
      for (int sidx_i = 0; sidx_i < 8; ++sidx_i) {
         const int swiz_j_idx = (sidx_i % 8) ^ sidx_j;
         const int swiz_i_idx_tr = (sidx_j % 8) ^ sidx_i;
         smem_buffer_tr[sidx_j][swiz_i_idx_tr] = smem_buffer[sidx_i][swiz_j_idx];
      }
   }
   ptx::fence_proxy_async(ptx::space_shared);
   __syncthreads();
   if (is_elected()) {
       int32_t tensor_coords[2] = {x, y};
       ptx::cp_async_bulk_tensor(ptx::space_global, ptx::space_shared,
                                 &tensor_map, tensor_coords, &smem_buffer_tr);
       ptx::cp_async_bulk_commit_group();
       ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
   }
   if (threadIdx.x == 0) (&bar)->~barrier();
}
```

主机端需要使用 128 字节 swizzle 创建对应的 tensor map：

```cpp
void* tensor_ptr = d_data;
CUtensorMap tensor_map{};
constexpr uint32_t rank = 2;
uint64_t size[rank] = {4 * 8, 8};
uint64_t stride[rank - 1] = {8 * sizeof(int4)};
// 内层 shared-memory box 维度（字节数）等于 swizzle span。
uint32_t box_size[rank] = {4 * 8, 8};
uint32_t elem_stride[rank] = {1, 1};

CUresult res = cuTensorMapEncodeTiled(
    &tensor_map,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
    rank, tensor_ptr, size, stride, box_size, elem_stride,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

kernel_tma<<<1, 8>>>(tensor_map);
```

**备注（Remark）**。这个示例用于展示 swizzle；原样示例并不追求性能，也不能扩展到给定维度之外。

**解释（Explanation）**。传输过程中，TMA engine 按 swizzle pattern 重排数据。pattern 定义 swizzle 宽度内 16 字节块到四个 bank 组成的子组之间的映射。`CUtensorMapSwizzle` 有四种选项：none、32 字节、64 字节和 128 字节；共享内存 box 的内维度必须小于等于 swizzle pattern 的跨度。

**Swizzle 模式**。共有四种模式；它们定义 128 字节范围内 16 字节块到八个“四 bank 子组”的映射以及新的共享内存索引关系。

![TMA Swizzle 模式总览](/images/chapter-04/swizzle-pattern.png)

*图 53：TMA Swizzle 模式总览。*

**注意事项（Considerations）**：

- **全局内存对齐**：全局内存必须按 128 字节对齐。
- **共享内存对齐**：为简单起见，共享内存应按 swizzle pattern 重复所需的字节数对齐。如果未按该重复跨度对齐，swizzle pattern 与共享内存之间会产生偏移。
- **内维度**：共享内存 block 的内维度必须满足表 25 的大小要求，否则指令无效。如果 swizzle 宽度超过内维度，还必须分配足以容纳完整 swizzle 宽度的共享内存。
- **粒度**：swizzle 映射粒度固定为 16 字节，数据按 16 字节块组织和访问，设计内存布局和访问模式时必须考虑这一点。

**Swizzle Pattern 指针偏移计算**。使用 TMA 时共享内存必须按 128 字节对齐。偏移量公式和索引关系如下：

| Swizzle 模式 | 偏移公式 | 索引关系 |
| --- | --- | --- |
| `CU_TENSOR_MAP_SWIZZLE_128B` | `(reinterpret_cast<uintptr_t>(smem_ptr)/128)%8` | `smem[y][x] <-> smem[y][((y+offset)%8)^x]` |
| `CU_TENSOR_MAP_SWIZZLE_64B` | `(reinterpret_cast<uintptr_t>(smem_ptr)/128)%4` | `smem[y][x] <-> smem[y][((y+offset)%4)^x]` |
| `CU_TENSOR_MAP_SWIZZLE_32B` | `(reinterpret_cast<uintptr_t>(smem_ptr)/128)%2` | `smem[y][x] <-> smem[y][((y+offset)%2)^x]` |

在图 53 中，这个偏移表示初始行偏移，因此会在 swizzle 索引计算中加到行索引 `y` 上：

```cpp
data_t* smem_ptr = &smem[0][0];
int offset = (reinterpret_cast<uintptr_t>(smem_ptr)/128)%8;
smem[y][((y+offset)%8)^x] = ...;
```

**表 25：计算能力 9 的不同 swizzle pattern 的要求和属性。**

| Pattern | Swizzle 宽度 | Shared box 内维度 | 重复周期 | 共享内存对齐 | 全局内存对齐 |
| --- | --- | --- | --- | --- | --- |
| `CU_TENSOR_MAP_SWIZZLE_128B` | 128 字节 | <=128 字节 | 1024 字节 | 128 字节 | 128 字节 |
| `CU_TENSOR_MAP_SWIZZLE_64B` | 64 字节 | <=64 字节 | 512 字节 | 128 字节 | 128 字节 |
| `CU_TENSOR_MAP_SWIZZLE_32B` | 32 字节 | <=32 字节 | 256 字节 | 128 字节 | 128 字节 |
| `CU_TENSOR_MAP_SWIZZLE_NONE`（默认） |  |  |  | 128 字节 | 16 字节 |

## 4.11.3 使用 STAS（Using STAS）

使用 thread block cluster 的 CUDA 应用可能需要在 cluster 内的 thread block 之间移动小数据元素。STAS 指令（计算能力 9.0 及更高版本，参见 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-async)）支持直接从寄存器到分布式共享内存的异步数据复制。STAS 只通过 libcu++ 中较低级别的 `cuda::ptx::st_async` API 暴露。

**维度（Dimensions）**。STAS 支持复制 4、8 或 16 字节。

**源与目标（Source and destination）**。STAS 异步复制只支持从寄存器到分布式共享内存。根据复制数据大小，目标指针必须按 4、8 或 16 字节对齐。

**异步性（Asynchronicity）**。STAS 数据传输是异步的，并被建模为异步线程操作。发起线程可以继续计算，同时硬件异步复制数据；实际是否异步由硬件实现决定。STAS 可使用共享内存 barrier 作为完成信号机制。

下面的示例在 thread-block cluster 中实现生产者-消费者模式。kernel 创建一个环形通信 pipeline：8 个 thread block 排成环，每个 block 同时向序列中的下一个 block 生产数据，并从上一个 block 消费数据。

每个 thread block 需要两个共享内存 barrier：`filled` 用于通知消费者 block 数据已经复制到共享内存 buffer，`ready` 用于通知生产者 block 消费者的 buffer 已经可以重新填充。

```cpp
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/ptx>

__global__ __cluster_dims__(8, 1, 1) void producer_consumer_kernel() {
    using namespace cooperative_groups;
    using namespace cuda::device;
    using namespace cuda::ptx;
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
    auto cluster = this_cluster();
    __shared__ int buffer[BLOCK_SIZE];
    __shared__ barrier_t filled;
    __shared__ barrier_t ready;

    if (threadIdx.x == 0) {
        init(&filled, 1);
        init(&ready, BLOCK_SIZE);
    }
    cluster.sync();

    int rk = cluster.block_rank();
    int rk_next = (rk + 1) % 8;
    int rk_prev = (rk + 7) % 8;
    auto buffer_next = cluster.map_shared_rank(buffer, rk_next);
    auto bar_next = cluster.map_shared_rank(barrier_native_handle(filled), rk_next);
    auto bar_prev = cluster.map_shared_rank(barrier_native_handle(ready), rk_prev);

    int phase = 0;
    for (int it = 0; it < 1000; ++it) {
        st_async(&buffer_next[threadIdx.x], rk, bar_next);
        if (threadIdx.x == 0) {
            mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared,
                                      barrier_native_handle(filled), sizeof(buffer));
        }
        while (!mbarrier_try_wait_parity(barrier_native_handle(filled), phase, 1000)) {}
        int r = buffer[threadIdx.x];
        // 使用数据执行某些操作。
        mbarrier_arrive(sem_release, scope_cluster, space_cluster, bar_prev);
        while (!mbarrier_try_wait_parity(barrier_native_handle(ready), phase, 1000)) {}
        phase ^= 1;
    }
}
```

- 每个 block 的第一个线程初始化共享内存 barrier；`filled` 初始化为 1，`ready` 初始化为 block 中的线程数。
- 执行 cluster-wide 同步，确保所有 barrier 初始化完成后，线程才开始通信。
- 每个线程确定邻居的 rank，并据此映射远程共享内存 barrier 和要写入的远程共享内存 buffer。
- 每轮迭代中：
  1. 作为生产者，每个线程向右邻居发送数据。
  2. 作为消费者，线程 0 到达本地 `filled` barrier，并声明预期接收的字节数。
  3. 作为消费者，每个线程等待本地 `filled` barrier，直到左邻居的数据到达。
  4. 作为消费者，每个线程使用数据执行操作。
  5. 作为消费者，每个线程通知左邻居自己已经处理完数据。
  6. 作为生产者，每个线程等待本地 `ready` barrier，直到右邻居准备好接收新数据。

每个 barrier 都必须使用正确的 memory space。对于映射后的远程 barrier，使用 `space_cluster`；对于本地 barrier，使用 `space_shared`。
