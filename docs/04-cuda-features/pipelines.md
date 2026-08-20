---
title: 4.10 流水线
description: CUDA pipeline 的初始化、提交、消费、异步内存操作跟踪与生产者-消费者模式
---

# 4.10 流水线（Pipelines）

流水线在[高级同步原语](../03-advanced-cuda/advanced-apis-and-features.html)中介绍，是一种用于分阶段安排工作并协调多缓冲生产者—消费者模式的机制，常用于将计算与[异步数据复制](./asynchronous-data-copies.html)重叠。

本节主要介绍如何通过 `cuda::pipeline` API 使用流水线，并在适用处指向相关 primitive。

## 4.10.1 初始化（Initialization）

`cuda::pipeline` 可以在不同线程作用域创建。对于 `cuda::thread_scope_thread` 以外的作用域，需要 `cuda::pipeline_shared_state<scope, count>` 对象来协调参与线程。该状态封装了有限资源，使流水线最多可以处理 `count` 个并发阶段。

```cpp
// 在线程作用域创建流水线
constexpr auto scope = cuda::thread_scope_thread;
cuda::pipeline<scope> pipeline = cuda::make_pipeline();
```

```cpp
// 在线程块作用域创建流水线
constexpr auto scope = cuda::thread_scope_block;
constexpr auto stages_count = 2;
__shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
auto pipeline = cuda::make_pipeline(group, &shared_state);
```

流水线可以是*统一的（unified）*，也可以是*分区的（partitioned）*。在统一流水线中，所有参与线程既是生产者也是消费者。在分区流水线中，每个参与线程要么是生产者，要么是消费者，并且在流水线对象的生命周期内不能改变角色。线程本地流水线不能分区。要创建分区流水线，需要向 `cuda::make_pipeline()` 提供生产者数量或线程角色。

```cpp
// 在线程块作用域创建分区流水线，只有线程 0 是生产者
constexpr auto scope = cuda::thread_scope_block;
constexpr auto stages_count = 2;
__shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
auto thread_role = (group.thread_rank() == 0) ? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;
auto pipeline = cuda::make_pipeline(group, &shared_state, thread_role);
```

为支持分区，共享的 `cuda::pipeline` 会产生额外开销，包括每个阶段使用一组共享内存 barrier 进行同步。即使流水线是统一的、原本可以使用 `__syncthreads()`，这些 barrier 仍会被使用。因此，在可能的情况下，最好使用能够避免这些开销的线程本地流水线。

## 4.10.2 提交工作（Submitting Work）

向流水线阶段提交工作包括：

1. 使用 `pipeline.producer_acquire()`，由一组生产者线程共同获取流水线的*头部（head）*。
2. 向流水线头部提交异步操作，例如 `memcpy_async`。
3. 使用 `pipeline.producer_commit()`，由参与线程共同*提交（推进）*流水线头部。

如果所有资源都在使用，`pipeline.producer_acquire()` 会阻塞生产者线程，直到消费者线程释放下一个流水线阶段的资源。

## 4.10.3 消费工作（Consuming Work）

从先前已提交的阶段消费工作包括：

1. 共同等待该阶段完成。例如，一组消费者线程使用 `pipeline.consumer_wait()` 等待尾部（最旧）阶段。
2. 使用 `pipeline.consumer_release()` 共同*释放*该阶段。

使用 `cuda::pipeline<cuda:thread_scope_thread>` 时，还可以使用友元函数 `cuda::pipeline_consumer_wait_prior<N>()`，等待除最后 N 个阶段之外的所有阶段完成；这类似于 primitive API 中的 `__pipeline_wait_prior(N)`。

## 4.10.4 Warp 纠缠（Warp Entanglement）

同一 warp 中的 CUDA 线程共享流水线机制。这种共享会使 warp 内提交的操作序列相互纠缠，在某些情况下影响性能。

**提交（Commit）**。提交操作会合并：对于调用提交操作的所有收敛线程，流水线序列只递增一次，它们提交的操作会被批量放入同一阶段。如果 warp 完全收敛，序列递增 1，所有提交操作会被批量放在流水线的同一阶段；如果 warp 完全发散，序列递增 32，提交操作会分布到不同阶段。

- 令 *PB* 表示 warp 共享流水线中操作的*实际*序列：`PB = {BP0, BP1, BP2, …, BPL}`。
- 令 *TB* 表示某个线程感知到的操作*感知*序列，仿佛序列只因该线程调用提交操作而递增：`TB = {BT0, BT1, BT2, …, BTL}`。

> `pipeline::producer_commit()` 的返回值来自线程感知到的批次序列。

- 线程感知序列中的一个索引总是对应 warp 共享实际序列中一个相等或更大的索引。只有所有提交操作都由完全收敛的线程调用时，两个序列才相等：`BTn ≤ BPm`，其中 `n ≤ m`。

例如，当 warp 完全发散时：

- warp 共享流水线的实际序列为 `PB = {0, 1, 2, 3, ..., 31}`（`PL = 31`）。
- 该 warp 中每个线程的感知序列为：线程 0：`TB = {0}`（`TL = 0`）；线程 1：`TB = {0}`（`TL = 0`）；…；线程 31：`TB = {0}`（`TL = 0`）。

**等待（Wait）**。CUDA 线程调用 `pipeline::consumer_wait()` 或 `pipeline_consumer_wait_prior<N>()`，等待感知序列 `TB` 中的批次完成。注意，`pipeline::consumer_wait()` 等价于 `pipeline_consumer_wait_prior<N>()`，其中 `N = PL`。

`wait prior` 变体会等待实际序列中至少直到 `PL-N`（含）的批次。由于 `TL ≤ PL`，等待直到 `PL-N`（含）的批次也包括等待 `TL-N` 批次。因此，当 `TL < PL` 时，线程会无意中等待额外的、更新的批次。在上面完全发散的 warp 示例中，每个线程可能等待全部 32 个批次。

> **注意**
>
> 建议由收敛线程调用 commit，以避免过度等待，使线程感知到的批次序列与实际序列保持对齐。
>
> 如果这些操作之前的代码使线程发散，应在调用 commit 操作前通过 `__syncwarp` 重新收敛 warp。

## 4.10.5 提前退出（Early Exit）

当参与流水线的线程必须提前退出时，该线程必须在退出前使用 `cuda::pipeline::quit()` 显式退出参与。其余参与线程可以正常执行后续操作。

## 4.10.6 跟踪异步内存操作（Tracking Asynchronous Memory Operations）

下面的示例演示如何使用流水线跟踪异步内存复制，协同地将数据从 global memory 复制到 shared memory。每个线程使用自己的流水线，独立提交内存复制，随后等待复制完成并消费数据。关于异步数据复制的更多信息，请参见第 3.2.5 节。

### CUDA C++ `cuda::pipeline`

```cpp
#include <cuda/pipeline>

__global__ void example_kernel(const float *in)
{
    constexpr int block_size = 128;
    __shared__ __align__(sizeof(float)) float buffer[4 * block_size];

    // 每个线程创建一个统一流水线
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    // 内存复制的第一阶段
    pipeline.producer_acquire();
    // 每个线程获取第一个 block 中的一个元素
    cuda::memcpy_async(buffer, in, sizeof(float), pipeline);
    pipeline.producer_commit();

    // 内存复制的第二阶段
    pipeline.producer_acquire();
    // 每个线程获取第二和第三个 block 中的一个元素
    cuda::memcpy_async(buffer + block_size, in + block_size, sizeof(float), pipeline);
    cuda::memcpy_async(buffer + 2 * block_size, in + 2 * block_size, sizeof(float), pipeline);
    pipeline.producer_commit();

    // 内存复制的第三阶段
    pipeline.producer_acquire();
    // 每个线程获取最后一个 block 中的一个元素
    cuda::memcpy_async(buffer + 3 * block_size, in + 3 * block_size, sizeof(float), pipeline);
    pipeline.producer_commit();

    // 等待最旧阶段（等待第一阶段）
    pipeline.consumer_wait();
    pipeline.consumer_release();
    // __syncthreads();
    // 使用第一阶段的数据

    // 等待最旧阶段（等待第二阶段）
    pipeline.consumer_wait();
    pipeline.consumer_release();
    // __syncthreads();
    // 使用第二阶段的数据

    // 等待最旧阶段（等待第三阶段）
    pipeline.consumer_wait();
    pipeline.consumer_release();
    // __syncthreads();
    // 使用第三阶段的数据
}
```

### CUDA C primitive

```cpp
#include <cuda_pipeline.h>

__global__ void example_kernel(const float *in)
{
    constexpr int block_size = 128;
    __shared__ __align__(sizeof(float)) float buffer[4 * block_size];

    // 内存复制的第一批
    // 每个线程获取第一个 block 中的一个元素
    __pipeline_memcpy_async(buffer, in, sizeof(float));
    __pipeline_commit();

    // 内存复制的第二批
    // 每个线程获取第二和第三个 block 中的一个元素
    __pipeline_memcpy_async(buffer + block_size, in + block_size, sizeof(float));
    __pipeline_memcpy_async(buffer + 2 * block_size, in + 2 * block_size, sizeof(float));
    __pipeline_commit();

    // 内存复制的第三批
    // 每个线程获取最后一个 block 中的一个元素
    __pipeline_memcpy_async(buffer + 3 * block_size, in + 3 * block_size, sizeof(float));
    __pipeline_commit();

    // 等待除最后两批之外的所有内存复制（等待第一批）
    __pipeline_wait_prior(2);
    // __syncthreads();
    // 使用第一批数据

    // 等待除最后一批之外的所有内存复制（等待第二批）
    __pipeline_wait_prior(1);
    // __syncthreads();
    // 使用第二批数据

    // 等待所有内存复制（等待第三批）
    __pipeline_wait_prior(0);
    // __syncthreads();
    // 使用最后一批数据
}
```

## 4.10.7 使用流水线的生产者—消费者模式（Producer-Consumer Pattern using Pipelines）

在[第 4.9.7 节](./asynchronous-barriers.html#497-使用-barrier-的生产者消费者模式)中，我们展示了如何对线程块进行空间分区，使用[异步 barrier](./asynchronous-barriers.html)实现生产者—消费者模式。使用 `cuda::pipeline` 后，可以用一个分区流水线简化该模式：每个数据缓冲区对应一个阶段，而不再为每个缓冲区使用两个异步 barrier。

### CUDA C++ `cuda::pipeline`

```cpp
#include <cuda/pipeline>
#include <cooperative_groups.h>

#pragma nv_diag_suppress static_var_with_dynamic_init

using pipeline = cuda::pipeline<cuda::thread_scope_block>;

__device__ void produce(pipeline &pipe, int num_stages, int stage, int num_batches, int batch, float *buffer, int buffer_len, float *in, int N)
{
  if (batch < num_batches)
  {
    pipe.producer_acquire();
    /* 使用异步内存复制将数据从 in(batch) 复制到 buffer(stage) */
    pipe.producer_commit();
  }
}

__device__ void consume(pipeline &pipe, int num_stages, int stage, int num_batches, int batch, float *buffer, int buffer_len, float *out, int N)
{
  pipe.consumer_wait();
  /* 消费 buffer(stage) 并更新 out(batch) */
  pipe.consumer_release();
}

__global__ void producer_consumer_pattern(float *in, float *out, int N, int buffer_len)
{
  auto block = cooperative_groups::this_thread_block();

  /* 下面声明的共享内存 buffer 大小为 2 * buffer_len，
     因此可以交替使用两个 buffer。
     buffer_0 = buffer，buffer_1 = buffer + buffer_len */
  __shared__ extern float buffer[];

  const int num_batches = N / buffer_len;
  // 创建一个具有 2 个阶段的分区流水线，一半线程为生产者，另一半为消费者。
  constexpr auto scope = cuda::thread_scope_block;
  constexpr int num_stages = 2;
  cuda::std::size_t producer_count = block.size() / 2;
  __shared__ cuda::pipeline_shared_state<scope, num_stages> shared_state;
  pipeline pipe = cuda::make_pipeline(block, &shared_state, producer_count);

  // 填充流水线
  if (block.thread_rank() < producer_count)
  {
    for (int s = 0; s < num_stages; ++s)
    {
      produce(pipe, num_stages, s, num_batches, s, buffer, buffer_len, in, N);
    }
  }

  // 处理各批数据
  int stage = 0;
  for (size_t b = 0; b < num_batches; ++b)
  {
    if (block.thread_rank() < producer_count)
    {
      // 预取下一批
      produce(pipe, num_stages, stage, num_batches, b + num_stages, buffer, buffer_len, in, N);
    }
    else
    {
      // 消费最旧的一批
      consume(pipe, num_stages, stage, num_batches, b, buffer, buffer_len, out, N);
    }
    stage = (stage + 1) % num_stages;
  }
}
```

在这个示例中，线程块的一半线程作为生产者，另一半作为消费者。首先需要创建一个 `cuda::pipeline` 对象。由于需要让一部分线程成为生产者、另一部分线程成为消费者，因此必须使用 `cuda::thread_scope_block` 作用域的**分区**流水线。分区流水线需要 `cuda::pipeline_shared_state` 来协调参与线程。我们在线程块作用域初始化一个两阶段流水线的状态，然后调用 `cuda::make_pipeline()`。接着，生产者线程提交从 `in` 到 `buffer` 的异步复制来填充流水线；此时所有数据复制都在进行中。最后，在主循环中遍历所有数据批次，并根据线程是生产者还是消费者，选择为未来批次提交另一个异步复制，或者消费当前批次。
