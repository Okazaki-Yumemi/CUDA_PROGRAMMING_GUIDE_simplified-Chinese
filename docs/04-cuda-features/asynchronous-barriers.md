---
title: 4.9 异步 Barrier
description: CUDA 异步 barrier 的初始化、阶段跟踪、事务跟踪与生产者-消费者模式
---

# 4.9 异步 Barrier（Asynchronous Barriers）

异步 barrier 在[高级同步原语](../03-advanced-cuda/advanced-apis-and-features.html)中介绍。它把 CUDA 同步能力扩展到 `__syncthreads()` 和 `__syncwarp()` 之外，实现细粒度、非阻塞的协调，并更好地重叠通信与计算。

本节主要介绍如何通过 `cuda::barrier` API 使用异步 barrier，并在适用处指向 `cuda::ptx` 和 primitive。

## 4.9.1 初始化（Initialization）

初始化必须发生在任何线程开始参与 barrier 之前。

### CUDA C++ `cuda::barrier`

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

__global__ void init_barrier()
{
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    // A single thread initializes the total expected arrival count.
    init(&bar, block.size());
  }
  block.sync();
}
```

### CUDA C++ `cuda::ptx`

```cpp
#include <cuda/ptx>
#include <cooperative_groups.h>

__global__ void init_barrier()
{
  __shared__ uint64_t bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    // A single thread initializes the total expected arrival count.
    cuda::ptx::mbarrier_init(&bar, block.size());
  }
  block.sync();
}
```

### CUDA C primitive

```cpp
#include <cuda_awbarrier_primitives.h>
#include <cooperative_groups.h>

__global__ void init_barrier()
{
  __shared__ uint64_t bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    // A single thread initializes the total expected arrival count.
    __mbarrier_init(&bar, block.size());
  }
  block.sync();
}
```

在任何线程参与 barrier 之前，必须使用 `cuda::barrier::init()` 友元函数初始化 barrier。这必须发生在任何线程到达 barrier 之前。这带来了一个启动问题：线程必须在参与 barrier 之前先同步，但线程又正是为了同步才创建 barrier。在本例中，参与线程属于一个 cooperative group，并使用 `block.sync()` 引导初始化。由于整个线程块都参与 barrier，也可以使用 `__syncthreads()`。

`init()` 的第二个参数是*预期到达计数（expected arrival count）*，即参与线程调用 `bar.arrive()` 的次数；达到该次数后，参与线程才会从 `bar.wait(std::move(token))` 调用中解除阻塞。在本例和前面的示例中，barrier 使用线程块中的线程数初始化，即 `cooperative_groups::this_thread_block().size()`，因此线程块内的所有线程都可以参与 barrier。

异步 barrier 可以灵活指定线程如何参与（拆分 arrive/wait）以及哪些线程参与。相比之下，`this_thread_block.sync()` 或 `__syncthreads()` 作用于整个线程块，`__syncwarp(mask)` 作用于指定的 warp 子集。不过，如果用户的意图是同步完整线程块或完整 warp，建议分别使用 `__syncthreads()` 和 `__syncwarp()`，以获得更好的性能。

## 4.9.2 Barrier 的阶段：到达、倒计数、完成与重置（A Barrier’s Phase: Arrival, Countdown, Completion, and Reset）

异步 barrier 会随着参与线程调用 `bar.arrive()`，从预期到达计数倒数到零。当倒计数达到零时，barrier 在当前阶段完成。当最后一次 `bar.arrive()` 使倒计数达到零时，倒计数会自动、原子地重置。重置会把倒计数恢复为预期到达计数，并将 barrier 推进到下一阶段。

`cuda::barrier::arrival_token` 类的 `token` 对象（由 `token = bar.arrive()` 返回）与 barrier 的当前阶段相关联。调用 `bar.wait(std::move(token))` 时，如果 barrier 仍处于当前阶段，也就是 token 关联的阶段与 barrier 的阶段相同，调用线程会被阻塞。如果在调用 `bar.wait(std::move(token))` 之前阶段已经推进（因为倒计数达到零），线程不会阻塞；如果线程在 `bar.wait(std::move(token))` 中被阻塞时阶段推进，线程会解除阻塞。

**必须明确知道在复杂的 arrive/wait 同步模式中，何时可能发生、何时不可能发生重置。**

- 线程对 `token = bar.arrive()` 和 `bar.wait(std::move(token))` 的调用必须按如下顺序排列：`token = bar.arrive()` 发生在 barrier 的当前阶段，而 `bar.wait(std::move(token))` 发生在同一阶段或下一阶段。
- 线程调用 `bar.arrive()` 时，barrier 的计数器必须非零。初始化 barrier 后，如果线程调用 `bar.arrive()` 使倒计数达到零，则必须在 barrier 能够被后续 `bar.arrive()` 重新使用之前调用 `bar.wait(std::move(token)`。
- `bar.wait()` 只能使用当前阶段或紧邻前一阶段的 `token` 对象调用。对 `token` 对象的其他取值，行为未定义。

对于简单的 arrive/wait 同步模式，遵守这些使用规则比较直接。

### 4.9.2.1 Warp 纠缠（Warp Entanglement）

Warp 发散会影响 arrive-on 操作更新 barrier 的次数。如果调用该操作的 warp 完全收敛，则 barrier 更新一次；如果 warp 完全发散，则会对 barrier 应用 32 次独立更新。

> **注意**
>
> 建议由收敛线程调用 `arrive-on(bar)`，以减少对 barrier 对象的更新。当这些操作之前的代码使线程发散时，应在调用 arrive-on 操作之前通过 `__syncwarp` 重新收敛 warp。

## 4.9.3 显式阶段跟踪（Explicit Phase Tracking）

异步 barrier 可以有多个阶段，具体取决于它被用于同步线程和内存操作的次数。我们可以不使用 token 跟踪 barrier 阶段翻转，而是直接使用 `cuda::ptx` 和 primitive API 提供的 `mbarrier_try_wait_parity()` 函数族跟踪阶段。

最简单的形式是，`cuda::ptx::mbarrier_try_wait_parity(uint64_t* bar, const uint32_t& phaseParity)` 函数等待具有特定奇偶性的阶段。`phaseParity` 操作数是 barrier 对象当前阶段或紧邻前一阶段的整数奇偶性。偶数阶段的整数奇偶性为 0，奇数阶段为 1。初始化 barrier 时，其阶段奇偶性为 0，因此 `phaseParity` 的有效值为 0 和 1。

在跟踪异步内存操作时，显式阶段跟踪很有用：它允许只有一个线程到达 barrier 并设置事务计数，而其他线程只等待基于奇偶性的阶段翻转。这可能比让所有线程到达 barrier 并使用 token 更高效。此功能仅适用于线程块作用域和 cluster 作用域的 shared-memory barrier。

### CUDA C++ `cuda::barrier`

```cpp
#include <cuda/ptx>
#include <cooperative_groups.h>

__device__ void compute(float *data, int iteration);

__global__ void split_arrive_wait(int iteration_count, float *data)
{
  using barrier_t = cuda::barrier<cuda::thread_scope_block>;
  __shared__ barrier_t bar;
  int parity = 0; // Initial phase parity is 0.
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    // Initialize barrier with expected arrival count.
    init(&bar, block.size());
  }
  block.sync();

  for (int i = 0; i < iteration_count; ++i)
  {
    /* code before arrive */

    // This thread arrives. Arrival does not block a thread.
    // Get a handle to the native barrier to use with cuda::ptx API.
    (void)cuda::ptx::mbarrier_arrive(cuda::device::barrier_native_handle(bar));

    compute(data, i);

    // Wait for all threads participating in the barrier to complete mbarrier_arrive().
    // Get a handle to the native barrier to use with cuda::ptx API.
    while (!cuda::ptx::mbarrier_try_wait_parity(cuda::device::barrier_native_handle(bar), parity)) {}
    // Flip parity.
    parity ^= 1;

    /* code after wait */
  }
}
```

### CUDA C++ `cuda::ptx`

```cpp
#include <cuda/ptx>
#include <cooperative_groups.h>

__device__ void compute(float *data, int iteration);

__global__ void split_arrive_wait(int iteration_count, float *data)
{
  __shared__ uint64_t bar;
  int parity = 0; // Initial phase parity is 0.
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    // Initialize barrier with expected arrival count.
    cuda::ptx::mbarrier_init(&bar, block.size());
  }
  block.sync();

  for (int i = 0; i < iteration_count; ++i)
  {
    /* code before arrive */

    // This thread arrives. Arrival does not block a thread.
    (void)cuda::ptx::mbarrier_arrive(&bar);

    compute(data, i);

    // Wait for all threads participating in the barrier to complete mbarrier_arrive().
    while (!cuda::ptx::mbarrier_try_wait_parity(&bar, parity)) {}
    // Flip parity.
    parity ^= 1;

    /* code after wait */
  }
}
```

### CUDA C primitive

```cpp
#include <cuda_awbarrier_primitives.h>
#include <cooperative_groups.h>

__device__ void compute(float *data, int iteration);

__global__ void split_arrive_wait(int iteration_count, float *data)
{
  __shared__ __mbarrier_t bar;
  bool parity = false; // Initial phase parity is false.
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    // Initialize barrier with expected arrival count.
    __mbarrier_init(&bar, block.size());
  }
  block.sync();

  for (int i = 0; i < iteration_count; ++i)
  {
    /* code before arrive */

    // This thread arrives. Arrival does not block a thread.
    (void)__mbarrier_arrive(&bar);

    compute(data, i);

    // Wait for all threads participating in the barrier to complete __mbarrier_arrive().
    while(!__mbarrier_try_wait_parity(&bar, parity, 1000)) {}
    parity ^= 1;

    /* code after wait */
  }
}
```

## 4.9.4 提前退出（Early Exit）

当参与一系列同步操作的线程必须提前退出该序列时，该线程必须在退出前显式退出参与。其余参与线程可以正常执行后续 arrive 和 wait 操作。

### CUDA C++ `cuda::barrier`

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

__device__ bool condition_check();

__global__ void early_exit_kernel(int N)
{
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    init(&bar, block.size());
  }
  block.sync();

  for (int i = 0; i < N; ++i)
  {
    if (condition_check())
    {
      bar.arrive_and_drop();
      return;
    }
    // Other threads can proceed normally.
    auto token = bar.arrive();

    /* code between arrive and wait */

    // Wait for all threads to arrive.
    bar.wait(std::move(token));

    /* code after wait */
  }
}
```

### CUDA C primitive

```cpp
#include <cuda_awbarrier_primitives.h>
#include <cooperative_groups.h>

__device__ bool condition_check();

__global__ void early_exit_kernel(int N)
{
  __shared__ __mbarrier_t bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    __mbarrier_init(&bar, block.size());
  }
  block.sync();

  for (int i = 0; i < N; ++i)
  {
    if (condition_check())
    {
      __mbarrier_token_t token = __mbarrier_arrive_and_drop(&bar);
      return;
    }
    // Other threads can proceed normally.
    __mbarrier_token_t token = __mbarrier_arrive(&bar);

    /* code between arrive and wait */

    // Wait for all threads to arrive.
    while (!__mbarrier_try_wait(&bar, token, 1000)) {}

    /* code after wait */
  }
}
```

`bar.arrive_and_drop()` 操作会在 barrier 上到达，以履行参与线程在**当前**阶段的到达义务；随后，它会减少**下一**阶段的预期到达计数，因此该线程之后不再需要到达此 barrier。

## 4.9.5 完成函数（Completion Function）

`cuda::barrier` API 支持可选的完成函数。`cuda::barrier` 的 `CompletionFunction` 每个阶段执行一次：在最后一个线程*到达*之后、任何线程从 `wait` 中解除阻塞之前执行。在该阶段到达 barrier 的线程执行的内存操作，对执行 `CompletionFunction` 的线程可见；而 `CompletionFunction` 中执行的所有内存操作，在等待 barrier 的线程从 `wait` 中解除阻塞后，对它们都可见。

### CUDA C++ `cuda::barrier`

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <functional>
namespace cg = cooperative_groups;

__device__ int divergent_compute(int *, int);
__device__ int independent_computation(int *, int);

__global__ void psum(int *data, int n, int *acc)
{
  auto block = cg::this_thread_block();

  constexpr int BlockSize = 128;
  __shared__ int smem[BlockSize];
  assert(BlockSize == block.size());
  assert(n % BlockSize == 0);

  auto completion_fn = [&]
  {
    int sum = 0;
    for (int i = 0; i < BlockSize; ++i)
    {
      sum += smem[i];
    }
    *acc += sum;
  };

  /* Barrier storage.
     Note: the barrier is not default-constructible because
           completion_fn is not default-constructible due
           to the capture. */
  using completion_fn_t = decltype(completion_fn);
  using barrier_t = cuda::barrier<cuda::thread_scope_block,
                                  completion_fn_t>;
  __shared__ std::aligned_storage<sizeof(barrier_t),
                                  alignof(barrier_t)>
      bar_storage;

  // Initialize barrier.
  barrier_t *bar = (barrier_t *)&bar_storage;
  if (block.thread_rank() == 0)
  {
    assert(*acc == 0);
    assert(blockDim.x == blockDim.y == blockDim.y == 1);
    new (bar) barrier_t{block.size(), completion_fn};
    /* equivalent to: init(bar, block.size(), completion_fn); */
  }
  block.sync();

  // Main loop.
  for (int i = 0; i < n; i += block.size())
  {
    smem[block.thread_rank()] = data[i] + *acc;
    auto token = bar->arrive();
    // We can do independent computation here.
    bar->wait(std::move(token));
    // Shared-memory is safe to re-use in the next iteration
    // since all threads are done with it, including the one
    // that did the reduction.
  }
}
```

## 4.9.6 跟踪异步内存操作（Tracking Asynchronous Memory Operations）

异步 barrier 可以用于跟踪异步内存复制。当异步复制操作绑定到 barrier 时，复制操作开始时会自动增加当前 barrier 阶段的预期计数，完成时会减少该计数。这样可以确保 barrier 的 `wait()` 操作一直阻塞到所有相关的异步内存复制完成，为同步多个并发内存操作提供方便的机制。

从计算能力 9.0 开始，线程块作用域或 cluster 作用域的 shared-memory 异步 barrier 可以**显式**跟踪异步内存操作。我们将这类 barrier 称为*异步事务 barrier（asynchronous transaction barrier）*。除了预期到达计数外，barrier 对象还可以接受*事务计数（transaction count）*，用于跟踪异步事务的完成情况。事务计数跟踪尚未完成的异步事务数量，单位由异步内存操作指定（通常为字节）。当前阶段要跟踪的事务计数可以在到达时使用 `cuda::device::barrier_arrive_tx()` 设置，也可以直接使用 `cuda::device::barrier_expect_tx()` 设置。当 barrier 使用事务计数时，它会在 wait 操作处阻塞线程，直到所有生产者线程都完成 arrive，且事务计数之和达到预期值。

### CUDA C++ `cuda::barrier`

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

__global__ void track_kernel()
{
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    init(&bar, block.size());
  }
  block.sync();

  auto token = cuda::device::barrier_arrive_tx(bar, 1, 0);

  bar.wait(cuda::std::move(token));
}
```

### CUDA C++ `cuda::ptx`

```cpp
#include <cuda/ptx>
#include <cooperative_groups.h>

__global__ void track_kernel()
{
  __shared__ uint64_t bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    cuda::ptx::mbarrier_init(&bar, block.size());
  }
  block.sync();

  uint64_t token = cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release, cuda::ptx::scope_cluster, cuda::ptx::space_shared, &bar, 1, 0);

  while (!cuda::ptx::mbarrier_try_wait(&bar, token)) {}
}
```

在这个示例中，`cuda::device::barrier_arrive_tx()` 操作构造了一个与当前阶段同步点关联的到达 token 对象。然后它将到达计数减 1，并将预期事务计数增加 0。由于事务计数更新为 0，barrier 没有跟踪任何事务。后续[使用 Tensor Memory Accelerator（TMA）](./asynchronous-data-copies.html)一节包含跟踪异步内存操作的示例。

## 4.9.7 使用 Barrier 的生产者—消费者模式（Producer-Consumer Pattern Using Barriers）

线程块可以进行空间分区，使不同线程执行独立操作。最常见的做法是将线程块中不同 warp 的线程分配给特定任务。这种技术称为*warp 专门化（warp specialization）*。

本节展示空间分区生产者—消费者模式的示例：一个线程子集产生数据，另一个互不重叠的线程子集同时消费数据。生产者—消费者空间分区模式需要两次单向同步，以管理生产者和消费者之间的数据缓冲区。

| 生产者 | 消费者 |
| --- | --- |
| 等待缓冲区准备好以便填充 | 发出缓冲区已准备好填充的信号 |
| 产生数据并填充缓冲区 |  |
| 发出缓冲区已填充的信号 | 等待缓冲区填充完成 |
|  | 消费已填充缓冲区中的数据 |

生产者线程等待消费者线程发出缓冲区准备好填充的信号；但消费者线程不等待该信号。消费者线程等待生产者线程发出缓冲区已填充的信号；但生产者线程不等待该信号。要实现完整的生产者/消费者并发，该模式至少需要双缓冲，每个缓冲区需要两个 barrier。

### CUDA C++ `cuda::barrier`

```cpp
#include <cuda/barrier>

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

__device__ void produce(barrier_t ready[], barrier_t filled[], float *buffer, int buffer_len, float *in, int N)
{
  for (int i = 0; i < N / buffer_len; ++i)
  {
    ready[i % 2].arrive_and_wait(); /* wait for buffer_(i%2) to be ready to be filled */
    /* produce, i.e., fill in, buffer_(i%2)  */
    barrier_t::arrival_token token = filled[i % 2].arrive(); /* buffer_(i%2) is filled */
  }
}

__device__ void consume(barrier_t ready[], barrier_t filled[], float *buffer, int buffer_len, float *out, int N)
{
  barrier_t::arrival_token token1 = ready[0].arrive(); /* buffer_0 is ready for initial fill */
  barrier_t::arrival_token token2 = ready[1].arrive(); /* buffer_1 is ready for initial fill */
  for (int i = 0; i < N / buffer_len; ++i)
  {
    filled[i % 2].arrive_and_wait(); /* wait for buffer_(i%2) to be filled */
    /* consume buffer_(i%2) */
    barrier_t::arrival_token token3 = ready[i % 2].arrive(); /* buffer_(i%2) is ready to be re-filled */
  }
}

__global__ void producer_consumer_pattern(int N, float *in, float *out, int buffer_len)
{
  constexpr int warpSize = 32;

  /* Shared memory buffer declared below is of size 2 * buffer_len
     so that we can alternatively work between two buffers.
     buffer_0 = buffer and buffer_1 = buffer + buffer_len */
  __shared__ extern float buffer[];

  /* bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
     while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively */
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier_t bar[4];

  if (threadIdx.x < 4)
  {
    init(bar + threadIdx.x, blockDim.x);
  }
  __syncthreads();

  if (threadIdx.x < warpSize)
  { produce(bar, bar + 2, buffer, buffer_len, in, N); }
  else
  { consume(bar, bar + 2, buffer, buffer_len, out, N); }
}
```

### CUDA C++ `cuda::ptx`

```cpp
#include <cuda/ptx>

__device__ void produce(barrier ready[], barrier filled[], float *buffer, int buffer_len, float *in, int N)
{
  for (int i = 0; i < N / buffer_len; ++i)
  {
    uint64_t token1 = cuda::ptx::mbarrier_arrive(ready[i % 2]);
    while(!cuda::ptx::mbarrier_try_wait(&ready[i % 2], token1)) {} /* wait for buffer_(i%2) to be ready to be filled */
    /* produce, i.e., fill in, buffer_(i%2)  */
    uint64_t token2 = cuda::ptx::mbarrier_arrive(&filled[i % 2]); /* buffer_(i%2) is filled */
  }
}

__device__ void consume(barrier ready[], barrier filled[], float *buffer, buffer_len, float *out, int N)
{
  uint64_t token1 = cuda::ptx::mbarrier_arrive(&ready[0]); /* buffer_0 is ready for initial fill */
  uint64_t token2 = cuda::ptx::mbarrier_arrive(&ready[1]); /* buffer_1 is ready for initial fill */
  for (int i = 0; i < N / buffer_len; ++i)
  {
    uint64_t token3 = cuda::ptx::mbarrier_arrive(&filled[i % 2]);
    while(!cuda::ptx::mbarrier_try_wait(&filled[i % 2], token3x)) {} /* wait for buffer_(i%2) to be filled */
    /* consume buffer_(i%2) */
    uint64_t token4 = cuda::ptx::mbarrier_arrive(&ready[i % 2]); /* buffer_(i%2) is ready to be re-filled */
  }
}

__global__ void producer_consumer_pattern(int N, float *in, float *out, int buffer_len)
{
  constexpr int warpSize = 32;

  /* Shared memory buffer declared below is of size 2 * buffer_len
     so that we can alternatively work between two buffers.
     buffer_0 = buffer and buffer_1 = buffer + buffer_len */
  __shared__ extern float buffer[];

  /* bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
     while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively */
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t bar[4];

  if (threadIdx.x < 4)
  {
    cuda::ptx::mbarrier_init(bar + block.thread_rank(), block.size());
  }
  __syncthreads();

  if (threadIdx.x < warpSize)
  {  produce(bar, bar + 2, buffer, buffer_len, in, N); }
  else
  {  consume(bar, bar + 2, buffer, buffer_len, out, N); }
}
```

### CUDA C primitive

```cpp
#include <cuda_awbarrier_primitives.h>

__device__ void produce(__mbarrier_t ready[], __mbarrier_t filled[], float *buffer, int buffer_len, float *in, int N)
{
  for (int i = 0; i < N / buffer_len; ++i)
  {
    __mbarrier_token_t token1 = __mbarrier_arrive(&ready[i % 2]); /* wait for buffer_(i%2) to be ready to be filled */
    while(!__mbarrier_try_wait(&ready[i % 2], token1, 1000)) {}
    /* produce, i.e., fill in, buffer_(i%2)  */
    __mbarrier_token_t token2 = __mbarrier_arrive(filled[i % 2]);  /* buffer_(i%2) is filled */
  }
}

__device__ void consume(__mbarrier_t ready[], __mbarrier_t filled[], float *buffer, int buffer_len, float *out, int N)
{
  __mbarrier_token_t token1 = __mbarrier_arrive(&ready[0]); /* buffer_0 is ready for initial fill */
  __mbarrier_token_t token2 = __mbarrier_arrive(&ready[1]); /* buffer_1 is ready for initial fill */
  for (int i = 0; i < N / buffer_len; ++i)
  {
    __mbarrier_token_t token3 = __mbarrier_arrive(&filled[i % 2]);
    while(!__mbarrier_try_wait(&filled[i % 2], token3, 1000)) {}
    /* consume buffer_(i%2) */
    __mbarrier_token_t token4 = __mbarrier_arrive(&ready[i % 2]); /* buffer_(i%2) is ready to be re-filled */
  }
}

__global__ void producer_consumer_pattern(int N, float *in, float *out, int buffer_len)
{
  constexpr int warpSize = 32;

  /* Shared memory buffer declared below is of size 2 * buffer_len
     so that we can alternatively work between two buffers.
     buffer_0 = buffer and buffer_1 = buffer + buffer_len */
  __shared__ extern float buffer[];

  /* bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
     while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively */
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ __mbarrier_t bar[4];

  if (threadIdx.x < 4)
  {
    __mbarrier_init(bar + threadIdx.x, blockDim.x);
  }
  __syncthreads();

  if (threadIdx.x < warpSize)
  { produce(bar, bar + 2, buffer, buffer_len, in, N); }
  else
  { consume(bar, bar + 2, buffer, buffer_len, out, N); }
}
```

在这个示例中，第一个 warp 专门作为生产者，其余 warp 专门作为消费者。所有生产者和消费者线程都参与（调用 `bar.arrive()` 或 `bar.arrive_and_wait()`）四个 barrier，因此预期到达计数等于 `block.size()`。

生产者线程等待消费者线程发出共享内存缓冲区可以填充的信号。要等待 barrier，生产者线程必须先在 `ready[i%2].arrive()` 上到达以取得 token，然后使用该 token 调用 `ready[i%2].wait(token)`。为简化操作，`ready[i%2].arrive_and_wait()` 合并了这两个操作：

```cpp
bar.arrive_and_wait();
/* is equivalent to */
bar.wait(bar.arrive());
```

生产者线程计算并填充 ready buffer，随后在 filled barrier 上调用 `filled[i%2].arrive()`，发出缓冲区已填充的信号。生产者线程此时不等待，而是等到下一次迭代的缓冲区（双缓冲）准备好填充。

消费者线程首先发出两个缓冲区都已准备好填充的信号。消费者线程此时不等待，而是等待本次迭代的缓冲区填充完成，即调用 `filled[i%2].arrive_and_wait()`。消费者线程消费缓冲区后，调用 `ready[i%2].arrive()` 发出缓冲区再次准备好填充的信号，然后等待下一次迭代的缓冲区填充完成。
