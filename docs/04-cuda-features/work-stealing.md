---
title: 4.12 使用 Cluster Launch Control 进行工作窃取
description: 使用 Cluster Launch Control 实现 CUDA kernel 的动态负载均衡
---

# 4.12 使用 Cluster Launch Control 进行工作窃取（Work Stealing with Cluster Launch Control）

处理数据量和计算量可变的问题，是开发 CUDA 应用时的一个重要环节。传统上，CUDA 开发者主要使用两种方式确定要启动的 kernel thread block 数量：每个 thread block 固定工作量，以及固定 thread block 数量。两种方法各有优缺点。

**每个 Thread Block 固定工作量（Fixed Work per Thread Block）**：根据问题规模决定 thread block 数量，同时保持每个 thread block 执行的工作量不变。

这种方法的主要优点是：

- **SM 之间的负载均衡**

  当 thread block 的运行时间存在差异，或者 thread block 数远多于 GPU 可同时执行的数量（因此产生低尾效应）时，GPU 调度器可以让某些 SM 比其他 SM 多运行一些 thread block。

- **抢占（Preemption）**

  即使较低优先级 kernel 已经开始执行，GPU 调度器也可以在它之后启动较高优先级 kernel：当低优先级 kernel 的 thread block 完成时，将高优先级 kernel 的 thread block 调度上去。高优先级 kernel 执行完毕后，调度器可以继续执行低优先级 kernel。

**固定 Thread Block 数量（Fixed Number of Thread Blocks）**：通常通过 block-stride loop 或 grid-stride loop 实现，thread block 数量不依赖问题规模，而每个 thread block 执行的工作量随问题规模变化。通常根据执行该 kernel 的 GPU 上的 SM 数量和期望占用率决定 thread block 数量。

这种方法的主要优点是：

- **降低 Thread Block 开销**

  这种方法不仅降低摊销后的 thread block 启动延迟，还能减少所有 thread block 之间共享操作带来的计算开销。这些开销可能显著高于启动延迟开销。

  例如在卷积 kernel 中，可以因为 thread block index 与计算卷积系数无关，而通过固定 thread block 数量减少计算卷积系数的序言执行次数，从而减少重复计算。

**Cluster Launch Control** 是 NVIDIA Blackwell GPU 架构（计算能力 10.0）引入的功能，旨在结合前两种方法的优点。它允许开发者取消 thread block 或 thread block cluster，从而更精细地控制 thread block 调度。这一机制实现了工作窃取：工作窃取是并行计算中的动态负载均衡技术，空闲处理器会主动从繁忙处理器的工作队列中“窃取”任务，而不是等待任务被分配。

![Cluster Launch Control 流程图](/images/chapter-04/cluster_launch_control.png)

*图 54：Cluster Launch Control 流程。该图展示 thread block 发起取消请求、请求成功后接管被取消 block 的索引并处理工作，以及请求失败后继续处理当前 block 的执行流程。*

使用 Cluster Launch Control 时，一个 thread block 会尝试取消另一个尚未开始执行的 thread block 的启动。如果取消请求成功，它会使用另一个 thread block 的索引来执行该任务，从而“窃取”其工作。如果没有更多可用的 thread block index，或者发生其他原因（例如调度了更高优先级的 kernel），取消就会失败。后一种情况下，如果 thread block 在取消失败后退出，调度器可以启动高优先级 kernel；高优先级 kernel 执行完后，调度器会继续调度当前 kernel 的剩余 thread block。上图展示了这一过程的执行流程。

下表总结三种方法的优点和缺点：

|  | 每个 Thread Block 固定工作量 | 固定 Thread Block 数量 | Cluster Launch Control |
| --- | --- | --- | --- |
| 降低开销 | 否 | 是 | 是 |
| 抢占 | 是 | 否 | 是 |
| 负载均衡 | 是 | 否 | 是 |

## 4.12.1 API 详情（API Details）

通过 Cluster Launch Control API 取消 thread block 是异步执行的，并使用共享内存 barrier 进行同步，其编程模式类似于异步数据复制。

该 API 通过 [libcu++](https://nvidia.github.io/cccl/unstable/libcudacxx/ptx_api.html) 提供，包含：

- 请求指令：将编码后的取消结果写入 `__shared__` 变量。
- 解码指令：提取成功/失败状态以及被取消的 thread block index。

请注意，Cluster Launch Control 操作被建模为 async proxy 操作。

### 4.12.1.1 Thread Block 取消（Thread Block Cancellation）

使用 Cluster Launch Control 的首选方式是由单个线程使用，即一次只发起一个请求。

取消过程包含五个步骤：

- **设置阶段（步骤 1–2）**：声明并初始化取消结果和同步变量。
- **工作窃取循环（步骤 3–5）**：重复执行请求、同步以及处理取消结果。

1. 声明 thread block 取消所需的变量：

   ```cpp
   __shared__ uint4 result; // 请求结果
   __shared__ uint64_t bar; // 同步 barrier
   int phase = 0;           // 同步 barrier 阶段
   ```

2. 使用单个到达计数初始化共享内存 barrier：

   ```cpp
   if (cg::thread_block::thread_rank() == 0)
       ptx::mbarrier_init(&bar, 1);
   __syncthreads();
   ```

3. 由单个线程提交异步取消请求，并设置事务计数：

   ```cpp
   if (cg::thread_block::thread_rank() == 0) {
       cg::invoke_one(cg::coalesced_threads(), [&](){ptx::clusterlaunchcontrol_try_cancel(&result, &bar);});
       ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
   }
   ```

   ::: note
   由于 thread block 取消是 uniform instruction，建议在 `invoke_one` 线程选择器中提交它。这样编译器可以优化掉 peeling loop。
   :::

4. 同步并完成异步取消请求：

   ```cpp
   while (!ptx::mbarrier_try_wait_parity(&bar, phase))
   {}
   phase ^= 1;
   ```

5. 获取取消状态和被取消的 thread block index：

   ```cpp
   bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
   if (success) {
       // 对于一维/二维 thread block，不需要以下三个值全部存在：
       int bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x(result);
       int by = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_y(result);
       int bz = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_z(result);
   }
   ```

6. 确保 async proxy 与 generic proxy 之间的共享内存操作可见，并防止工作窃取循环不同迭代之间的数据竞争。

### 4.12.1.2 Thread Block 取消的约束（Constraints on Thread Block Cancellation）

这些约束与取消请求失败有关：

- **观察到**之前失败的请求后，再提交另一个取消请求属于未定义行为。

  在下面两个代码示例中，假设第一次取消请求失败，只有第一个示例表现出未定义行为。第二个示例在两个取消请求之间没有进行观察，因此是正确的。

  **无效代码：**

  ```cpp
  // 第一个请求：
  ptx::clusterlaunchcontrol_try_cancel(&result0, &bar0);

  // 查询第一个请求：
  [此处为同步 bar0 的代码]
  bool success0 = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result0);
  assert(!success0); // 已观察到失败；第二个取消请求无效

  // 第二个请求 - 下一行属于未定义行为：
  ptx::clusterlaunchcontrol_try_cancel(&result1, &bar1);
  ```

  **有效代码：**

  ```cpp
  // 第一个请求：
  ptx::clusterlaunchcontrol_try_cancel(&result0, &bar0);

  // 第二个请求：
  ptx::clusterlaunchcontrol_try_cancel(&result1, &bar1);

  // 查询第一个请求：
  [此处为同步 bar0 的代码]
  bool success0 = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result0);
  assert(!success0); // 已观察到失败；第二个取消请求有效
  ```

- 获取失败取消请求的 thread block index 属于未定义行为。
- 不建议由多个线程提交取消请求。这样会取消多个 thread block，并需要谨慎处理，例如：
  - 每个提交线程必须提供唯一的 `__shared__` result 指针，以避免数据竞争。
  - 如果使用同一个 barrier 进行同步，则必须相应调整到达计数和事务计数。

## 4.12.2 示例：向量-标量乘法（Example: Vector-Scalar Multiplication）

下面的小节通过向量-标量乘法 kernel 演示使用 Cluster Launch Control 进行工作窃取。我们展示同一问题的两个变体：一个使用 thread block，另一个使用 thread block cluster。

### 4.12.2.1 用例：Thread Block（Use-case: Thread Blocks）

下面三个 kernel 分别展示每个 thread block 固定工作量、固定 thread block 数量以及 Cluster Launch Control 三种方法，用于实现向量-标量乘法 $\overline{v} := \alpha \overline{v}$。

- **每个 Thread Block 固定工作量：**

  ```cpp
  __global__
  void kernel_fixed_work (float* data, int n)
  {
      // Prologue:
      float alpha = compute_scalar();

      // Computation:
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n)
          data[i] *= alpha;
  }

  // Launch: kernel_fixed_work<<<(n + 1023) / 1024, 1024>>>(data, n);
  ```

- **固定 Thread Block 数量：**

  ```cpp
  __global__
  void kernel_fixed_blocks (float* data, int n)
  {
      // Prologue:
      float alpha = compute_scalar();

      // Computation:
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      while (i < n) {
          data[i] *= alpha;
          i += gridDim.x * blockDim.x;
      }
  }

  // Launch: kernel_fixed_blocks<<<SM_COUNT, 1024>>>(data, n);
  ```

- **Cluster Launch Control：**

  ```cpp
  #include <cooperative_groups.h>
  #include <cuda/ptx>

  namespace cg = cooperative_groups;
  namespace ptx = cuda::ptx;

  __global__
  void kernel_cluster_launch_control (float* data, int n)
  {
      // Cluster launch control initialization:
      __shared__ uint4 result;
      __shared__ uint64_t bar;
      int phase = 0;

      if (cg::thread_block::thread_rank() == 0)
          ptx::mbarrier_init(&bar, 1);

      // Prologue:
      float alpha = compute_scalar(); // 此代码片段未展示 device function。

      // Work-stealing loop:
      int bx = blockIdx.x; // 假设 thread block 使用一维 x 轴。

      while (true) {
          // 防止下一轮覆盖 result，
          // （也确保第一轮迭代时 barrier 已初始化）：
          __syncthreads();

          // Cancellation request:
          if (cg::thread_block::thread_rank() == 0) {
              // 在 async proxy 中获取 result 的写权限：
              ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);

              cg::invoke_one(cg::coalesced_threads(), [&](){ptx::clusterlaunchcontrol_try_cancel(&result, &bar);});
              ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
          }

          // Computation:
          int i = bx * blockDim.x + threadIdx.x;
          if (i < n)
              data[i] *= alpha;

          // Cancellation request synchronization:
          while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase))
          {}
          phase ^= 1;

          // Cancellation request decoding:
          bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
          if (!success)
              break;

          bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);

          // 在 async proxy 中释放 result 的读权限：
          ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
      }
  }

  // Launch: kernel_cluster_launch_control<<<(n + 1023) / 1024, 1024>>>(data, n);
  ```

### 4.12.2.2 用例：Thread Block Cluster（Use-case: Thread Block Clusters）

对于 [thread block cluster](../02-programming-gpus/intro-to-cuda-cpp.html#thread-block-clusters)，thread block 取消步骤与非 cluster 情况相同，只需要做少量调整。同样，不建议在**一个 cluster 内由多个线程**提交取消请求，因为这会尝试取消多个 cluster。

- 取消请求由单个 cluster 线程提交。
- 每个 cluster 的每个 thread block 的共享内存 result 都会收到被取消 thread block index 的相同（编码后的）值，也就是 result 会被 multicast。所有 thread block 收到的 result 对应 cluster 内的本地 block index `{0, 0, 0}`。因此，cluster 内的 thread block 需要加上各自的本地 block index。
- 每个 cluster 的 thread block 使用本地 `__shared__` 内存 barrier 执行同步。barrier 操作必须使用 `ptx::scope_cluster` 作用域。
- 在 cluster 情况下执行取消要求所有 thread block 都已存在。可以通过 `cg::cluster_group::sync()`（来自 `sync` API）保证所有 thread block 都在运行。

下面的 kernel 展示使用 thread block cluster 的 Cluster Launch Control 方法。

```cpp
#include <cooperative_groups.h>
#include <cuda/ptx>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

__global__ __cluster_dims__(2, 1, 1)
void kernel_cluster_launch_control (float* data, int n)
{
    // Cluster launch control initialization:
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;

    if (cg::thread_block::thread_rank() == 0) {
        ptx::mbarrier_init(&bar, 1);
        ptx::fence_mbarrier_init(ptx::sem_release, ptx::scope_cluster); // CGA-level fence。
    }

    // Prologue:
    float alpha = compute_scalar(); // 此代码片段未展示 device function。

    // Work-stealing loop:
    int bx = blockIdx.x; // 假设 thread block 使用一维 x 轴。

    while (true) {
        // 防止下一轮覆盖 result，
        // （也确保第一轮迭代时所有 thread block 都已启动）：
        cg::cluster_group::sync();

        // 由单个 cluster 线程发起取消请求：
        if (cg::cluster_group::thread_rank() == 0) {
            // 在 async proxy 中获取 result 的写权限：
            ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);

            cg::invoke_one(cg::coalesced_threads(), [&](){ptx::clusterlaunchcontrol_try_cancel_multicast(&result, &bar);});
        }

        // 每个 thread block 跟踪取消完成状态：
        if (cg::thread_block::thread_rank() == 0)
            ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cluster, ptx::space_shared, &bar, sizeof(uint4));

        // Computation:
        int i = bx * blockDim.x + threadIdx.x;
        if (i < n)
            data[i] *= alpha;

        // Cancellation request synchronization:
        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cluster, &bar, phase))
        {}
        phase ^= 1;

        // Cancellation request decoding:
        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
        if (!success)
            break;

        bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
        bx += cg::cluster_group::block_index().x; // 加上本地偏移量。

        // 在 async proxy 中释放 result 的读权限：
        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
    }
}

// Launch: kernel_cluster_launch_control<<<(n + 1023) / 1024, 1024>>>(data, n);
```
