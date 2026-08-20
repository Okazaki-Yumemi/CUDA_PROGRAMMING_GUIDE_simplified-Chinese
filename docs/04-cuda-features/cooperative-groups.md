---
title: 4.4 Cooperative Groups
description: Cooperative Groups 协作线程组、同步与集体操作的完整中文翻译
---

# 4.4 Cooperative Groups（协作组）

## 4.4.1 介绍（Introduction）

Cooperative Groups 是 CUDA 编程模型的扩展，用于组织协作线程组。Cooperative Groups 允许开发者控制线程协作的粒度，从而表达更丰富、更高效的并行分解。Cooperative Groups 还提供常见并行原语的实现，例如 scan 和并行 reduce。

在 Cooperative Groups 引入之前，CUDA 编程模型只提供了一种简单的协作线程同步结构：由 thread block 中所有线程共同参与的 barrier，对应的内置函数是 `__syncthreads()`。为了表达更广泛的并行交互模式，许多注重性能的程序员不得不自行编写临时且不安全的原语，用来同步单个 warp 内的线程，或同步运行在同一 GPU 上的一组 thread block。尽管这些做法往往能够带来有价值的性能提升，但它们也形成了越来越多脆弱的代码，随着时间推移以及 GPU 代际变化，编写、调优和维护的成本都很高。Cooperative Groups 为编写高性能代码提供了安全且面向未来的机制。

完整的 Cooperative Groups API 可参见 [Cooperative Groups API](../05-technical-appendices/device-callable-apis.html#cg-api-partition-header)。

## 4.4.2 Cooperative Group 句柄与成员函数（Cooperative Group Handle & Member Functions）

Cooperative Groups 通过 Cooperative Group 句柄管理。该句柄允许参与线程了解自己在组中的位置、组大小以及其他组信息。下表列出部分组成员函数。

| 访问器 | 返回值 |
|---|---|
| `thread_rank()` | 调用线程的 rank。 |
| `num_threads()` | 组中的线程总数。 |
| `thread_index()` | 线程在已启动 block 内的三维索引。 |
| `dim_threads()` | 以线程为单位表示的已启动 block 的三维尺寸。 |

表 10：部分成员函数。

完整的成员函数列表可参见 [Cooperative Groups API](../05-technical-appendices/device-callable-apis.html#cg-api-common-header)。

## 4.4.3 默认行为 / 无组执行（Default Behavior / Groupless Execution）

表示 grid 和 thread block 的组会根据 kernel 启动配置隐式创建。这些“隐式”组为开发者提供了起点，开发者可以进一步将其显式分解为更细粒度的组。可通过以下方法访问隐式组：

| 访问器 | 组范围 |
|---|---|
| `this_thread_block()` | 返回一个包含当前 thread block 全部线程的组句柄。 |
| `this_grid()` | 返回一个包含整个 grid 全部线程的组句柄。 |
| `coalesced_threads()` [^1] | 返回当前 warp 中当前处于活动状态的线程组句柄。 |
| `this_cluster()` [^2] | 返回当前 cluster 中线程组的句柄。 |

表 11：由 CUDA Runtime 隐式创建的 Cooperative Groups。

[^1]: `coalesced_threads()` 运算符返回该时刻处于活动状态的线程集合；它不保证返回哪些线程（只要线程处于活动状态即可），也不保证这些线程在整个执行期间始终保持合并状态。

[^2]: 当启动非 cluster grid 时，`this_cluster()` 假定 cluster 为 1×1×1。需要计算能力 9.0 或更高。

更多信息可参见 [Cooperative Groups API](../05-technical-appendices/device-callable-apis.html#cg-api-common-header)。

### 4.4.3.1 尽早创建隐式组句柄（Create Implicit Group Handles As Early As Possible）

为获得最佳性能，建议预先创建隐式组的句柄，即尽可能早地、在发生任何分支之前创建，并在整个 kernel 中使用该句柄。

### 4.4.3.2 只通过引用传递组句柄（Only Pass Group Handles by Reference）

将组句柄传入函数时，建议通过引用传递。组句柄必须在声明时初始化，因为它没有默认构造函数。不建议通过复制构造来创建组句柄。

## 4.4.4 创建 Cooperative Groups（Creating Cooperative Groups）

组通过将父组划分为子组来创建。划分组时，会创建一个组句柄来管理产生的子组。开发者可以使用以下划分操作：

| 划分类型 | 说明 |
|---|---|
| `tiled_partition` | 将父组划分为一系列固定大小的子组，并以一维行主序排列。 |
| `labeled_partition` | 根据条件标签将父组划分为一维子组；标签可以是任意整型。 |
| `binary_partition` | `labeled_partition` 的特化形式，标签只能是 `0` 或 `1`。 |

表 12：Cooperative Group 划分操作。

下面的示例展示如何创建 tiled partition：

```cpp
namespace cg = cooperative_groups;
// 获取当前线程的 cooperative group
cg::thread_block my_group = cg::this_thread_block();

// 将 cooperative group 划分为大小为 8 的 tile
cg::thread_block_tile<8> my_subgroup = cg::tiled_partition<8>(my_group);

// 以 my_subgroup 为单位执行工作
```

应采用哪种划分策略取决于具体上下文。更多信息可参见 [Cooperative Groups API](../05-technical-appendices/device-callable-apis.html#cg-api-partition-header)。

### 4.4.4.1 避免组创建风险（Avoiding Group Creation Hazards）

划分组是集体操作，组中的所有线程都必须参与。如果组是在并非所有线程都能到达的条件分支中创建的，可能导致死锁或数据损坏。

## 4.4.5 同步（Synchronization）

在 Cooperative Groups 引入之前，CUDA 编程模型只允许在线程 block 完成 kernel 的边界处进行 block 间同步。Cooperative Groups 允许开发者以不同粒度同步协作线程组。

### 4.4.5.1 Sync（Sync）

可以调用集体 `sync()` 函数同步一个组。与 `__syncthreads()` 类似，`sync()` 函数提供以下保证：

- 组中线程在同步点之前执行的所有内存访问（例如读和写），在同步点之后对组中所有线程可见。

- 组中所有线程都到达同步点之后，任何线程才能继续越过该同步点执行。

下面的示例展示与 `__syncthreads()` 等价的 `cooperative_groups::sync()`：

```cpp
namespace cg = cooperative_groups;

cg::thread_block my_group = cg::this_thread_block();

// 同步 block 中的线程
cg::sync(my_group);
```

Cooperative Groups 可用于同步整个 grid。从 CUDA 13 开始，Cooperative Groups 不再支持多设备同步。详情请参见[大规模组](#4458-大规模组-large-scale-groups)一节。

更多同步信息可参见 [Cooperative Groups API](../05-technical-appendices/device-callable-apis.html#cg-api-sync-header)。

### 4.4.5.2 Barriers（Barriers）

Cooperative Groups 提供类似 `cuda::barrier` 的 barrier API，可用于更高级的同步。Cooperative Groups barrier API 与 `cuda::barrier` 有以下几个关键区别：

- Cooperative Groups barrier 会自动初始化。

- 每个阶段中，组内所有线程都必须到达 barrier 并等待一次。

- `barrier_arrive` 返回一个 `arrival_token` 对象，该对象必须传给对应的 `barrier_wait`；传入后它会被消耗，不能再次使用。

使用 Cooperative Groups barrier 时，程序员必须注意避免以下风险：

- 调用 `barrier_arrive` 后、调用 `barrier_wait` 前，组不能执行任何集体操作。

- `barrier_wait` 只能保证组内所有线程都调用了 `barrier_arrive`；`barrier_wait` **不能**保证组内所有线程都调用了 `barrier_wait`。

```cpp
namespace cg = cooperative_groups;

cg::thread_block my_group = this_block();
cg::cluster_group cluster = this_cluster();

auto token = cluster.barrier_arrive();

// 可选：执行一些本地处理，以隐藏同步延迟
     local_processing(my_group);

// 确保 cluster 中所有其他 block 都在运行，并且已初始化共享数据，之后才能访问 dsmem
cluster.barrier_wait(std::move(token));
```

## 4.4.6 集体操作（Collective Operations）

Cooperative Groups 包含一组可由线程组执行的集体操作。为了完成操作，指定组中的所有线程都必须参与。

除非 [Cooperative Groups API](../05-technical-appendices/device-callable-apis.html#cg-api-partition-header) 明确允许不同值，否则组内所有线程在每次集体调用中都必须为对应参数传入相同值。否则，该调用的行为未定义。

### 4.4.6.1 Reduce（Reduce）

`reduce` 函数用于对指定组中每个线程提供的数据执行并行归约。必须通过提供下表中的一个运算符指定归约类型。

| 运算符 | 返回值 |
|---|---|
| `plus` | 组中所有值的和。 |
| `less` | 最小值。 |
| `greater` | 最大值。 |
| `bit_and` | 按位 AND 归约。 |
| `bit_or` | 按位 OR 归约。 |
| `bit_xor` | 按位 XOR 归约。 |

表 13：Cooperative Groups 归约运算符。

如果硬件支持，会使用硬件加速归约（需要计算能力 8.0 或更高）。对于不支持硬件加速的旧硬件，会使用软件回退实现。只有 4 字节类型能获得硬件加速。

更多归约信息可参见 [Cooperative Groups API](../05-technical-appendices/device-callable-apis.html#cg-api-reduce-header)。

下面的示例展示如何使用 `cooperative_groups::reduce()` 执行 block 范围的求和归约。

```cpp
namespace cg = cooperative_groups;

cg::thread_block my_group = cg::this_thread_block();

int val = data[threadIdx.x];

int sum = cg::reduce(my_group, val, cg::plus<int>());

// 保存归约结果
if (my_group.thread_rank() == 0) {
   result[blockIdx.x] = sum;
}
```

### 4.4.6.2 Scans（Scans）

Cooperative Groups 包含 `inclusive_scan` 和 `exclusive_scan` 的实现，可用于任意大小的组。这些函数对指定组中每个线程提供的数据执行 scan 操作。

程序员可以选择指定归约运算符，参见上面的[归约运算符表](#cooperative-groups-reduction-operators)。

```cpp
namespace cg = cooperative_groups;

cg::thread_block my_group = cg::this_thread_block();

int val = data[my_group.thread_rank()];

int exclusive_sum = cg::exclusive_scan(my_group, val, cg::plus<int>());

result[my_group.thread_rank()] = exclusive_sum;
```

更多 scan 信息可参见 [Cooperative Groups Scan API](../05-technical-appendices/device-callable-apis.html#cg-api-scan-header)。

### 4.4.6.3 Invoke One（Invoke One）

当必须由单个线程代表一个组执行串行工作的一部分时，Cooperative Groups 提供 `invoke_one` 函数。

- `invoke_one` 从调用组中选择一个任意线程，并使用该线程、给定参数调用提供的可调用函数。

- `invoke_one_broadcast` 与 `invoke_one` 相同，但还会把调用结果广播给组中所有线程。

线程选择机制不保证是确定性的。

下面的示例展示 `invoke_one` 的基本用法。

```cpp
namespace cg = cooperative_groups;
cg::thread_block my_group = cg::this_thread_block();

// 确保 thread block 中只有一个线程打印消息
cg::invoke_one(my_group, []() {
   printf("Hello from one thread in the block!");
});

// 同步，确保所有线程等待消息打印完成
cg::sync(my_group);
```

可调用函数内部不允许与调用组中的线程通信或同步。允许与调用组之外的线程通信。

## 4.4.7 异步数据移动（Asynchronous Data Movement）

CUDA 中 Cooperative Groups 的 `memcpy_async` 功能提供了在全局内存与共享内存之间执行异步内存拷贝的方法。`memcpy_async` 对优化内存传输、重叠计算与数据传输尤其有用，可以提升性能。

`memcpy_async` 函数用于启动从全局内存到共享内存的异步加载。`memcpy_async` 的设计用途类似“预取”：在需要数据之前先加载它。

`wait` 函数让组中所有线程等待异步内存传输完成。组中所有线程都必须调用 `wait`，之后才能访问共享内存中的数据。

下面的示例展示如何使用 `memcpy_async` 和 `wait` 预取数据。

```cpp
namespace cg = cooperative_groups;

cg::thread_group my_group = cg::this_thread_block();

__shared__ int shared_data[];

// 执行从全局内存到共享内存的异步拷贝
cg::memcpy_async(my_group, shared_data + my_group.rank(), input + my_group.rank(), sizeof(int));

// 在此处执行工作来隐藏延迟。此时不能使用 shared_data

// 等待异步拷贝完成
cg::wait(my_group);

// 预取的数据现在可用
```

更多信息请参见 [Cooperative Groups API](../05-technical-appendices/device-callable-apis.html#cg-api-async-header)。

### 4.4.7.1 Memcpy Async 对齐要求（Memcpy Async Alignment Requirements）

只有当源是全局内存、目标是共享内存，并且二者都至少按 4 字节对齐时，`memcpy_async` 才会异步执行。为获得最佳性能，建议共享内存和全局内存都按 16 字节对齐。

## 4.4.8 大规模组（Large Scale Groups）

Cooperative Groups 支持跨越整个 grid 的大规模组。前文描述的所有 Cooperative Group 功能都可用于这些大规模组，但有一个重要例外：同步整个 grid 必须使用 `cudaLaunchCooperativeKernel` runtime 启动 API。

从 CUDA 13 开始，多设备启动 API 以及 Cooperative Groups 的相关引用已被移除。

### 4.4.8.1 何时使用 `cudaLaunchCooperativeKernel`（When to use `cudaLaunchCooperativeKernel`）

`cudaLaunchCooperativeKernel` 是 CUDA runtime API 函数，用于启动使用 Cooperative Groups 的单设备 kernel，尤其适合需要 block 间同步的 kernel。该函数确保 kernel 中的所有线程都能跨整个 grid 同步和协作；传统 CUDA kernel 只能在线程 block 内同步，无法实现这一点。`cudaLaunchCooperativeKernel` 还确保 kernel 启动是原子的，即如果 API 调用成功，则指定数量的 thread block 会在指定设备上启动。

良好的实践是先查询设备属性 `cudaDevAttrCooperativeLaunch`，确认设备支持协作启动：

```cpp
int dev = 0;
int supportsCoopLaunch = 0;
cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
```

如果设备 0 支持该属性，上述代码会将 `supportsCoopLaunch` 设为 1。只有计算能力 6.0 及更高的设备受支持。此外，还必须运行在以下平台之一：

- 不使用 MPS 的 Linux 平台；
- 使用 MPS 且设备计算能力为 7.0 或更高的 Linux 平台；
- 最新的 Windows 平台。
