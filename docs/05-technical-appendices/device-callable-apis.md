---
title: 5.6 Device-Callable APIs and Intrinsics（设备可调用 API 与内建函数）
description: 可从 CUDA kernel 和 device 代码调用的 API、内建函数、同步原语与 CUDA Device Runtime 参考
---

<a id="device-callable-apis-and-intrinsics"></a>
<span id="device-callable-intro"></span>

# 5.6 Device-Callable APIs and Intrinsics（设备可调用 API 与内建函数）

本节包含可从 CUDA kernel 和 device 代码调用的 API 与内建函数的参考材料及 API 文档。

<a id="memory-barrier-primitives-interface"></a>
<span id="async-barriers-primitives-api"></span>

## 5.6.1 Memory Barrier Primitives Interface（内存栅栏原语接口）

原语 API 是对 `cuda::barrier` 功能的类 C 接口。通过包含头文件 `<cuda_awbarrier_primitives.h>` 可以使用这些原语。

<a id="data-types"></a>
<span id="memory-barrier-primitives-datatypes"></span>

### 5.6.1.1 Data Types（数据类型）

```cuda
typedef /* implementation defined */ __mbarrier_t;
typedef /* implementation defined */ __mbarrier_token_t;
```

<a id="memory-barrier-primitives-api"></a>
<span id="id1"></span>

### 5.6.1.2 Memory Barrier Primitives API（内存栅栏原语 API）

```cuda
uint32_t __mbarrier_maximum_count();
void __mbarrier_init(__mbarrier_t* bar, uint32_t expected_count);
```

- `bar` 必须是指向 `__shared__` 内存的指针。

- `expected_count <= __mbarrier_maximum_count()`。

- 将 `*bar` 在当前阶段和下一阶段的预期到达计数初始化为 `expected_count`。

```cuda
void __mbarrier_inval(__mbarrier_t* bar);
```

- `bar` 必须是指向位于共享内存中的栅栏对象的指针。

- 在相应共享内存可以重新用于其他用途之前，必须使 `*bar` 失效。

```cuda
__mbarrier_token_t __mbarrier_arrive(__mbarrier_t* bar);
```

- 在调用此函数之前，必须完成 `*bar` 的初始化。

- 待处理计数不得为零。

- 将栅栏当前阶段的待处理计数以原子方式减一。

- 返回一个到达令牌；该令牌关联减一之前栅栏的状态。

```cuda
__mbarrier_token_t __mbarrier_arrive_and_drop(__mbarrier_t* bar);
```

- 在调用此函数之前，必须完成 `*bar` 的初始化。

- 待处理计数不得为零。

- 以原子方式将当前阶段的待处理计数和栅栏下一阶段的预期计数都减一。

- 返回一个到达令牌；该令牌关联减一之前栅栏的状态。

```cuda
bool __mbarrier_test_wait(__mbarrier_t* bar, __mbarrier_token_t token);
```

- `token` 必须关联 `*bar` 的紧邻前一阶段或当前阶段。

- 如果 `token` 关联 `*bar` 的紧邻前一阶段，则返回 `true`；否则返回 `false`。

```cuda
bool __mbarrier_test_wait_parity(__mbarrier_t* bar, bool phase_parity);
```

- `phase_parity` 必须表示 `*bar` 当前阶段或紧邻前一阶段的奇偶性。值为 `true` 表示奇数阶段，值为 `false` 表示偶数阶段。

- 如果 `phase_parity` 表示 `*bar` 紧邻前一阶段的整数奇偶性，则返回 `true`；否则返回 `false`。

```cpp
bool __mbarrier_try_wait(__mbarrier_t* bar, __mbarrier_token_t token, uint32_t max_sleep_nanosec);
```

- `token` 必须关联 `*bar` 的紧邻前一阶段或当前阶段。

- 如果 `token` 关联 `*bar` 的紧邻前一阶段，则返回 `true`。否则，正在执行的线程可能会被挂起。当指定阶段完成时，被挂起的线程恢复执行（返回 `true`）；或者在系统相关的时间限制到期之前恢复执行（返回 `false`）。

- `max_sleep_nanosec` 指定以纳秒为单位的时间限制；在确定时间限制时可以使用它替代系统相关的限制。

```cpp
bool __mbarrier_try_wait_parity(__mbarrier_t* bar, bool phase_parity, uint32_t max_sleep_nanosec);
```

- `phase_parity` 必须表示 `*bar` 当前阶段或紧邻前一阶段的奇偶性。值为 `true` 表示奇数阶段，值为 `false` 表示偶数阶段。

- 如果 `phase_parity` 表示 `*bar` 紧邻前一阶段的整数奇偶性，则返回 `true`。否则，正在执行的线程可能会被挂起。当指定阶段完成时，被挂起的线程恢复执行（返回 `true`）；或者在系统相关的时间限制到期之前恢复执行（返回 `false`）。

- `max_sleep_nanosec` 指定以纳秒为单位的时间限制；在确定时间限制时可以使用它替代系统相关的限制。

<a id="pipeline-primitives-interface"></a>
<span id="id2"></span>

## 5.6.2 Pipeline Primitives Interface（流水线原语接口）

流水线原语为 `<cuda/pipeline>` 中提供的功能提供类 C 接口。通过包含头文件 `<cuda_pipeline.h>` 可以使用流水线原语接口。在不使用 ISO C++ 2011 兼容模式编译时，应包含头文件 `<cuda_pipeline_primitives.h>`。

> **注意：**流水线原语 API 只支持跟踪从全局内存到共享内存的异步复制，并且复制必须满足特定的大小和对齐要求。它提供了与 `cuda::thread_scope_thread` 作用域的 `cuda::pipeline` 对象等价的功能。

<a id="memcpy-async-primitive"></a>

### 5.6.2.1 `memcpy_async` Primitive（`memcpy_async` 原语）

```cuda
void __pipeline_memcpy_async(void* __restrict__ dst_shared,
                             const void* __restrict__ src_global,
                             size_t size_and_align,
                             size_t zfill=0);
```

- 请求异步执行以下操作：

  ```cuda
  size_t i = 0;
  for (; i < size_and_align - zfill; ++i) ((char*)dst_shared)[i] = ((char*)src_global)[i]; /* copy */
  for (; i < size_and_align; ++i) ((char*)dst_shared)[i] = 0; /* zero-fill */
  ```

- 要求：

  - `dst_shared` 必须是指向 `memcpy_async` 共享内存目标的指针。

  - `src_global` 必须是指向 `memcpy_async` 全局内存源的指针。

  - `size_and_align` 必须是 `4`、`8` 或 `16`。

  - `zfill <= size_and_align`。

  - `size_and_align` 必须是 `dst_shared` 和 `src_global` 的对齐值。

- 在等待 `memcpy_async` 操作完成之前，任何线程修改源内存或观察目标内存，都是数据竞争。在提交 `memcpy_async` 操作与等待其完成之间，执行以下任一操作都会引入数据竞争：

  - 从 `dst_shared` 加载。

  - 存储到 `dst_shared` 或 `src_global`。

  - 对 `dst_shared` 或 `src_global` 应用原子更新。

<a id="commit-primitive"></a>

### 5.6.2.2 Commit Primitive（提交原语）

```cuda
void __pipeline_commit();
```

- 将已提交的 `memcpy_async` 操作提交到流水线，作为当前批次。

<a id="wait-primitive"></a>

### 5.6.2.3 Wait Primitive（等待原语）

```cuda
void __pipeline_wait_prior(size_t N);
```

- 设 `{0, 1, 2, ..., L}` 是一个线程调用 `__pipeline_commit()` 时所关联的一系列索引。

- 等待批次至少完成到 `L-N`（包括 `L-N`）。

<a id="arrive-on-barrier-primitive"></a>
<span id="arrive-primitive"></span>

### 5.6.2.4 Arrive On Barrier Primitive（到达栅栏原语）

```cuda
void __pipeline_arrive_on(__mbarrier_t* bar);
```

- `bar` 指向共享内存中的一个栅栏。

- 当本调用之前按顺序排列的所有 `memcpy_async` 操作完成时，将栅栏到达计数加一；此时到达计数再减一，因此对到达计数的净影响为零。用户有责任确保到达计数的增加不会超过 `__mbarrier_maximum_count()`。

<a id="cooperative-groups-api"></a>

## 5.6.3 Cooperative Groups API（Cooperative Groups API）

<a id="cooperative-groups-h"></a>
<span id="cg-api-common-header"></span>

### 5.6.3.1 cooperative_groups.h

<a id="class-thread-block"></a>
<span id="cg-api-thread-block"></span>

#### 5.6.3.1.1 class thread_block（`thread_block` 类）

任何 CUDA 程序员都熟悉一类线程：线程块。Cooperative Groups 扩展引入了新的数据类型 `thread_block`，用于在 kernel 内显式表示这一概念。

```cpp
class thread_block;
```

通过以下方式构造：

```cpp
thread_block g = this_thread_block();
```

**公共成员函数：**

- `static void sync()`：同步组中指定的线程，等价于 `g.barrier_wait(g.barrier_arrive())`。

- `thread_block::arrival_token barrier_arrive()`：到达 thread block 栅栏，并返回一个需要传递给 `barrier_wait()` 的令牌。

- `void barrier_wait(thread_block::arrival_token&& t)`：等待 `thread_block` 栅栏；接收由 `barrier_arrive()` 返回的到达令牌的右值引用。

- `static unsigned int thread_rank()`：调用线程在 `[0, num_threads)` 中的 rank。

- `static dim3 group_index()`：线程块在已启动 grid 中的三维索引。

- `static dim3 thread_index()`：线程在已启动线程块中的三维索引。

- `static dim3 dim_threads()`：以线程为单位表示的已启动线程块的尺寸。

- `static unsigned int num_threads()`：组中的线程总数。

旧版成员函数（别名）：

- `static unsigned int size()`：组中的线程总数（`num_threads()` 的别名）。

- `static dim3 group_dim()`：已启动线程块的尺寸（`dim_threads()` 的别名）。

**示例：**

```cpp
/// 将一个整数从全局内存加载到共享内存
__global__ void kernel(int *globalInput) {
    __shared__ int x;
    thread_block g = this_thread_block();
    // 在线程块中选择一个 leader
    if (g.thread_rank() == 0) {
        // 从全局内存加载数据，供所有线程使用
        x = (*globalInput);
    }
    // 数据加载到共享内存后，如果线程块中的所有线程都需要看到它，
    // 则需要进行同步
    g.sync(); // 等价于 __syncthreads();
}
```

<a id="class-cluster-group"></a>
<span id="cg-api-cluster-group"></span>

#### 5.6.3.1.2 class cluster_group（`cluster_group` 类）

这个组对象表示单个 cluster 中启动的所有线程。计算能力 9.0 及更高的所有硬件都提供这些 API。在此类硬件上，如果启动的是非 cluster grid，这些 API 会将 cluster 视为 `1x1x1`。

```cpp
class cluster_group;
```

通过以下方式构造：

```cpp
cluster_group g = this_cluster();
```

**公共成员函数：**

- `static void sync()`：同步组中指定的线程，等价于 `g.barrier_wait(g.barrier_arrive())`。

- `static cluster_group::arrival_token barrier_arrive()`：到达 cluster 栅栏，并返回一个需要传递给 `barrier_wait()` 的令牌。

- `static void barrier_wait(cluster_group::arrival_token&& t)`：等待 cluster 栅栏；接收由 `barrier_arrive()` 返回的到达令牌的右值引用。

- `static unsigned int thread_rank()`：调用线程在 `[0, num_threads)` 中的 rank。

- `static unsigned int block_rank()`：调用线程块在 `[0, num_blocks)` 中的 rank。

- `static unsigned int num_threads()`：组中的线程总数。

- `static unsigned int num_blocks()`：组中的线程块总数。

- `static dim3 dim_threads()`：以线程为单位表示的已启动 cluster 尺寸。

- `static dim3 dim_blocks()`：以线程块为单位表示的已启动 cluster 尺寸。

- `static dim3 block_index()`：调用线程块在已启动 cluster 中的三维索引。

- `static unsigned int query_shared_rank(const void *addr)`：获取某个共享内存地址所属的线程块 rank。

- `static T* map_shared_rank(T *addr, int rank)`：获取 cluster 中另一个线程块的共享内存变量地址。

旧版成员函数（别名）：

- `static unsigned int size()`：组中的线程总数（`num_threads()` 的别名）。

<a id="class-grid-group"></a>
<span id="cg-api-grid-group"></span>

#### 5.6.3.1.3 class grid_group（`grid_group` 类）

这个组对象表示单个 grid 中启动的所有线程。除 `sync()` 外，其他 API 始终可用；但如果要跨 grid 同步，必须使用 cooperative launch API。

```cpp
class grid_group;
```

通过以下方式构造：

```cpp
grid_group g = this_grid();
```

**公共成员函数：**

- `bool is_valid() const`：返回该 `grid_group` 是否可以执行同步。

- `void sync() const`：同步组中指定的线程，等价于 `g.barrier_wait(g.barrier_arrive())`。

- `grid_group::arrival_token barrier_arrive()`：到达 grid 栅栏，并返回一个需要传递给 `barrier_wait()` 的令牌。

- `void barrier_wait(grid_group::arrival_token&& t)`：等待 grid 栅栏；接收由 `barrier_arrive()` 返回的到达令牌的右值引用。

- `static unsigned long long thread_rank()`：调用线程在 `[0, num_threads)` 中的 rank。

- `static unsigned long long block_rank()`：调用线程块在 `[0, num_blocks)` 中的 rank。

- `static unsigned long long cluster_rank()`：调用 cluster 在 `[0, num_clusters)` 中的 rank。

- `static unsigned long long num_threads()`：组中的线程总数。

- `static unsigned long long num_blocks()`：组中的线程块总数。

- `static unsigned long long num_clusters()`：组中的 cluster 总数。

- `static dim3 dim_blocks()`：以线程块为单位表示的已启动 grid 尺寸。

- `static dim3 dim_clusters()`：以 cluster 为单位表示的已启动 grid 尺寸。

- `static dim3 block_index()`：线程块在已启动 grid 中的三维索引。

- `static dim3 cluster_index()`：cluster 在已启动 grid 中的三维索引。

旧版成员函数（别名）：

- `static unsigned long long size()`：组中的线程总数（`num_threads()` 的别名）。

- `static dim3 group_dim()`：已启动 grid 的尺寸（`dim_blocks()` 的别名）。

<a id="class-thread-block-tile"></a>
<span id="cg-api-thread-block-tile"></span>

#### 5.6.3.1.4 class thread_block_tile（`thread_block_tile` 类）

这是 tiled group 的模板版本：使用模板参数指定 tile 的大小。由于大小在编译时已知，因此有可能实现更优的执行。

```cpp
template <unsigned int Size, typename ParentT = void>
class thread_block_tile;
```

通过以下方式构造：

```cpp
template <unsigned int Size, typename ParentT>
_CG_QUALIFIER thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g)
```

`Size` 必须是 2 的幂，且小于或等于 1024。请参阅 Notes 部分，了解在计算能力 7.5 或更低的硬件上创建大小大于 32 的 tile 所需的额外步骤。

`ParentT` 是该组从中划分出来的父类型。它会自动推导；但如果其值为 `void`，则该信息会存储在组句柄中，而不是存储在类型中。

**公共成员函数：**

- `void sync() const`：同步组中指定的线程。

- `unsigned long long num_threads() const`：组中的线程总数。

- `unsigned long long thread_rank() const`：调用线程在 `[0, num_threads)` 中的 rank。

- `unsigned long long meta_group_size() const`：返回划分父组时创建的组数。

- `unsigned long long meta_group_rank() const`：该组在从父组划分出的 tile 集合中的线性 rank（受 `meta_group_size` 限制）。

- `T shfl(T var, unsigned int src_rank) const`：参见 [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions)。**注意：**对于大小大于 32 的 tile，组中所有线程必须指定相同的 `src_rank`，否则行为未定义。

- `T shfl_up(T var, int delta) const`：参见 [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions)；只适用于大小小于或等于 32 的 tile。

- `T shfl_down(T var, int delta) const`：参见 [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions)；只适用于大小小于或等于 32 的 tile。

- `T shfl_xor(T var, int delta) const`：参见 [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions)；只适用于大小小于或等于 32 的 tile。

- `int any(int predicate) const`：参见 [Warp Vote Functions](cpp-language-extensions.html#warp-vote-functions)。

- `int all(int predicate) const`：参见 [Warp Vote Functions](cpp-language-extensions.html#warp-vote-functions)。

- `unsigned int ballot(int predicate) const`：参见 [Warp Vote Functions](cpp-language-extensions.html#warp-vote-functions)；只适用于大小小于或等于 32 的 tile。

- `unsigned int match_any(T val) const`：参见 [Warp Match Functions](cpp-language-extensions.html#warp-match-functions)；只适用于大小小于或等于 32 的 tile。

- `unsigned int match_all(T val, int &pred) const`：参见 [Warp Match Functions](cpp-language-extensions.html#warp-match-functions)；只适用于大小小于或等于 32 的 tile。

旧版成员函数（别名）：

- `unsigned long long size() const`：组中的线程总数（`num_threads()` 的别名）。

**注意事项：**

- 这里使用 `thread_block_tile` 模板数据结构；组的大小作为模板参数传递给 `tiled_partition` 调用，而不是作为实参传递。

- 使用 C++11 或更高版本编译时，`shfl`、`shfl_up`、`shfl_down` 和 `shfl_xor` 函数可以接受任意类型的对象。这意味着只要满足下面的约束，就可以 shuffle 非整数类型：

  - 必须是可平凡复制的类型，即 `is_trivially_copyable<T>::value == true`。

  - 对于大小小于或等于 32 的 tile，`sizeof(T) <= 32`；对于更大的 tile，`sizeof(T) <= 8`。

- 在计算能力 7.5 或更低的硬件上，大小大于 32 的 tile 需要为其保留少量内存。可以使用 `cooperative_groups::block_tile_memory` 结构模板完成，该模板实例必须位于共享内存或全局内存中。

  ```cpp
  template <unsigned int MaxBlockSize = 1024>
  struct block_tile_memory;
  ```

  `MaxBlockSize` 指定当前线程块中的最大线程数。对于只以较小线程数启动的 kernel，可以用该参数减少 `block_tile_memory` 的共享内存占用。

  随后必须将这个 `block_tile_memory` 传递给 `cooperative_groups::this_thread_block`，这样得到的 `thread_block` 才能被划分为大小大于 32 的 tile。接收 `block_tile_memory` 参数的 `this_thread_block` 重载是集体操作，必须由 `thread_block` 中的所有线程调用。

  在计算能力 8.0 及更高的硬件上，也可以使用 `block_tile_memory`，从而让同一份源代码面向多种不同的计算能力。在共享内存中实例化时，如果不需要它，该对象应不占用内存。

**示例：**

```cpp
/// 以下代码将创建两组 tiled group，大小分别为 32 和 4：
/// 后者的来源信息编码在类型中，前者则将其存储在句柄中
thread_block block = this_thread_block();
thread_block_tile<32> tile32 = tiled_partition<32>(block);
thread_block_tile<4, thread_block> tile4 = tiled_partition<4>(block);
```

```cpp
/// 以下代码将在所有计算能力上创建大小为 128 的 tile。
/// 在计算能力 8.0 或更高版本上，可以省略 block_tile_memory。
__global__ void kernel(...) {
    // 为 thread_block_tile 使用保留共享内存，
    //   指定 block 大小最多为 256 个线程。
    __shared__ block_tile_memory<256> shared;
    thread_block thb = this_thread_block(shared);

    // 创建包含 128 个线程的 tile。
    auto tile = tiled_partition<128>(thb);

    // ...
}
```

<a id="class-coalesced-group"></a>
<span id="cg-api-coalesced-group"></span>

#### 5.6.3.1.5 class coalesced_group（`coalesced_group` 类）

在 CUDA 的 SIMT 架构中，多处理器在硬件层面以称为 warp 的 32 个线程为一组执行线程。如果应用代码中存在依赖数据的条件分支，使一个 warp 中的线程发生分歧，那么 warp 会依次执行每个分支，并禁用不在当前路径上的线程。仍处于当前路径上的线程称为 coalesced 线程。Cooperative Groups 提供了发现这些线程并创建包含所有 coalesced 线程的组的功能。

通过 `coalesced_threads()` 构造组句柄是一种机会式操作。它返回调用时刻处于活动状态的线程集合，不保证返回哪些线程（只要线程处于活动状态即可），也不保证这些线程在整个执行过程中始终保持 coalesced 状态（集体操作执行时它们会重新聚合，但之后仍可能再次分歧）。

```cpp
class coalesced_group;
```

通过以下方式构造：

```cpp
coalesced_group active = coalesced_threads();
```

**公共成员函数：**

- `void sync() const`：同步组中指定的线程。

- `unsigned long long num_threads() const`：组中的线程总数。

- `unsigned long long thread_rank() const`：调用线程在 `[0, num_threads)` 中的 rank。

- `unsigned long long meta_group_size() const`：返回划分父组时创建的组数。如果该组是通过查询活动线程集合创建的，例如通过 `coalesced_threads()` 创建，则 `meta_group_size()` 的值为 1。

- `unsigned long long meta_group_rank() const`：该组在从父组划分出的 tile 集合中的线性 rank（受 `meta_group_size` 限制）。如果该组是通过查询活动线程集合创建的，例如通过 `coalesced_threads()` 创建，则 `meta_group_rank()` 的值始终为 0。

- `T shfl(T var, unsigned int src_rank) const`：参见 [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions)。

- `T shfl_up(T var, int delta) const`：参见 [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions)。

- `T shfl_down(T var, int delta) const`：参见 [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions)。

- `int any(int predicate) const`：参见 [Warp Vote Functions](cpp-language-extensions.html#warp-vote-functions)。

- `int all(int predicate) const`：参见 [Warp Vote Functions](cpp-language-extensions.html#warp-vote-functions)。

- `unsigned int ballot(int predicate) const`：参见 [Warp Vote Functions](cpp-language-extensions.html#warp-vote-functions)。

- `unsigned int match_any(T val) const`：参见 [Warp Match Functions](cpp-language-extensions.html#warp-match-functions)。

- `unsigned int match_all(T val, int &pred) const`：参见 [Warp Match Functions](cpp-language-extensions.html#warp-match-functions)。

旧版成员函数（别名）：

- `unsigned long long size() const`：组中的线程总数（`num_threads()` 的别名）。

**注意事项：**

使用 C++11 或更高版本编译时，`shfl`、`shfl_up` 和 `shfl_down` 函数可以接受任意类型的对象。这意味着只要满足以下约束，就可以 shuffle 非整数类型：

- 必须是可平凡复制的类型，即 `is_trivially_copyable<T>::value == true`。

- `sizeof(T) <= 32`。

**示例：**

```cpp
/// 假设代码中有一个分支，其中每个 warp 只有第 2、4 和 8 个线程处于活动状态。
/// 放在该分支中的 coalesced_threads() 调用，会为每个 warp 创建一个名为 active 的组，
/// 其中有三个线程（rank 为 0 到 2）。
__global__ void kernel(int *globalInput) {
    // 假设 globalInput 表示线程 2、4、8 应处理数据
    if (threadIdx.x == *globalInput) {
        coalesced_group active = coalesced_threads();
        // active 包含 rank 0 到 2
        active.sync();
    }
}
```

<a id="cooperative-groups-async-h"></a>
<span id="cg-api-async-header"></span>

### 5.6.3.2 cooperative_groups/async.h

<a id="memcpy-async"></a>
<span id="cg-api-async-memcpy"></span>

#### 5.6.3.2.1 `memcpy_async`

`memcpy_async` 是组范围的集体 memcpy，利用硬件加速支持执行从全局内存到共享内存的非阻塞内存事务。给定组中指定的一组线程，`memcpy_async` 会通过单个流水线阶段移动指定数量的字节或输入类型的元素。为了使用 `memcpy_async` API 获得最佳性能，共享内存和全局内存都需要 16 字节对齐。需要注意的是，虽然一般情况下这是一个 memcpy，但只有当源是全局内存、目标是共享内存，且二者都可以按 16、8 或 4 字节对齐寻址时，它才是异步的。异步复制的数据只能在调用 `wait` 或 `wait_prior` 之后读取，因为这些调用会表明相应阶段已经完成了将数据移动到共享内存的操作。

等待所有未完成请求可能会失去一些灵活性（但会更简单）。为了高效地重叠数据传输和执行，需要能够在等待并处理请求 **N** 时，启动 **N+1** 个 `memcpy_async` 请求。为此，使用 `memcpy_async`，并通过基于集体阶段的 `wait_prior` API 等待它。更多信息请参阅 [`wait` 和 `wait_prior`](#wait-and-wait-prior)。

**用法 1：**

```cpp
template <typename TyGroup, typename TyElem, typename TyShape>
void memcpy_async(
  const TyGroup &group,
  TyElem *__restrict__ _dst,
  const TyElem *__restrict__ _src,
  const TyShape &shape
);
```

复制 **`shape` 个字节**。

**用法 2：**

```cpp
template <typename TyGroup, typename TyElem, typename TyDstLayout, typename TySrcLayout>
void memcpy_async(
  const TyGroup &group,
  TyElem *__restrict__ dst,
  const TyDstLayout &dstLayout,
  const TyElem *__restrict__ src,
  const TySrcLayout &srcLayout
);
```

复制 **`min(dstLayout, srcLayout)` 个元素**。如果布局类型为 `cuda::aligned_size_t<N>`，二者必须指定相同的对齐值。

**勘误：**CUDA 11.1 引入的、同时接受源布局和目标布局的 `memcpy_async` API，要求布局以元素而不是字节为单位提供。元素类型从 `TyElem` 推导，其大小为 `sizeof(TyElem)`。如果使用 `cuda::aligned_size_t<N>` 作为布局，则指定的元素数乘以 `sizeof(TyElem)` 必须是 N 的倍数，并且建议使用 `std::byte` 或 `char` 作为元素类型。

如果指定的复制 shape 或布局的类型为 `cuda::aligned_size_t<N>`，则保证对齐至少为 `min(16, N)`。在这种情况下，`dst` 和 `src` 指针都必须按 N 字节对齐，并且复制的字节数必须是 N 的倍数。

**代码生成要求：**最低计算能力 5.0；要实现异步性，需要计算能力 8.0；需要 C++11。

必须包含 `cooperative_groups/memcpy_async.h` 头文件。

**示例：**

```cpp
/// 此示例从全局内存中流式读取 elementsPerThreadBlock 个数据，
/// 送入大小受限的共享内存块 elementsInShared 进行处理。
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

__global__ void kernel(int* global_data) {
    cg::thread_block tb = cg::this_thread_block();
    const size_t elementsPerThreadBlock = 16 * 1024;
    const size_t elementsInShared = 128;
    __shared__ int local_smem[elementsInShared];

    size_t copy_count;
    size_t index = 0;
    while (index < elementsPerThreadBlock) {
        cg::memcpy_async(tb, local_smem, elementsInShared, global_data + index, elementsPerThreadBlock - index);
        copy_count = min(elementsInShared, elementsPerThreadBlock - index);
        cg::wait(tb);
        // 使用 local_smem
        index += copy_count;
    }
}
```

<a id="wait-and-wait-prior"></a>
<span id="cg-api-async-wait"></span>

#### 5.6.3.2.2 `wait` and `wait_prior`（`wait` 与 `wait_prior`）

```cpp
template <typename TyGroup>
void wait(TyGroup & group);

template <unsigned int NumStages, typename TyGroup>
void wait_prior(TyGroup & group);
```

`wait` 和 `wait_prior` 集体操作允许等待 `memcpy_async` 复制完成。`wait` 会阻塞调用线程，直到之前的所有复制都完成。`wait_prior` 允许最新的 `NumStages` 个复制仍未完成，同时等待更早的请求完成。因此，如果总共请求了 `N` 次复制，它会等待前 `N-NumStages` 次完成，而最后的 `NumStages` 次复制可能仍在进行。`wait` 和 `wait_prior` 都会同步指定的组。

**代码生成要求：**最低计算能力 5.0；要实现异步性，需要计算能力 8.0；需要 C++11。

必须包含 `cooperative_groups/memcpy_async.h` 头文件。

**示例：**

```cpp
/// 此示例从全局内存中流式读取 elementsPerThreadBlock 个数据，
/// 送入大小受限的共享内存块 elementsInShared，并以两个阶段进行处理。
/// 启动阶段 N 时，可以等待并处理阶段 N-1。
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

__global__ void kernel(int* global_data) {
    cg::thread_block tb = cg::this_thread_block();
    const size_t elementsPerThreadBlock = 16 * 1024 + 64;
    const size_t elementsInShared = 128;
    __align__(16) __shared__ int local_smem[2][elementsInShared];
    int stage = 0;
    // 首先启动一个额外请求
    size_t copy_count = elementsInShared;
    size_t index = copy_count;
    cg::memcpy_async(tb, local_smem[stage], elementsInShared, global_data, elementsPerThreadBlock - index);
    while (index < elementsPerThreadBlock) {
        // 现在启动下一次请求……
        cg::memcpy_async(tb, local_smem[stage ^ 1], elementsInShared, global_data + index, elementsPerThreadBlock - index);
        // ……但等待前一次请求
        cg::wait_prior<1>(tb);

        // 此时它已经可用，可以在这里处理 local_smem[stage]
        // (...)
        //

        // 计算实际复制的数据量，供下一次迭代使用。
        copy_count = min(elementsInShared, elementsPerThreadBlock - index);
        index += copy_count;

        // 根据 local_smem[stage] 上完成的工作是否允许线程提前执行，
        // 这里可能需要 cg::sync(tb)
        // 切换到下一阶段
        stage ^= 1;
    }
    cg::wait(tb);
    // 最后可以在这里处理 local_smem[stage]
}
```

<a id="cooperative-groups-partition-h"></a>
<span id="cg-api-partition-header"></span>

### 5.6.3.3 cooperative_groups/partition.h

<a id="tiled-partition"></a>
<span id="cg-api-partition-tiled"></span>

#### 5.6.3.3.1 `tiled_partition`

```cpp
template <unsigned int Size, typename ParentT>
thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g);
```

```cpp
thread_group tiled_partition(const thread_group& parent, unsigned int tilesz);
```

`tiled_partition` 方法是一个集体操作，会把父组划分为一维、行主序排列的子组 tile。总共会创建 `size(parent) / tilesz` 个子组，因此父组大小必须能被 `Size` 整除。允许的父组是 `thread_block` 或 `thread_block_tile`。

在所有父组成员调用该操作之前，实现可能让调用线程等待，之后才恢复执行。该功能受原生硬件大小限制，即 `1/2/4/8/16/32`；并且 `cg::size(parent)` 必须大于 `Size` 参数。模板版本的 `tiled_partition` 还支持大小 `64/128/256/512`，但在计算能力 7.5 或更低的硬件上需要额外步骤，详情请参阅 [`class thread_block_tile`](#class-thread-block-tile)。

**代码生成要求：**最低计算能力 5.0；对于大于 32 的大小，需要 C++11。

<a id="labeled-partition"></a>
<span id="cg-api-partition-labeled"></span>

#### 5.6.3.3.2 `labeled_partition`

```cpp
template <typename Label>
coalesced_group labeled_partition(const coalesced_group& g, Label label);
```

```cpp
template <unsigned int Size, typename Label>
coalesced_group labeled_partition(const thread_block_tile<Size>& g, Label label);
```

`labeled_partition` 方法是一个集体操作，会将父组划分为一维子组，并保证每个子组中的线程是 coalesced 的。实现会计算条件标签，并把标签值相同的线程分配到同一组。

`Label` 可以是任意整型。

在所有父组成员调用该操作之前，实现可能让调用线程等待，之后才恢复执行。

**注意：**该功能仍在评估中，未来可能会发生轻微变化。

**代码生成要求：**最低计算能力 7.0，需要 C++11。

<a id="binary-partition"></a>
<span id="cg-api-partition-binary"></span>

#### 5.6.3.3.3 `binary_partition`

```cpp
coalesced_group binary_partition(const coalesced_group& g, bool pred);
```

```cpp
template <unsigned int Size>
coalesced_group binary_partition(const thread_block_tile<Size>& g, bool pred);
```

`binary_partition()` 方法是一个集体操作，会将父组划分为一维子组，并保证每个子组中的线程是 coalesced 的。实现会计算谓词，并把谓词值相同的线程分配到同一组。这是 `labeled_partition()` 的特化形式，其中标签只能是 `0` 或 `1`。

在所有父组成员调用该操作之前，实现可能让调用线程等待，之后才恢复执行。

**示例：**

```cpp
/// 此示例将大小为 32 的 tile 划分为奇数值组和偶数值组
_global__ void oddEven(int *inputArr) {
    auto block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);

    // inputArr 包含随机整数
    int elem = inputArr[block.thread_rank()];
    // 此后，tile32 被划分为 2 个组：
    // 一个子 tile 中 elem&1 为 true，另一个子 tile 中为 false
    auto subtile = cg::binary_partition(tile32, (elem & 1));
}
```

<a id="cooperative-groups-reduce-h"></a>
<span id="cg-api-reduce-header"></span>

### 5.6.3.4 cooperative_groups/reduce.h

<a id="reduce-operators"></a>
<span id="cg-api-reduce-operators"></span>

#### 5.6.3.4.1 `Reduce` Operators（`Reduce` 运算符）

下面是一些可用于 `reduce` 基本操作的函数对象原型。

```cpp
namespace cooperative_groups {
  template <typename Ty>
  struct cg::plus;

  template <typename Ty>
  struct cg::less;

  template <typename Ty>
  struct cg::greater;

  template <typename Ty>
  struct cg::bit_and;

  template <typename Ty>
  struct cg::bit_xor;

  template <typename Ty>
  struct cg::bit_or;
}
```

Reduce 受限于实现能够在编译时获得的信息。因此，为了使用计算能力 8.0 中引入的内建函数，`cg::` 命名空间公开了几个与硬件对应的函数对象。这些对象看起来类似于 C++ STL 中提供的对象，但 `less/greater` 除外。它们与 STL 存在差异的原因是：这些函数对象的设计目标是实际对应硬件内建函数的操作。

**函数语义：**

- `cg::plus`：接收两个值，并使用 `operator+` 返回二者之和。

- `cg::less`：接收两个值，并使用 `operator<` 返回较小者。区别在于它**返回较小的值**，而不是布尔值。

- `cg::greater`：接收两个值，并使用 `operator<` 返回较大者。区别在于它**返回较大的值**，而不是布尔值。

- `cg::bit_and`：接收两个值，并返回 `operator&` 的结果。

- `cg::bit_xor`：接收两个值，并返回 `operator^` 的结果。

- `cg::bit_or`：接收两个值，并返回 `operator|` 的结果。

**示例：**

```cpp
{
    // cg::plus<int> 在 cg::reduce 中有特化，并在 CC 8.0+ 上调用 __reduce_add_sync(...)
    cg::reduce(tile, (int)val, cg::plus<int>());

    // cg::plus<float> 无法匹配加速器，因此执行基于标准 shuffle 的归约
    cg::reduce(tile, (float)val, cg::plus<float>());

    // 虽然支持向量的单个分量，但 reduce 不会对以下类型使用硬件内建函数
    // 对于向量和可能使用的自定义类型，还需要定义对应的运算符
    int4 vec = {...};
    cg::reduce(tile, vec, cg::plus<int4>());

    // 最后，lambda 和其他函数对象无法被检查以进行分派，
    // 因此会使用提供的函数对象执行基于 shuffle 的归约。
    cg::reduce(tile, (int)val, [](int l, int r) -> int {return l + r;});
}
```

<a id="reduce"></a>
<span id="cg-api-reduce-reduce-function"></span>

#### 5.6.3.4.2 `reduce`

```cpp
template <typename TyGroup, typename TyArg, typename TyOp>
auto reduce(const TyGroup& group, TyArg&& val, TyOp&& op) -> decltype(op(val, val));
```

`reduce` 对传入组中各线程提供的数据执行归约操作。在计算能力 8.0 及更高的设备上，对于算术加法、最小值或最大值，以及逻辑 AND、OR 或 XOR，该操作会利用硬件加速；在较旧代际的硬件上则提供软件回退。只有 4 字节类型能获得硬件加速。

`group`：有效的组类型是 `coalesced_group` 和 `thread_block_tile`。

`val`：满足以下要求的任意类型：

- 是可平凡复制的类型，即 `is_trivially_copyable<TyArg>::value == true`。

- 对于 `coalesced_group` 和大小小于或等于 32 的 tile，`sizeof(T) <= 32`；对于更大的 tile，`sizeof(T) <= 8`。

- 具有适用于给定函数对象的算术或比较运算符。

**注意：**组中的不同线程可以为该参数传递不同的值。

`op`：对整型提供硬件加速的有效函数对象包括 `plus()`、`less()`、`greater()`、`bit_and()`、`bit_xor()` 和 `bit_or()`。这些函数对象必须先构造，因此需要 `TyVal` 模板参数，例如 `plus<int>()`。Reduce 也支持可以通过 `operator()` 调用的 lambda 和其他函数对象。

**异步归约：**

```cpp
template <typename TyGroup, typename TyArg, typename TyAtomic, typename TyOp>
void reduce_update_async(const TyGroup& group, TyAtomic& atomic, TyArg&& val, TyOp&& op);

template <typename TyGroup, typename TyArg, typename TyAtomic, typename TyOp>
void reduce_store_async(const TyGroup& group, TyAtomic& atomic, TyArg&& val, TyOp&& op);

template <typename TyGroup, typename TyArg, typename TyOp>
void reduce_store_async(const TyGroup& group, TyArg* ptr, TyArg&& val, TyOp&& op);
```

API 的 `*_async` 变体会异步计算结果，并由参与线程中的一个线程将结果存储到或更新到指定目标，而不是由每个线程返回结果。要观察这些异步调用的效果，必须同步调用线程组，或者同步包含这些线程的更大组。

- 对于原子存储或原子更新变体，`atomic` 参数可以是 [CUDA C++ 标准库](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives.html)中提供的 `cuda::atomic` 或 `cuda::atomic_ref`。该 API 变体只在 CUDA C++ 标准库支持这些类型的平台和设备上可用。归约结果会根据指定的 `op` 原子地更新 atomic，例如在使用 `cg::plus()` 时，结果会以原子方式加到 atomic 上。`atomic` 持有的类型必须与 `TyArg` 的类型匹配。atomic 的作用域必须包含组中的所有线程；如果多个组并发使用同一个 atomic，则该作用域必须包含所有使用它的组中的全部线程。原子更新使用 relaxed 内存顺序执行。

- 对于指针存储变体，归约结果会弱存储到 `dst` 指针中。

<a id="cooperative-groups-scan-h"></a>
<span id="cg-api-scan-header"></span>

### 5.6.3.5 cooperative_groups/scan.h

<a id="inclusive-scan-and-exclusive-scan"></a>
<span id="cg-api-scan-functions"></span>

#### 5.6.3.5.1 `inclusive_scan` and `exclusive_scan`（`inclusive_scan` 与 `exclusive_scan`）

```cpp
template <typename TyGroup, typename TyVal, typename TyFn>
auto inclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

template <typename TyGroup, typename TyVal>
TyVal inclusive_scan(const TyGroup& group, TyVal&& val);

template <typename TyGroup, typename TyVal, typename TyFn>
auto exclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

template <typename TyGroup, typename TyVal>
TyVal exclusive_scan(const TyGroup& group, TyVal&& val);
```

`inclusive_scan` 和 `exclusive_scan` 对传入组中各线程提供的数据执行 scan 操作。对于 `exclusive_scan`，每个线程的结果是 rank 小于该线程的线程所提供数据的归约；`inclusive_scan` 的结果还包括调用线程自己的数据。

`group`：有效的组类型是 `coalesced_group` 和 `thread_block_tile`。

`val`：满足以下要求的任意类型：

- 是可平凡复制的类型，即 `is_trivially_copyable<TyArg>::value == true`。

- 对于 `coalesced_group` 和大小小于或等于 32 的 tile，`sizeof(T) <= 32`；对于更大的 tile，`sizeof(T) <= 8`。

- 具有适用于给定函数对象的算术或比较运算符。

**注意：**组中的不同线程可以为该参数传递不同的值。

`op`：为方便使用而定义的函数对象包括 `plus()`、`less()`、`greater()`、`bit_and()`、`bit_xor()` 和 `bit_or()`，详见 [`cooperative_groups/reduce.h`](#cg-api-reduce-header)。这些函数对象必须先构造，因此需要 `TyVal` 模板参数，例如 `plus<int>()`。`inclusive_scan` 和 `exclusive_scan` 也支持可以通过 `operator()` 调用的 lambda 和其他函数对象。不带该参数的重载使用 `cg::plus<TyVal>()`。

**Scan 更新：**

```cpp
template <typename TyGroup, typename TyAtomic, typename TyVal, typename TyFn>
auto inclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

template <typename TyGroup, typename TyAtomic, typename TyVal>
TyVal inclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val);

template <typename TyGroup, typename TyAtomic, typename TyVal, typename TyFn>
auto exclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

template <typename TyGroup, typename TyAtomic, typename TyVal>
TyVal exclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val);
```

`*_scan_update` 集体操作额外接收一个 `atomic` 参数，它可以是 [CUDA C++ 标准库](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives.html)提供的 `cuda::atomic` 或 `cuda::atomic_ref`。这些 API 变体只在 CUDA C++ 标准库支持这些类型的平台和设备上可用。它们会按照 `op`，使用组中所有线程输入值之和更新 `atomic`。每个线程会将 `atomic` 的旧值与 scan 结果合并后返回。`atomic` 持有的类型必须与 `TyVal` 的类型匹配。atomic 的作用域必须包含组中的所有线程；如果多个组并发使用同一个 atomic，则该作用域必须包含所有使用它的组中的全部线程。原子更新使用 relaxed 内存顺序执行。

下面的伪代码说明 scan 的更新变体如何工作：

```cpp
/*
 inclusive_scan_update 的行为如下方代码块所示，
 但 reduce 和 inclusive_scan 会同时计算。
auto total = reduce(group, val, op);
TyVal old;
if (group.thread_rank() == selected_thread) {
    atomically {
        old = atomic.load();
        atomic.store(op(old, total));
    }
}
old = group.shfl(old, selected_thread);
return op(inclusive_scan(group, val, op), old);
*/
```

必须包含 `cooperative_groups/scan.h` 头文件。

**使用 `exclusive_scan` 进行流压缩的示例：**

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

// 只有通过 test_fn 谓词测试的数据才从 input 放入 output
template<typename Group, typename Data, typename TyFn>
__device__ int stream_compaction(Group &g, Data *input, int count, TyFn&& test_fn, Data *output) {
    int per_thread = count / g.num_threads();
    int thread_start = min(g.thread_rank() * per_thread, count);
    int my_count = min(per_thread, count - thread_start);

    // 从 input 的本线程部分获取所有通过测试的元素，
    // 将它们放入数组中连续的一段，并统计数量。
    int i = thread_start;
    while (i < my_count + thread_start) {
        if (test_fn(input[i])) {
            i++;
        }
        else {
            my_count--;
            input[i] = input[my_count + thread_start];
        }
    }

    // 对各线程的计数执行 scan，计算本线程在输出中的起始索引
    int my_idx = cg::exclusive_scan(g, my_count);

    for (i = 0; i < my_count; ++i) {
        output[my_idx + i] = input[thread_start + i];
    }
    // 返回 output 中元素的总数
    return g.shfl(my_idx + my_count, g.num_threads() - 1);
}
```

**使用 `exclusive_scan_update` 动态分配缓冲区空间的示例：**

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

// 为便于理解，缓冲区划分采用静态方式；
// 替换此函数即可实现任意动态分配方案。
__device__ int calculate_buffer_space_needed(cg::thread_block_tile<32>& tile) {
    return tile.thread_rank() % 2 + 1;
}

__device__ int my_thread_data(int i) {
    return i;
}

__global__ void kernel() {
    __shared__ extern int buffer[];
    __shared__ cuda::atomic<int, cuda::thread_scope_block> buffer_used;

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    buffer_used = 0;
    block.sync();

    // 每个线程计算所需的缓冲区大小
    int buf_needed = calculate_buffer_space_needed(tile);

    // 对每个线程的需求执行 scan，结果是该线程部分的偏移量。
    // buffer_used 以原子方式更新为所有线程输入之和，
    // 以便正确偏移其他 tile 的分配位置。
    int buf_offset =
        cg::exclusive_scan_update(tile, buffer_used, buf_needed);

    // 每个线程使用线程特定的数据填充自己的缓冲区部分
    for (int i = 0 ; i < buf_needed ; ++i) {
        buffer[buf_offset + i] = my_thread_data(i);
    }

    block.sync();
    // buffer_used 现在保存已分配内存的总量
    // buffer 为 {0, 0, 1, 0, 0, 1 ...};
}
```

<a id="cooperative-groups-sync-h"></a>
<span id="cg-api-sync-header"></span>

### 5.6.3.6 cooperative_groups/sync.h

<a id="barrier-arrive-and-barrier-wait"></a>
<span id="cg-api-sync-barrier"></span>

#### 5.6.3.6.1 `barrier_arrive` and `barrier_wait`（`barrier_arrive` 与 `barrier_wait`）

```cpp
T::arrival_token T::barrier_arrive();
void T::barrier_wait(T::arrival_token&&);
```

`barrier_arrive` 和 `barrier_wait` 成员函数提供类似于 [`cuda::barrier`](../04-cuda-features/asynchronous-barriers.html#asynchronous-barriers) 的同步 API。Cooperative Groups 会自动初始化组栅栏，但由于这些操作具有集体性质，到达和等待操作还存在一项额外限制：每个阶段中，组内所有线程都必须在栅栏处到达一次并等待一次。当使用一个组调用 `barrier_arrive` 后，在通过 `barrier_wait` 调用观察到该栅栏阶段完成之前，对该组调用任何集体操作或再次到达栅栏的结果都是未定义的。阻塞在 `barrier_wait` 上的线程可能会在其他线程调用 `barrier_wait` 之前从同步中释放，但前提是组中的所有线程都已经调用了 `barrier_arrive`。组类型 `T` 可以是任意<a href="../04-cuda-features/cooperative-groups.html#cooperative-groups-implicit-groups">隐式组</a>。这允许线程在到达后、等待同步完成前执行独立工作，从而隐藏一部分同步延迟。`barrier_arrive` 返回一个 `arrival_token` 对象，必须将它传递给对应的 `barrier_wait`。令牌以这种方式被消耗，不能再次用于其他 `barrier_wait` 调用。

**使用 `barrier_arrive` 和 `barrier_wait` 跨 cluster 同步共享内存初始化的示例：**

```cpp
#include <cooperative_groups.h>

using namespace cooperative_groups;

void __device__ init_shared_data(const thread_block& block, int *data);
void __device__ local_processing(const thread_block& block);
void __device__ process_shared_data(const thread_block& block, int *data);

__global__ void cluster_kernel() {
    extern __shared__ int array[];
    auto cluster = this_cluster();
    auto block   = this_thread_block();

    // 使用此线程块初始化共享状态
    init_shared_data(block, &array[0]);

    auto token = cluster.barrier_arrive(); // 让其他线程块知道此线程块正在运行且数据已初始化

    // 执行一些本地处理，以隐藏同步延迟
    local_processing(block);

    // 将 cluster 中下一个线程块的共享内存映射到本地地址
    int *dsmem = cluster.map_shared_rank(&array[0], (cluster.block_rank() + 1) % cluster.num_blocks());

    // 确保 cluster 中所有其他线程块都在运行且已初始化共享数据，然后再访问 dsmem
    cluster.barrier_wait(std::move(token));

    // 使用分布式共享内存中的数据
    process_shared_data(block, dsmem);
    cluster.sync();
}
```

<a id="sync"></a>
<span id="cg-api-sync-function"></span>

#### 5.6.3.6.2 `sync`

```cpp
static void T::sync();

template <typename T>
void sync(T& group);
```

`sync` 同步组中指定的线程。组类型 `T` 可以是现有的任意组类型，因为所有组类型都支持同步。它既可以作为每种组类型的成员函数使用，也可以作为接收组参数的自由函数使用。

<a id="grid-synchronization"></a>
<span id="cg-api-sync-grid"></span>

##### 5.6.3.6.2.1 Grid Synchronization（Grid 同步）

在 Cooperative Groups 引入之前，CUDA 编程模型只允许在线程块完成 kernel 的边界处进行线程块之间的同步。kernel 边界会隐式地使状态失效，并可能带来性能影响。

例如，在某些使用场景中，应用包含大量小 kernel，每个 kernel 表示处理流水线中的一个阶段。当前 CUDA 编程模型要求存在这些 kernel，以确保负责一个流水线阶段的线程块已经产生数据，然后负责下一阶段的线程块才可以使用这些数据。在这类场景中，如果能够提供全局的线程块间同步，应用就可以重构为使用持久线程块；当某个阶段完成时，这些线程块能够在 device 上同步。

在 kernel 内跨 grid 同步时，只需使用 `grid.sync()` 函数：

```cpp
grid_group grid = this_grid();
grid.sync();
```

启动 kernel 时，不能使用 `<<<...>>>` 执行配置语法，而必须使用 CUDA runtime 的 `cudaLaunchCooperativeKernel` 启动 API，或使用等价的 CUDA driver API。

**示例：**

为了保证线程块在 GPU 上共同驻留，需要仔细考虑启动的线程块数量。例如，可以启动与 SM 数量相同的线程块：

```cpp
int dev = 0;
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, dev);
// 初始化，然后启动
cudaLaunchCooperativeKernel((void*)my_kernel, deviceProp.multiProcessorCount, numThreads, args);
```

或者，可以使用占用率计算器计算每个 SM 可以同时容纳多少线程块，从而最大化暴露的并行度：

```cpp
/// 在默认 stream 上启动一个可以最大程度填满 GPU 的 grid，并传入 kernel 参数
int numBlocksPerSm = 0;
 // my_kernel 启动时使用的线程数
int numThreads = 128;
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, dev);
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, my_kernel, numThreads, 0);
// 启动
void *kernelArgs[] = { /* add kernel args */ };
dim3 dimBlock(numThreads, 1, 1);
dim3 dimGrid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);
cudaLaunchCooperativeKernel((void*)my_kernel, dimGrid, dimBlock, kernelArgs);
```

良好的实践是先查询设备属性 `cudaDevAttrCooperativeLaunch`，确保设备支持 cooperative launch：

```cpp
int dev = 0;
int supportsCoopLaunch = 0;
cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
```

如果设备 0 支持该属性，上述调用会将 `supportsCoopLaunch` 设置为 1。只支持计算能力 6.0 及更高的设备。此外，还必须运行在以下平台之一：

- 没有 MPS 的 Linux 平台。

- 使用 MPS 且设备计算能力为 7.0 或更高的 Linux 平台。

- 最新的 Windows 平台。

**注意：**组中的所有线程都必须参与集体操作，否则行为未定义。

**相关内容：**`thread_block` 数据类型派生自更通用的 `thread_group` 数据类型；后者可以表示更广泛的一类组。

<a id="cuda-device-runtime"></a>
<span id="id3"></span>

## 5.6.4 CUDA Device Runtime（CUDA Device Runtime）

CUDA device runtime 是可在 kernel 代码中使用的 API，提供许多与 host 端 CUDA Runtime API 相同的能力。这些 API 最常用于 [CUDA Dynamic Parallelism](../04-cuda-features/cuda-dynamic-parallelism.html#cuda-dynamic-parallelism) 或 [Device Graph Launch](../04-cuda-features/cuda-graphs.html#cuda-graphs-device-graph-launch) 场景。

<a id="including-device-runtime-api-in-cuda-code"></a>
<span id="device-runtime-including"></span>

### 5.6.4.1 Including Device Runtime API in CUDA Code（在 CUDA 代码中包含 Device Runtime API）

与 host 端 runtime API 类似，CUDA device runtime API 的函数原型会在程序编译期间自动包含。不需要显式包含 `cuda_device_runtime_api.h`。

<a id="memory-in-the-cuda-device-runtime"></a>
<span id="device-runtime-memory"></span>

### 5.6.4.2 Memory in the CUDA Device Runtime（CUDA Device Runtime 中的内存）

<a id="configuration-options"></a>
<span id="device-runtime-configuration-options"></span>

#### 5.6.4.2.1 Configuration Options（配置选项）

Device runtime 系统软件的资源分配由 host 程序通过 `cudaDeviceSetLimit()` API 控制。必须在启动任何 kernel 之前设置限制，并且在 GPU 正在运行程序时不能更改限制。

可以设置以下命名限制：

| 限制 | 行为 |
|---|---|
| `cudaLimitDevRuntimePendingLaunchCount` | 控制为尚未开始执行的 kernel 启动和事件缓冲预留的内存量；这些操作可能因为依赖尚未解决或缺少执行资源而尚未开始执行。缓冲区满时，在 device 端 kernel 启动期间尝试分配启动槽会失败并返回 `cudaErrorLaunchOutOfResources`；尝试分配事件槽会失败并返回 `cudaErrorMemoryAllocation`。默认启动槽数量为 2048。应用可以设置 `cudaLimitDevRuntimePendingLaunchCount`，增加启动槽和/或事件槽数量。分配的事件槽数量是该限制值的两倍。 |
| `cudaLimitStackSize` | 控制每个 GPU 线程的栈大小（单位为字节）。CUDA driver 会根据需要，在每次 kernel 启动时自动增大每线程栈大小。每次启动后，这个大小不会重置为原始值。要把每线程栈大小设置为其他值，可以调用 `cudaDeviceSetLimit()` 设置此限制。栈会立即调整大小；如有必要，device 会阻塞，直到所有先前请求的任务完成。可以调用 `cudaDeviceGetLimit()` 获取当前每线程栈大小。 |

<a id="allocation-and-lifetime"></a>
<span id="device-runtime-memory-allocation-and-lifetime"></span>

#### 5.6.4.2.2 Allocation and Lifetime（分配与生命周期）

在 host 和 device 环境中，`cudaMalloc()` 与 `cudaFree()` 具有不同语义。在 host 中调用时，`cudaMalloc()` 从未使用的 device 内存中分配一个新区域；从 device runtime 调用时，这些函数会映射到 device 端的 `malloc()` 和 `free()`。这意味着在 device 环境中，可分配内存总量受 device 端 `malloc()` 堆大小限制，该大小可能小于可用的未使用 device 内存。此外，在 host 程序中对由 device 上的 `cudaMalloc()` 分配的指针调用 `cudaFree()`，或反过来操作，都是错误的。

|  | Host 上的 `cudaMalloc()` | Device 上的 `cudaMalloc()` |
|---|---|---|
| Host 上的 `cudaFree()` | 支持 | 不支持 |
| Device 上的 `cudaFree()` | 不支持 | 支持 |
| 分配限制 | 可用 device 内存 | `cudaLimitMallocHeapSize` |

<a id="memory-declarations"></a>
<span id="id4"></span>

##### 5.6.4.2.2.1 Memory Declarations（内存声明）

<a id="device-and-constant-memory"></a>
<span id="id5"></span>

###### 5.6.4.2.2.1.1 Device and Constant Memory（Device 内存与常量内存）

在文件作用域使用 `__device__` 或 `__constant__` 内存空间说明符声明的内存，在使用 device runtime 时具有相同的行为。所有 kernel 都可以读写 device 变量，无论该 kernel 最初由 host 还是 device runtime 启动。等价地，所有 kernel 对模块作用域声明的 `__constant__` 都具有相同的视图。

<a id="textures-and-surfaces"></a>
<span id="id6"></span>

###### 5.6.4.2.2.1.2 Textures and Surfaces（纹理与表面）

> Device runtime 不允许在 device 代码中创建或销毁纹理对象或表面对象。由 host 创建的纹理对象和表面对象可以在 device 上自由使用和传递。无论动态创建的位置在哪里，动态创建的纹理对象始终有效，并且可以从 parent 传递给 child kernel。

> **注意：**device runtime 不支持在由 device 启动的 kernel 中使用旧式模块作用域纹理和表面（即计算能力 2.0 或 Fermi 风格的纹理和表面）。模块作用域的旧式纹理可以由 host 创建，并像其他 kernel 一样在 device 代码中使用；但只能由顶层 kernel 使用（即由 host 启动的 kernel）。

<a id="shared-memory-variable-declarations"></a>
<span id="id7"></span>

###### 5.6.4.2.2.1.3 Shared Memory Variable Declarations（共享内存变量声明）

在 CUDA C++ 中，共享内存既可以声明为静态大小的文件作用域变量或函数作用域变量，也可以声明为 `extern` 变量，其大小由 kernel 调用方通过启动配置参数在运行时确定。这两种声明在 device runtime 下都有效。

```cpp
__global__ void permute(int n, int *data) {
   extern __shared__ int smem[];
   if (n <= 1)
       return;

   smem[threadIdx.x] = data[threadIdx.x];
   __syncthreads();

   permute_data(smem, n);
   __syncthreads();

   // 由于不能将 SMEM 传递给子 kernel，因此写回 GMEM。
   data[threadIdx.x] = smem[threadIdx.x];
   __syncthreads();

   if (threadIdx.x == 0) {
       permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data);
       permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data+n/2);
   }
}

void host_launch(int *data) {
    permute<<< 1, 256, 256*sizeof(int) >>>(256, data);
}
```

<a id="constant-memory"></a>
<span id="id8"></span>

###### 5.6.4.2.2.1.4 Constant Memory（常量内存）

不能从 device 修改常量。常量只能从 host 修改；但如果某个 grid 正在并发访问该常量，则 host 在该常量生命周期中的任意时刻修改它，行为都是未定义的。

<a id="symbol-addresses"></a>
<span id="id9"></span>

###### 5.6.4.2.2.1.5 Symbol Addresses（符号地址）

Device 端符号（即标记为 `__device__` 的符号）可以在 kernel 中直接通过 `&` 运算符引用，因为所有全局作用域的 device 变量都处于 kernel 可见的地址空间中。`__constant__` 符号也适用这一规则，但在这种情况下，指针会指向只读数据。

由于 device 端符号可以直接引用，那些用于引用符号的 CUDA runtime API（例如 `cudaMemcpyToSymbol()` 或 `cudaGetSymbolAddress()`）就没有必要，并且 device runtime 不支持这些 API。这意味着即使在启动 child kernel 之前，也不能在运行中的 kernel 内修改常量数据，因为对 `__constant__` 空间的引用是只读的。

<a id="sm-id-and-warp-id"></a>
<span id="id10"></span>

### 5.6.4.3 SM Id and Warp Id（SM ID 与 Warp ID）

请注意，在 PTX 中，`%smid` 和 `%warpid` 被定义为 volatile 值。Device runtime 可能会将线程块重新调度到不同的 SM 上，以更高效地管理资源。因此，依赖 `%smid` 或 `%warpid` 在某个线程或线程块的生命周期内保持不变是不安全的。

<a id="launch-setup-apis"></a>
<span id="id11"></span>

### 5.6.4.4 Launch Setup APIs（启动设置 API）

[Device-Side Kernel Launch](../04-cuda-features/cuda-dynamic-parallelism.html#dynamic-parallelism-device-runtime-kernel-launch) 描述了从 device 代码启动 kernel 的语法；它使用与 host CUDA Runtime API 相同的三尖括号启动表示法。

Kernel 启动是通过 device runtime library 暴露的系统级机制。也可以通过 PTX 直接使用 `cudaGetParameterBuffer()` 和 `cudaLaunchDevice()` API。CUDA 应用可以自行调用这些 API，并遵循与 PTX 相同的要求。在这两种情况下，用户都必须按照规范以正确格式正确填充所有必需的数据结构。这些数据结构保证向后兼容。

与 host 端启动一样，device 端运算符 `<<<>>>` 会映射到底层 kernel 启动 API。这使得面向 PTX 的用户能够执行启动操作。NVCC 前端会将 `<<<>>>` 转换为这些调用。

| Runtime API 启动函数 | 与 Host Runtime 行为的差异（没有列出差异时行为相同） |
|---|---|
| `cudaGetParameterBuffer` | 自动从 `<<<>>>` 生成。注意：该 API 与 host 端等价 API 不同。 |
| `cudaLaunchDevice` | 自动从 `<<<>>>` 生成。注意：该 API 与 host 端等价 API 不同。 |

表 61：仅 Device 启动实现的新函数。

这些启动函数的 API 与 CUDA Runtime API 的 API 不同，定义如下：

```cpp
extern   device   cudaError_t cudaGetParameterBuffer(void **params);
extern __device__ cudaError_t cudaLaunchDevice(void *kernel,
                                        void *params, dim3 gridDim,
                                        dim3 blockDim,
                                        unsigned int sharedMemSize = 0,
                                        cudaStream_t stream = 0);
```

<a id="device-management"></a>

### 5.6.4.5 Device Management（设备管理）

Device runtime 不支持多 GPU；device runtime 只能在当前正在执行它的设备上操作。不过，可以查询系统中任意支持 CUDA 的设备的属性。

<a id="api-reference"></a>
<span id="device-runtime-api-reference"></span>

### 5.6.4.6 API Reference（API 参考）

这里详细说明 CUDA Runtime API 中由 device runtime 支持的部分。Host 和 device runtime API 具有相同的语法；除非另有说明，二者语义相同。下表概述了这些 API 与 host 端可用版本之间的关系。

表 62：支持的 API 函数。

| Runtime API 函数 | 详细信息 |
|---|---|
| `cudaDeviceGetCacheConfig` | |
| `cudaDeviceGetLimit` | |
| `cudaGetLastError` | Last error 是每线程状态，而不是每线程块状态。 |
| `cudaPeekAtLastError` | |
| `cudaGetErrorString` | |
| `cudaGetDeviceCount` | |
| `cudaDeviceGetAttribute` | 可以返回任意设备的属性。 |
| `cudaGetDevice` | 始终返回从 host 端观察到的当前设备 ID。 |
| `cudaStreamCreateWithFlags` | 必须传入 `cudaStreamNonBlocking` 标志。 |
| `cudaStreamDestroy` | |
| `cudaStreamWaitEvent` | |
| `cudaEventCreateWithFlags` | 必须传入 `cudaEventDisableTiming` 标志。 |
| `cudaEventRecord` | |
| `cudaEventDestroy` | |
| `cudaFuncGetAttributes` | |
| `cudaMemcpyAsync` | 对所有 `memcpy/memset` 函数：只支持异步 `memcpy/set` 函数；只允许 device-to-device `memcpy`；不能传入 local 或 shared memory 指针。 |
| `cudaMemcpy2DAsync` | 同上。 |
| `cudaMemcpy3DAsync` | 同上。 |
| `cudaMemsetAsync` | 同上。 |
| `cudaMemset2DAsync` | |
| `cudaMemset3DAsync` | |
| `cudaRuntimeGetVersion` | |
| `cudaMalloc` | 不能在 device 上对 host 创建的指针调用 `cudaFree`，反之亦然。 |
| `cudaFree` | 不能在 device 上对 host 创建的指针调用 `cudaFree`，反之亦然。 |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | |
| `cudaOccupancyMaxPotentialBlockSize` | |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMem` | |

<a id="api-errors-and-launch-failures"></a>
<span id="id12"></span>

### 5.6.4.7 API Errors and Launch Failures（API 错误与启动失败）

与 CUDA runtime 的通常行为一样，任意函数都可能返回错误代码。最近一次返回的错误代码会被记录，并可以通过调用 `cudaGetLastError()` 获取。错误按线程记录，因此每个线程都能识别自己最近生成的错误。错误代码的类型为 `cudaError_t`。

与 host 端启动类似，device 端启动可能因多种原因失败（参数无效等）。用户必须调用 `cudaGetLastError()` 来确定启动是否产生错误；但是，启动后没有错误并不意味着 child kernel 已成功完成。

对于 device 端异常，例如访问无效地址，child grid 中的错误会返回给 host。

<a id="device-runtime-streams"></a>
<span id="id13"></span>

### 5.6.4.8 Device Runtime Streams（Device Runtime Stream）

CUDA device runtime 暴露了特殊的命名 stream，为从 device 启动的 kernel 和 graph 提供特定行为。与 device graph launch 相关的命名 stream 记录在 [Device Launch](../04-cuda-features/cuda-graphs.html#cuda-graphs-device-graph-device-launch) 中。CUDA device runtime 中还可以用于 kernel 和 memcpy 操作的两个命名 stream 是 `cudaStreamFireAndForget` 和 `cudaStreamTailLaunch`。源文在此处重复列出 `cudaStreamTailLaunch`；结合后文 Fire-and-Forget 小节，第二项应为 `cudaStreamFireAndForget`。这些命名 stream 的具体行为记录在本节中。

命名 stream 和未命名（NULL）stream 都可由 device runtime 使用。不能将 stream 句柄传递给 parent grid 或 child grid。换句话说，应将 stream 视为创建它的 grid 私有的资源。

Host 端 NULL stream 的跨 stream 栅栏语义在 device 上不受支持（详情见下文）。为了保留与 host runtime 的语义兼容性，所有 device stream 都必须使用 `cudaStreamCreateWithFlags()` API 创建，并传入 `cudaStreamNonBlocking` 标志。CUDA device runtime 不提供 `cudaStreamCreate()` API。

由于 device runtime 不支持 `cudaStreamSynchronize()` 和 `cudaStreamQuery()`，当应用需要知道 stream 启动的 child kernel 已完成时，应使用启动到 `cudaStreamTailLaunch` stream 中的 kernel。

<a id="the-implicit-null-stream"></a>
<span id="id14"></span>

#### 5.6.4.8.1 The Implicit (NULL) Stream（隐式 NULL Stream）

在 host 程序中，未命名（NULL）stream 与其他 stream 之间具有额外的栅栏同步语义（详情请参见[阻塞与非阻塞 stream 以及默认 stream](../02-programming-gpus/asynchronous-execution.html#async-execution-blocking-non-blocking-default-stream)）。Device runtime 在一个线程块中的所有线程之间提供一个共享的隐式、未命名 stream；但由于所有命名 stream 都必须使用 `cudaStreamNonBlocking` 标志创建，因此启动到 NULL stream 的工作不会对其他 stream 中的待处理工作插入隐式依赖（包括其他线程块的 NULL stream）。

<a id="the-fire-and-forget-stream"></a>
<span id="fire-and-forget-stream"></span>

#### 5.6.4.8.2 The Fire-and-Forget Stream（Fire-and-Forget Stream）

Fire-and-Forget 命名 stream（`cudaStreamFireAndForget`）允许用户以更少的样板代码、且无需 stream 跟踪开销，启动 fire-and-forget 工作。它在功能上等同于每次启动时创建一个新 stream 并向该 stream 启动工作，但速度更快。

Fire-and-Forget 启动会立即安排启动，不依赖先前启动的 grid 是否完成。除 parent grid 结束时的隐式同步外，没有其他 grid 启动可以依赖 Fire-and-Forget 启动的完成。因此，在 parent grid 的 Fire-and-Forget 工作完成之前，tail launch 或 parent grid stream 中的下一个 grid 都不会启动。

```cpp
// 在此示例中，C2 的启动不会等待 C1 完成
__global__ void P( ... ) {
   C1<<< ... , cudaStreamFireAndForget >>>( ... );
   C2<<< ... , cudaStreamFireAndForget >>>( ... );
}
```

Fire-and-Forget stream 不能用于记录事件或等待事件。尝试这样做会导致 `cudaErrorInvalidValue`。使用 `CUDA_FORCE_CDP1_IF_SUPPORTED` 定义进行编译时，不支持 Fire-and-Forget stream。使用 Fire-and-Forget stream 要求以 64 位模式编译。

<a id="the-tail-launch-stream"></a>
<span id="tail-launch-stream"></span>

#### 5.6.4.8.3 The Tail Launch Stream（Tail Launch Stream）

Tail launch 命名 stream（`cudaStreamTailLaunch`）允许一个 grid 在自身完成后安排另一个 grid 启动。在大多数情况下，应当可以使用 tail launch 实现与 `cudaDeviceSynchronize()` 相同的功能。

每个 grid 都有自己的 tail launch stream。一个 grid 启动的所有非 tail launch 工作，都会在启动 tail stream 之前隐式同步。也就是说，parent grid 的 tail launch 不会启动，直到 parent grid 以及 parent grid 向普通 stream、每线程 stream 或 Fire-and-Forget stream 启动的所有工作都完成。如果两个 grid 启动到同一个 grid 的 tail launch stream，后启动的 grid 不会启动，直到前一个 grid 及其所有后代工作都完成。

```cpp
// 在此示例中，C2 只有在 C1 完成后才会启动。
__global__ void P( ... ) {
   C1<<< ... , cudaStreamTailLaunch >>>( ... );
   C2<<< ... , cudaStreamTailLaunch >>>( ... );
}
```

启动到 tail launch stream 的 grid，在 parent grid 的所有工作完成之前不会启动，包括 parent 在所有非 tail launch stream 中启动的其他 grid（及其后代），甚至包括在 tail launch 之后执行或启动的工作。

```cpp
// 在此示例中，C 只有在 X、F 和 P 全部完成后才会启动。
__global__ void P( ... ) {
   C<<< ... , cudaStreamTailLaunch >>>( ... );
   X<<< ... , cudaStreamPerThread >>>( ... );
   F<<< ... , cudaStreamFireAndForget >>>( ... )
}
```

在 parent grid 的 stream 中，下一 grid 不会在 parent grid 的 tail launch 工作完成之前启动。换句话说，tail launch stream 的行为就像被插入 parent grid 与其 parent grid stream 中下一个 grid 之间。

```cpp
// 在此示例中，P2 只有在 C 完成后才会启动。
__global__ void P1( ... ) {
   C<<< ... , cudaStreamTailLaunch >>>( ... );
}

__global__ void P2( ... ) {
}

int main ( ... ) {
   ...
   P1<<< ... >>>( ... );
   P2<<< ... >>>( ... );
   ...
}
```

每个 grid 只拥有一个 tail launch stream。要让多个 grid 通过 tail launch 并发启动，可以使用下面的示例方式。

```cpp
// 在此示例中，C1 和 C2 会在 P 完成后并发启动
__global__ void T( ... ) {
   C1<<< ... , cudaStreamFireAndForget >>>( ... );
   C2<<< ... , cudaStreamFireAndForget >>>( ... );
}

__global__ void P( ... ) {
   ...
   T<<< ... , cudaStreamTailLaunch >>>( ... );
}
```

Tail launch stream 不能用于记录事件或等待事件。尝试这样做会导致 `cudaErrorInvalidValue`。使用 `CUDA_FORCE_CDP1_IF_SUPPORTED` 定义进行编译时，不支持 tail launch stream。使用 tail launch stream 要求以 64 位模式编译。

<a id="ecc-errors"></a>
<span id="id15"></span>

### 5.6.4.9 ECC Errors（ECC 错误）

CUDA kernel 内的代码无法获知 ECC 错误。只有在整个启动树完成后，ECC 错误才会在 host 端报告。嵌套程序执行期间产生的任何 ECC 错误，都可能生成异常或继续执行，具体取决于错误和配置。
