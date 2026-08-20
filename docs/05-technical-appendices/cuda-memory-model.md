---
title: 5.7 CUDA C++ Memory Model（CUDA C++ 内存模型）
description: CUDA C++ 线程作用域、同步原语、原子性与数据竞争
---

<a id="cuda-c-memory-model"></a>
<span id="cuda-cplusplus-memory-model"></span>

# 5.7 CUDA C++ Memory Model（CUDA C++ 内存模型）

标准 C++ 给人的印象是：同步线程的成本是均匀的，而且很低。

CUDA C++ 则不同：线程之间相距越远，同步成本越高。在一个线程块内部的线程之间，同步成本较低；而在运行于多块 GPU 和 CPU 上的系统中，任意线程之间的同步成本很高。

为了处理并非均匀、也并非总是很低的线程同步成本，CUDA C++ 扩展了标准 C++ 内存模型以及 `cuda::` 命名空间中的并发设施，并引入了**线程作用域（thread scope）**；在默认情况下，它仍保留标准 C++ 的语法和语义。

<a id="thread-scopes"></a>
<span id="cuda-cplusplus-memory-model-thread-scopes"></span>

## 5.7.1 Thread Scopes（线程作用域）

**线程作用域（thread scope）**规定了哪些线程能够通过同步原语相互同步，例如 [`cuda::atomic`](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic.html) 或 [`cuda::barrier`](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/barrier.html)。

```cuda
namespace cuda {

enum thread_scope {
  thread_scope_system,
  thread_scope_device,
  thread_scope_block,
  thread_scope_thread
};

}  // namespace cuda
```

<a id="scope-relationships"></a>

### 5.7.1.1 Scope Relationships（作用域关系）

每个程序线程都通过一个或多个线程作用域关系与其他程序线程建立关系：

- 系统中的每个线程都通过 *system* 线程作用域与系统中的其他线程建立关系：`cuda::thread_scope_system`。

- 同一 CUDA 设备、同一<a href="../04-cuda-features/memory-synchronization-domains.html#memory-synchronization-domains">内存同步域</a>中的每个 GPU 线程，都通过 *device* 线程作用域与该设备中的其他 GPU 线程建立关系：`cuda::thread_scope_device`。

- 同一 CUDA 线程块中的每个 GPU 线程，都通过 *block* 线程作用域与该线程块中的其他 GPU 线程建立关系：`cuda::thread_scope_block`。

- 每个线程都通过 *thread* 线程作用域与自身建立关系：`cuda::thread_scope_thread`。

<a id="synchronization-primitives"></a>

## 5.7.2 Synchronization primitives（同步原语）

当使用 `cuda::thread_scope_system` 作用域实例化时，`std::` 和 `cuda::std::` 命名空间中的类型，与 `cuda::` 命名空间中相应类型的行为相同。

<a id="atomicity"></a>

## 5.7.3 Atomicity（原子性）

如果满足以下条件，某个原子操作就在它所指定的作用域内具有原子性：

- 它指定的作用域不是 `cuda::thread_scope_system`；**或者**

- 作用域是 `cuda::thread_scope_system`，并且满足以下任一条件：

  - 它影响<a href="../04-cuda-features/unified-memory.html#um-details-intro">系统分配的内存</a>中的对象，并且 [`pageableMemoryAccess`](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_146116bab1064b5d7d0642d78f6c27ce1) 为 `1` [0]；**或者**

  - 它影响<a href="../04-cuda-features/unified-memory.html#um-details-intro">托管内存</a>中的对象，并且 [`concurrentManagedAccess`](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b) 为 `1`；**或者**

  - 它影响<a href="../02-programming-gpus/unified-and-system-memory.html#memory-mapped-memory">映射内存</a>中的对象，并且 [`hostNativeAtomicSupported`](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f) 为 `1`；**或者**

  - 它是在<a href="../02-programming-gpus/unified-and-system-memory.html#memory-mapped-memory">映射内存</a>上影响自然对齐对象的加载或存储操作，对象大小为 `1`、`2`、`4`、`8` 或 `16` 字节 [1]；**或者**

  - 它影响 GPU 内存中的对象，只有 GPU 线程访问该对象，并且：

    - 在每个访问 `srcDev` 的设备与对象所在 GPU（`dstDev`）之间，调用 [`cudaDeviceGetP2PAttribute`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g2f597e2acceab33f60bd61c41fea0c1b) 得到的 `cudaDevP2PAttrNativeAtomicSupported` 值为 `1`；**或者**

    - 只有来自单个 GPU 的 GPU 线程并发访问该对象。

> **注意：**
>
> - [0] 如果 [`PageableMemoryAccessUsesHostPagetables`](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1e9f1ed6bffd5606eb81d438728a844ca) 为 `0`，则对内存映射文件或 `hugetlbfs` 分配执行的原子操作不具有原子性。
>
> - [1] 如果 [`hostNativeAtomicSupported`](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f) 为 `0`，则在系统作用域内对自然对齐的 16 字节对象执行原子加载或存储，且该对象位于<a href="../04-cuda-features/unified-memory.html#um-details-intro">系统分配的内存</a>或<a href="../02-programming-gpus/unified-and-system-memory.html#memory-mapped-memory">映射内存</a>中时，需要系统提供支持。NVIDIA 不知道有哪个系统缺少这种支持，并且 CUDA 没有提供可用于检测此类系统的 API 查询。

有关<a href="../04-cuda-features/unified-memory.html#um-details-intro">系统分配的内存</a>、<a href="../04-cuda-features/unified-memory.html#um-details-intro">托管内存</a>、<a href="../02-programming-gpus/unified-and-system-memory.html#memory-mapped-memory">映射内存</a>、CPU 内存和 GPU 内存的更多信息，请参阅本指南中的相关章节。

<a id="data-races"></a>

## 5.7.4 Data Races（数据竞争）

对 ISO/IEC IS 14882（C++ 标准）的 [`intro.races paragraph 21`](https://eel.is/c++draft/intro.races) 作如下修改：

> 如果程序的执行包含两个可能并发且相互冲突的操作，其中至少一个操作在**包含执行另一个操作的线程的作用域内不是原子的**，并且两者之间不存在 happens-before 关系（以下关于信号处理程序的特殊情形除外），则程序的执行包含数据竞争。任何此类数据竞争都会导致未定义行为。[…]

对 ISO/IEC IS 14882（C++ 标准）的 [`thread.barrier.class paragraph 4`](https://eel.is/c++draft/thread.barrier.class#4) 作如下修改：

> 4\. `barrier` 的成员函数（析构函数除外）的并发调用，不会引入数据竞争，**就像这些调用是原子操作一样**。[…]

对 ISO/IEC IS 14882（C++ 标准）的 [`thread.latch.class paragraph 2`](https://eel.is/c++draft/thread.latch.class#2) 作如下修改：

> 2\. `latch` 的成员函数（析构函数除外）的并发调用，不会引入数据竞争，**就像这些调用是原子操作一样**。[…]

对 ISO/IEC IS 14882（C++ 标准）的 [`thread.sema.cnt paragraph 3`](https://eel.is/c++draft/thread.sema.cnt#3) 作如下修改：

> 3\. `counting_semaphore` 的成员函数（析构函数除外）的并发调用，不会引入数据竞争，**就像这些调用是原子操作一样**。

对 ISO/IEC IS 14882（C++ 标准）的 [`thread.stoptoken.intro paragraph 5`](https://eel.is/c++draft/thread#stoptoken.intro-5) 作如下修改：

> 对 `request_stop`、`stop_requested` 和 `stop_possible` 函数的调用不会引入数据竞争，**就像这些调用是原子操作一样**。[…]

对 ISO/IEC IS 14882（C++ 标准）的 [`atomics.fences paragraph 2 through 4`](https://eel.is/c++draft/atomics.fences#2) 作如下修改：

> 如果存在原子操作 X 和 Y，且二者都作用于某个原子对象 M，同时满足 A 排序在 X 之前、X 修改 M、Y 排序在 B 之前，并且 Y 读取了 X 写入的值，或读取了假设 X 是释放操作时其所引领的释放序列中任意副作用写入的值，则释放栅栏 A 与获取栅栏 B 同步；此外，**操作 A、B、X 和 Y 中的每个操作所指定的作用域，都必须包含执行其他每个操作的线程**。
>
> 如果存在原子操作 X，且 A 排序在 X 之前、X 修改原子对象 M，并且 B 读取了 X 写入的值，或读取了假设 X 是释放操作时其所引领的释放序列中任意副作用写入的值，则释放栅栏 A 与对原子对象 M 执行获取操作的原子操作 B 同步；此外，**操作 A、B 和 X 中的每个操作所指定的作用域，都必须包含执行其他每个操作的线程**。
>
> 如果存在 M 上的原子操作 X，且 X 排序在 B 之前，并读取了 A 写入的值，或读取了 A 所引领的释放序列中任意副作用写入的值，则作为原子对象 M 上释放操作的原子操作 A 与获取栅栏 B 同步；此外，**操作 A、B 和 X 中的每个操作所指定的作用域，都必须包含执行其他每个操作的线程**。

<a id="example-message-passing"></a>
<span id="cuda-cplusplus-memory-model-message-passing"></span>

## 5.7.5 Example: Message Passing（示例：消息传递）

下面的示例中，线程块 `0` 中的线程通过标志 `f`，把存储在变量 `x` 中的消息传递给线程块 `1` 中的线程：

| 阶段 | 代码 |
|---|---|
| **初始状态** | `int x = 0, f = 0;` |
| **线程 0，线程块 0** | `x = 42;`<br>`cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);`<br>`flag.store(1, memory_order_release);` |
| **线程 0，线程块 1** | `cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);`<br>`while(flag.load(memory_order_acquire) != 1);`<br>`assert(x == 42);` |

在下面这个前一示例的变体中，两个线程在没有同步的情况下并发访问对象 `f`，这会导致**数据竞争**，并表现为**未定义行为**：

| 阶段 | 代码 |
|---|---|
| **初始状态** | `int x = 0, f = 0;` |
| **线程 0，线程块 0** | `x = 42;`<br>`cuda::atomic_ref<int, cuda::thread_scope_block> flag(f);`<br>`flag.store(1, memory_order_release); // UB: data race` |
| **线程 0，线程块 1** | `cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);`<br>`while(flag.load(memory_order_acquire) != 1); // UB: data race`<br>`assert(x == 42);` |

虽然对 `f` 执行的内存操作（存储和加载）是原子的，但存储操作的作用域是“**线程块作用域**”。由于存储操作由线程块 0 中的线程 0 执行，它只包含线程块 0 中的其他线程。然而，执行加载操作的线程位于线程块 1，也就是说，它不在由线程块 0 中执行的存储操作所包含的作用域内，因此存储和加载并不具有“原子性”，并引入了数据竞争。

更多示例请参阅 [PTX 内存一致性模型 litmus 测试](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#axioms)。
