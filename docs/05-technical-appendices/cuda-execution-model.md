---
title: 5.8 CUDA C++ Execution Model（CUDA C++ 执行模型）
description: CUDA C++ host thread、device thread、CUDA API 与执行进度模型
---

<a id="cuda-c-execution-model"></a>
<span id="cuda-cplusplus-execution-model"></span>

# 5.8 CUDA C++ Execution Model（CUDA C++ 执行模型）

CUDA C++ 旨在为所有 device 执行线程提供[并行前进进度保证（parallel forward progress）](https://eel.is/c++draft/intro.progress#9)，从而支持使用 CUDA C++ 并行化已有的 C++ 应用程序。

以下内容是对当前 [ISO 国际标准 ISO/IEC 14882——C++ 编程语言](https://eel.is/c++draft/)草案中 [`[intro.progress]`](https://eel.is/c++draft/intro.progress) 一节的修改与扩展。CUDA C++ 编程语言是 C++ 编程语言的扩展；修改的部分会明确标出，并以**粗体**显示差异，其余部分均为新增内容。

<details>
<summary>标准中的前进进度定义（Forward progress definitions）</summary>

- [`[intro.progress.7]`](https://eel.is/c++draft/intro.progress#7)：对于提供[并发前进进度保证](https://eel.is/c++draft/intro.progress#def:concurrent_forward_progress_guarantees)的执行线程，只要该线程尚未终止，实现就必须保证该线程最终会取得进展。

  > **注 5：**无论其他执行线程（如果有）是否已经取得或正在取得进展，这一要求都适用。“最终满足”这一要求意味着该事件会在未指定但有限的时间内发生。

- [`[intro.progress.9]`](https://eel.is/c++draft/intro.progress#9)：对于提供[并行前进进度保证](https://eel.is/c++draft/intro.progress#9)的执行线程，如果该线程尚未执行任何执行步骤，实现不必保证它最终会取得进展；一旦该线程执行过一个步骤，它就提供[并发前进进度保证](https://eel.is/c++draft/intro.progress#def:concurrent_forward_progress_guarantees)。

  > **注 6：**这没有规定何时启动该执行线程；启动时间通常由创建该执行线程的实体规定。例如，一个提供并发前进进度保证、并从任务集合中按任意顺序逐个执行任务的执行线程，满足这些任务的并行前进进度要求。

</details>

<a id="host-threads"></a>
<span id="cuda-cplusplus-execution-model-host-threads"></span>

## 5.8.1 Host threads（Host 线程）

由 host 实现创建、用于执行 [`main`](https://en.cppreference.com/w/cpp/language/main_function)、[`std::thread`](https://en.cppreference.com/w/cpp/thread/thread) 和 [`std::jthread`](https://en.cppreference.com/w/cpp/thread/jthread) 的执行线程所获得的前进进度，是 host 实现定义的行为 [`[intro.progress]`](https://eel.is/c++draft/intro.progress)。通用 host 实现应提供并发前进进度。

如果 host 实现提供[并发前进进度](https://eel.is/c++draft/intro.progress#7)（[`intro.progress.7`](https://eel.is/c++draft/intro.progress#7)），则 CUDA C++ 会为 device 线程提供[并行前进进度](https://eel.is/c++draft/intro.progress#9)（[`intro.progress.9`](https://eel.is/c++draft/intro.progress#9)）。

<a id="device-threads"></a>
<span id="cuda-cplusplus-execution-model-device-threads"></span>

## 5.8.2 Device threads（Device 线程）

一旦 device 线程取得进展：

- 如果它属于一个 [Cooperative Grid](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g504b94170f83285c71031be6d5d15f73)，则其 grid 中的所有 device 线程最终都应取得进展。

- 否则，其<a href="../02-programming-gpus/intro-to-cuda-cpp.html#thread-block-clusters">线程块集群</a>中的所有 device 线程最终都应取得进展。

  > **注：**其他线程块集群中的线程不保证最终取得进展。
  >
  > **注：**这意味着其线程块集群中的所有 device 线程最终都应取得进展。

按照如下内容修改 [`[intro.progress.1]`](https://eel.is/c++draft/intro.progress#1)（修改部分以**粗体**显示）：

实现可以假设任意 **host** 线程最终会执行以下操作之一：

> 1. 终止；
>
> 2. 调用函数 [`std::this_thread::yield`](https://en.cppreference.com/w/cpp/thread/yield)（[`[thread.thread.this]`](http://eel.is/c++draft/thread.thread.this)）；
>
> 3. 调用库 I/O 函数；
>
> 4. 通过 volatile glvalue 执行访问；
>
> 5. 执行同步操作或原子操作；或者
>
> 6. 继续执行一个平凡的无限循环（[`[stmt.iter.general]`](http://eel.is/c++draft/stmt.iter.general)）。

**实现可以假设任意 device 线程最终会执行以下操作之一：**

> 1. **终止；**
>
> 2. **调用库 I/O 函数；**
>
> 3. **通过 volatile glvalue 执行访问，但指定对象具有自动存储期的情况除外；或者**
>
> 4. **执行同步操作或原子读取操作，但指定对象具有自动存储期的情况除外。**
>
> **注：**device 线程相对于 host 线程的一些当前限制，是 NVIDIA 已知的、未来可能修复的实现缺陷。例如，device 线程最终只对具有自动存储期的对象执行 volatile 操作或原子操作时产生的未定义行为，就是其中之一。不过，device 线程相对于 host 线程的其他限制是有意作出的选择。这些选择能够启用一些性能优化；如果 device 线程严格遵循 C++ 标准，这些优化将无法实现。例如，为最终只执行原子写入或栅栏的程序提供前进进度，会在实际收益很小的情况下降低总体性能。

<details>
<summary>由于修改 [intro.progress.1]，host 与 device 线程的前进进度保证差异示例</summary>

以下示例使用上文 host 和 device 线程实现假设中的条款编号，分别记为“host.threads.&lt;id&gt;”和“device.threads.&lt;id&gt;”。代码中的行号和注释保留自官方示例。

```cuda
1// Example: Execution.Model.Device.0
2// Outcome: grid eventually terminates per device.threads.4 because the atomic object does not have automatic storage duration.
3__global__ void ex0(cuda::atomic_ref<int, cuda::thread_scope_device> atom) {
4    if (threadIdx.x == 0) {
5        while(atom.load(cuda::memory_order_relaxed) == 0);
6    } else if (threadIdx.x == 1) {
7        atom.store(1, cuda::memory_order_relaxed);
8    }
9}
```

```cuda
1// Example: Execution.Model.Device.1
2// Allowed outcome: No thread makes progress because device threads don't support host.threads.2.
3__global__ void ex1() {
4    while(true) cuda::std::this_thread::yield();
5}
```

```cuda
1// Example: Execution.Model.Device.2
2// Allowed outcome: No thread makes progress because device threads don't support host.threads.4
3// for objects with automatic storage duration (see exception in device.threads.3).
4__global__ void ex2() {
5    volatile bool True = true;
6    while(True);
7}
```

```cuda
1// Example: Execution.Model.Device.3
2// Allowed outcome: No thread makes progress because device threads don't support host.threads.5
3// for objects with automatic storage duration (see exception in device.threads.4).
4__global__ void ex3() {
5    cuda::atomic<bool, cuda::thread_scope_thread> True = true;
6    while(True.load());
7}
```

```cuda
1// Example: Execution.Model.Device.4
2// Allowed outcome: No thread makes progress because device threads don't support host.thread.6.
3__global void ex4() {
4    while(true) { /* empty */ }
5}
```

</details>

<a id="cuda-apis"></a>
<span id="cuda-cplusplus-execution-model-cuda-apis"></span>

## 5.8.3 CUDA APIs（CUDA API）

一次 CUDA API 调用最终必须返回，或者确保至少一个 device 线程取得进展。

CUDA 查询函数（例如 [`cudaStreamQuery`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435)、[`cudaEventQuery`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2bf738909b4a059023537eaa29d8a5b7) 等）不应在没有任何 device 线程取得进展的情况下持续返回 `cudaErrorNotReady`。

> **注：**device 线程不必与 API 调用“相关”；例如，在一个 stream 或进程上操作的 API，可以确保另一个 stream 或进程上的 device 线程取得进展。
>
> **注：**测试程序是否符合 CUDA API 前进进度的一种简单但并不充分的方法，是设置环境变量 `CUDA_DEVICE_MAX_CONNECTIONS=1` 和 `CUDA_LAUNCH_BLOCKING=1` 后运行程序，然后检查程序是否仍能终止。如果不能，程序存在错误。该方法并不充分，因为它无法捕获所有前进进度错误；但它能够捕获许多此类错误。

<details>
<summary>CUDA API 前进进度保证示例</summary>

```cuda
     1// Example: Execution.Model.API.1
     2// Outcome: if no other device threads (e.g., from other processes) are making progress,
     3// this program terminates and returns cudaSuccess.
     4// Rationale: CUDA guarantees that if the device is empty:
     5// - `cudaDeviceSynchronize` eventually ensures that at least one device-thread makes progress, which implies that eventually `hello_world` grid and one of its device-threads start.
     6// - All thread-block threads eventually start (due to "if a device thread makes progress, all other threads in its thread-block cluster eventually make progress").
     7// - Once all threads in thread-block arrive at `__syncthreads` barrier, all waiting threads are unblocked.
     8// - Therefore all device threads eventually exit the `hello_world`` grid.
     9// - And `cudaDeviceSynchronize`` eventually unblocks.
    10__global__ void hello_world() { __syncthreads(); }
    11int main() {
    12    hello_world<<<1,2>>>();
    13    return (int)cudaDeviceSynchronize();
    14}
```

```cuda
     1// Example: Execution.Model.API.2
     2// Allowed outcome: eventually, no thread makes progress.
     3// Rationale: the `cudaDeviceSynchronize` API below is only called if a device thread eventually makes progress and sets the flag.
     4// However, CUDA only guarantees that `producer` device thread eventually starts if the synchronization API is called.
     5// Therefore, the host thread may never be unblocked from the flag spin-loop.
     6cuda::atomic<int, cuda::thread_scope_system> flag = 0;
     7__global__ void producer() { flag.store(1); }
     8int main() {
     9    cudaHostRegister(&flag, sizeof(flag));
    10    producer<<<1,1>>>();
    11    while (flag.load() == 0);
    12    return cudaDeviceSynchronize();
    13}
```

```cuda
     1// Example: Execution.Model.API.3
     2// Allowed outcome: eventually, no thread makes progress.
     3// Rationale: same as Example.Model.API.2, with the addition that a single CUDA query API call does not guarantee
     4// the device thread eventually starts, only repeated CUDA query API calls do (see Execution.Model.API.4).
     5cuda::atomic<int, cuda::thread_scope_system> flag = 0;
     6__global__ void producer() { flag.store(1); }
     7int main() {
     8    cudaHostRegister(&flag, sizeof(flag));
     9    producer<<<1,1>>>();
    10    (void)cudaStreamQuery(0);
    11    while (flag.load() == 0);
    12    return cudaDeviceSynchronize();
    13}
```

```cuda
     1// Example: Execution.Model.API.4
     2// Outcome: terminates.
     3// Rationale: same as Example.Model.API.3, but this example repeatedly calls
     4// a CUDA query API in within the flag spin-loop, which guarantees that the device thread
     5// eventually makes progress.
     6cuda::atomic<int, cuda::thread_scope_system> flag = 0;
     7__global__ void producer() { flag.store(1); }
     8int main() {
     9    cudaHostRegister(&flag, sizeof(flag));
    10    producer<<<1,1>>>();
    11    while (flag.load() == 0) {
    12        (void)cudaStreamQuery(0);
    13    }
    14    return cudaDeviceSynchronize();
    15}
```

</details>

<a id="dependencies"></a>
<span id="cuda-cplusplus-execution-model-cuda-dependencies"></span>

### 5.8.3.1 Dependencies（依赖）

device 线程只有在其所有依赖完成后才能启动。

> **注：**阻止 device 线程开始取得进展的依赖可以通过<a href="../02-programming-gpus/asynchronous-execution.html#cuda-streams">CUDA Stream Commands</a>创建。例如，这些依赖可能要求完成<a href="../02-programming-gpus/asynchronous-execution.html#cuda-events">CUDA Events</a>或<a href="../02-programming-gpus/intro-to-cuda-cpp.html#kernels">CUDA Kernels</a>等操作。

<details>
<summary>由于依赖产生的 CUDA API 前进进度保证示例</summary>

```cuda
     1// Example: Execution.Model.Stream.0
     2// Allowed outcome: eventually, no thread makes progress.
     3// Rationale: while CUDA guarantees that one device thread makes progress, since there
     4// is no dependency between `first` and `second`, it does not guarantee which thread,
     5// and therefore it could always pick the device thread from `second`, which then never
     6// unblocks from the spin-loop.
     7// That is, `second` may starve `first`.
     8cuda::atomic<int, cuda::thread_scope_system> flag = 0;
     9__global__ void first() { flag.store(1, cuda::memory_order_relaxed); }
    10__global__ void second() { while(flag.load(cuda::memory_order_relaxed) == 0) {} }
    11int main() {
    12    cudaHostRegister(&flag, sizeof(flag));
    13    cudaStream_t s0, s1;
    14    cudaStreamCreate(&s0);
    15    cudaStreamCreate(&s1);
    16    first<<<1,1,0,s0>>>();
    17    second<<<1,1,0,s1>>>();
    18    return cudaDeviceSynchronize();
    19}
```

```cuda
     1// Example: Execution.Model.Stream.1
     2// Outcome: terminates.
     3// Rationale: same as Execution.Model.Stream.0, but this example has a stream dependency
     4// between first and second, which requires CUDA to run the grids in order.
     5cuda::atomic<int, cuda::thread_scope_system> flag = 0;
     6__global__ void first() { flag.store(1, cuda::memory_order_relaxed); }
     7__global__ void second() { while(flag.load(cuda::memory_order_relaxed) == 0) {} }
     8int main() {
     9    cudaHostRegister(&flag, sizeof(flag));
    10    cudaStream_t s0;
    11    cudaStreamCreate(&s0);
    12    first<<<1,1,0,s0>>>();
    13    second<<<1,1,0,s0>>>();
    14    return cudaDeviceSynchronize();
    15}
```

</details>
