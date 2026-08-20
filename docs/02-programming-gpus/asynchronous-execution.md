---
title: 2.5 异步执行
description: CUDA 异步并发、stream、event、回调、同步、默认 stream 与 CUDA Graph
---

# 2.5 异步执行（Asynchronous Execution）

> 本页按 NVIDIA CUDA Programming Guide Release 13.3 的官方网页逐段翻译。原文页面：[2.5. Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)；官方页面标注的更新时间为 2026-05-27。

## 2.5.1 什么是异步并发执行（What is Asynchronous Concurrent Execution）

CUDA 允许多个任务并发或重叠执行，具体包括：

- host 上的计算；
- device 上的计算；
- 从 host 到 device 的内存传输；
- 从 device 到 host 的内存传输；
- 同一 device 内部的内存传输；
- device 之间的内存传输。

这种并发通过异步接口表达：分发函数调用或 kernel launch 会立即返回。异步调用通常在所分发的操作完成前返回，甚至可能在异步操作开始前就返回。这样，应用就可以在原操作执行的同时执行其他任务。当应用需要最初分发操作的最终结果时，必须执行某种同步，以确保该操作已经完成。一个典型的并发执行模式是让 host/device 内存传输与计算重叠，从而减少或消除传输开销。

![图 20：使用 CUDA stream 的异步并发执行](/images/chapter-02/cuda_streams.png)

<div class="figure-caption"><strong>图 20：使用 CUDA stream 的异步并发执行。</strong> 原 PDF 第 65 页。不同 stream 中的计算与数据传输可以在资源允许时重叠；同一 stream 内的操作仍遵循其入队顺序。</div>

一般来说，异步接口通常提供三种与分发操作同步的方式：

- **阻塞方式**：应用调用一个会阻塞或等待的函数，直到操作完成；
- **非阻塞方式**或轮询方式：应用调用一个立即返回、并提供操作状态信息的函数；
- **回调方式**：操作完成后执行预先注册的函数。

虽然编程接口是异步的，但实际能否并发执行各种操作，取决于 CUDA 版本和所用硬件的 compute capability；这些细节会在本指南后面的[计算能力](../05-technical-appendices/compute-capabilities.html)中介绍。

在[同步 CPU 与 GPU](./intro-to-cuda-cpp.html#214-同步-cpu-与-gpu-synchronizing-cpu-and-gpu)一节中介绍了 CUDA runtime 函数 `cudaDeviceSynchronize()`。这是一个阻塞调用，会等待此前发出的所有工作完成。需要它的原因是 kernel launch 是异步的，会立即返回。CUDA 同时提供阻塞和非阻塞同步 API，也支持使用 host 侧回调函数。

CUDA 中异步执行的核心 API 组件是 **CUDA Streams** 和 **CUDA Events**。本节后续内容会解释如何使用这些元素表达 CUDA 中的异步执行。

另一个相关主题是 **CUDA Graphs**。它允许预先定义一组异步操作组成的图，然后以极小开销重复执行。本节的[使用 Stream Capture 初识 CUDA Graph](#2592-使用-stream-capture-初识-cuda-graph-introduction-to-cuda-graphs-with-stream-capture)会非常初步地介绍 CUDA Graph；更完整的讨论见[CUDA Graph](../04-cuda-features/cuda-graphs.html)。

## 2.5.2 CUDA Stream

在最基本的层面，CUDA stream 是一种让程序员表达操作序列的抽象。stream 类似一个工作队列：程序可以把内存复制或 kernel launch 等操作加入队列，操作会按顺序执行。给定 stream 队列最前端的操作执行后会出队，下一项操作来到队首并等待执行。stream 中操作的执行顺序是串行的，且与操作加入 stream 的顺序相同。

应用可以同时使用多个 stream。此时，runtime 会根据 GPU 资源状态，从有可用工作的 stream 中选择任务执行。stream 可以被分配优先级，优先级只是影响调度的提示，并不保证特定的执行顺序。

在 stream 上运行的 API 调用和 kernel launch 相对于 host 线程是异步的。应用可以通过等待某个 stream 变为空来与其同步，也可以在 device 级别同步。

CUDA 有一个 default stream；没有指定特定 stream 的操作和 kernel launch 会加入这个 default stream。未指定 stream 的代码示例，都是隐式使用 default stream。default stream 有一些专门语义，见[阻塞、非阻塞 stream 与 default stream](#256-阻塞和非阻塞-stream-以及-default-stream-blocking-and-non-blocking-streams-and-the-default-stream)。

### 2.5.2.1 创建和销毁 CUDA Stream（Creating and Destroying CUDA Streams）

可以使用 `cudaStreamCreate()` 函数创建 CUDA stream。该调用初始化一个 stream handle，后续函数调用可以使用它识别这个 stream。

```cpp
cudaStream_t stream;        // Stream handle
cudaStreamCreate(&stream);  // Create a new stream

// stream based operations ...

cudaStreamDestroy(stream);  // Destroy the stream
```

如果应用调用 `cudaStreamDestroy()` 时 device 仍在 `stream` 中执行工作，stream 会先完成其中的全部工作，再被销毁。

### 2.5.2.2 在 CUDA Stream 中启动 Kernel（Launching Kernels in CUDA Streams）

启动 kernel 的常用三尖括号语法同样可以用来把 kernel 启动到指定 stream 中。stream 作为 kernel launch 的额外参数指定。下面的例子把名为 `kernel` 的 kernel 启动到 handle 为 `stream` 的 stream 中；该 handle 类型为 `cudaStream_t`，并假设此前已经创建。

```cpp
kernel<<<grid, block, shared_mem_size, stream>>>(...);
```

kernel launch 是异步的，函数调用会立即返回。假定 kernel launch 成功，kernel 会在 `stream` 中执行；kernel 执行期间，应用可以在 CPU 上或 GPU 的其他 stream 中执行其他任务。

### 2.5.2.3 在 CUDA Stream 中启动内存传输（Launching Memory Transfers in CUDA Streams）

可以使用 `cudaMemcpyAsync()` 把内存传输加入 stream。这个函数与 `cudaMemcpy()` 类似，但多一个参数，用于指定内存传输所使用的 stream。下面的代码块把 `size` 个字节从 host 内存中的 `src` 复制到 device 内存中的 `dst`，并放入 `stream`：

```cpp
// Copy `size` bytes from `src` to `dst` in stream `stream`
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
```

和其他异步函数调用一样，这个调用会立即返回；而 `cudaMemcpy()` 会阻塞，直到内存传输完成。为了安全访问传输结果，应用必须通过某种同步方式确定操作已经完成。

`cudaMemcpy2D()` 等其他 CUDA 内存传输函数也有异步版本。

::: tip 注意
如果要异步执行涉及 CPU 内存的复制，host 缓冲区必须是 pinned、page-locked 内存。使用未 pinned、未 page-locked 的 host 内存时，`cudaMemcpyAsync()` 仍能正确运行，但会退化为同步行为，不能与其他工作重叠，从而抑制异步内存传输的性能收益。建议使用 `cudaMallocHost()` 分配用于向 GPU 发送数据或从 GPU 接收数据的缓冲区。
:::

### 2.5.2.4 Stream 同步（Stream Synchronization）

与 stream 同步最简单的方式是等待 stream 中没有任务。可以使用 `cudaStreamSynchronize()` 或 `cudaStreamQuery()` 两种函数。

`cudaStreamSynchronize()` 会阻塞，直到 stream 中的所有工作完成。

```cpp
// Wait for the stream to be empty of tasks
cudaStreamSynchronize(stream);

// At this point the stream is done
// and we can access the results of stream operations safely
```

如果不想阻塞，只想快速检查 stream 是否为空，可以使用 `cudaStreamQuery()`：

```cpp
// Have a peek at the stream
// returns cudaSuccess if the stream is empty
// returns cudaErrorNotReady if the stream is not empty
cudaError_t status = cudaStreamQuery(stream);

switch (status) {
    case cudaSuccess:
        // The stream is empty
        std::cout << "The stream is empty" << std::endl;
        break;
    case cudaErrorNotReady:
        // The stream is not empty
        std::cout << "The stream is not empty" << std::endl;
        break;
    default:
        // An error occurred - we should handle this
        break;
};
```

## 2.5.3 CUDA Event

CUDA event 是向 CUDA stream 中插入标记的机制。它们可以看作跟踪 stream 中任务进度的示踪粒子。假设向一个 stream 中启动两个 kernel。如果没有这样的跟踪 event，我们只能知道 stream 是否为空；如果有一个操作依赖第一个 kernel 的输出，在 stream 为空之前不能安全地启动该操作，而等到 stream 为空时两个 kernel 都已经完成。

使用 CUDA event 可以更早地开始依赖操作：把一个 event 加入 stream，并放在第一个 kernel 后、第二个 kernel 前，然后等待该 event 到达 stream 队首。这样，在知道第一个 kernel 已完成、但第二个 kernel 尚未开始时，就可以安全启动依赖操作。以这种方式使用 CUDA event，可以构建操作与 stream 之间的依赖图；这个图的类比会直接对应后面的[CUDA Graph](#2592-使用-stream-capture-初识-cuda-graph-introduction-to-cuda-graphs-with-stream-capture)讨论。

CUDA event 还保存时间信息，可用于测量 kernel launch 和内存传输的时间。

### 2.5.3.1 创建和销毁 CUDA Event（Creating and Destroying CUDA Events）

可以使用 `cudaEventCreate()` 和 `cudaEventDestroy()` 创建和销毁 CUDA event。

```cpp
cudaEvent_t event;

// Create the event
cudaEventCreate(&event);

// do some work involving the event

// Once the work is done and the event is no longer needed
// we can destroy the event
cudaEventDestroy(event);
```

不再需要 event 时，应用负责销毁它。

### 2.5.3.2 向 CUDA Stream 插入 Event（Inserting Events into CUDA Streams）

可以使用 `cudaEventRecord()` 向 stream 中插入 CUDA event。

```cpp
cudaEvent_t event;
cudaStream_t stream;

// Create the event
cudaEventCreate(&event);

// Insert the event into the stream
cudaEventRecord(event, stream);
```

### 2.5.3.3 测量 CUDA Stream 中的操作（Timing Operations in CUDA Streams）

CUDA event 可以用于测量包括 kernel 在内的各种 stream 操作。当 event 到达 stream 队首时，会记录时间戳。在 stream 中用两个 event 包围 kernel，就可以准确测量 kernel 执行时长：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaEvent_t start;
cudaEvent_t stop;

// create the events
cudaEventCreate(&start);
cudaEventCreate(&stop);

// record the start event
cudaEventRecord(start, stream);

// launch the kernel
kernel<<<grid, block, 0, stream>>>(...);

// record the stop event
cudaEventRecord(stop, stream);

// wait for the stream to complete
// both events will have been triggered
cudaStreamSynchronize(stream);

// get the timing
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

// clean up
cudaEventDestroy(start);
cudaEventDestroy(stop);
cudaStreamDestroy(stream);
```

### 2.5.3.4 检查 CUDA Event 状态（Checking the Status of CUDA Events）

和检查 stream 状态一样，可以用阻塞或非阻塞方式检查 event 状态。

`cudaEventSynchronize()` 会阻塞，直到 event 完成。下面的代码先向 stream 中启动 kernel，再加入 event，然后启动第二个 kernel。可以使用 `cudaEventSynchronize()` 等待第一个 kernel 后的 event 完成；原则上，此时可以立即启动依赖任务，可能早于 `kernel2` 完成。

```cpp
cudaEvent_t event;
cudaStream_t stream;

// create the stream
cudaStreamCreate(&stream);

// create the event
cudaEventCreate(&event);

// launch a kernel into the stream
kernel<<<grid, block, 0, stream>>>(...);

// Record the event
cudaEventRecord(event, stream);

// launch a kernel into the stream
kernel2<<<grid, block, 0, stream>>>(...);

// Wait for the event to complete
// Kernel 1 will be  guaranteed to have completed
// and we can launch the dependent task.
cudaEventSynchronize(event);
dependentCPUtask();

// Wait for the stream to be empty
// Kernel 2 is guaranteed to have completed
cudaStreamSynchronize(stream);

// destroy the event
cudaEventDestroy(event);

// destroy the stream
cudaStreamDestroy(stream);
```

还可以使用 `cudaEventQuery()` 以非阻塞方式检查 event 是否完成。下面的例子向一个 stream 中启动两个 kernel。第一个 kernel `kernel1` 生成希望复制到 host 的数据，但 CPU 侧同时还有工作要做。代码把 `kernel1`、event 和 `kernel2` 依次加入 `stream1`，然后进入 CPU 工作循环，间歇性检查 event 是否完成，以判断 `kernel1` 是否结束。如果结束，就向 `stream2` 加入 device-to-host 复制。这样可以让 CPU 工作、GPU kernel 执行和 device-to-host 复制重叠。

```cpp
cudaEvent_t event;
cudaStream_t stream1;
cudaStream_t stream2;

size_t size = LARGE_NUMBER;
float* d_data;
float* h_data;

// Create some data
cudaMalloc(&d_data, size);
cudaMallocHost(&h_data, size);

// create the streams
cudaStreamCreate(&stream1);   // Processing stream
cudaStreamCreate(&stream2);   // Copying stream
bool copyStarted = false;

// create the event
cudaEventCreate(&event);

// launch kernel1 into the stream
kernel1<<<grid, block, 0, stream1>>>(d_data, size);
// enqueue an event following kernel1
cudaEventRecord(event, stream1);

// launch kernel2 into the stream
kernel2<<<grid, block, 0, stream1>>>();

// while the kernels are running do some work on the CPU
// but check if kernel1 has completed because then we will start
// a device to host copy in stream2
while ( not allCPUWorkDone() || not copyStarted ) {
    doNextChunkOfCPUWork();

    // peek to see if kernel 1 has completed
    // if so enqueue a non-blocking copy into stream2
    if ( not copyStarted ) {
        if( cudaEventQuery(event) == cudaSuccess ) {
            cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2);
            copyStarted = true;
        }
    }
}

// wait for both streams to be done
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

// destroy the event
cudaEventDestroy(event);

// destroy the streams and free the data
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
cudaFree(d_data);
free(h_data);
```

## 2.5.4 Stream 回调函数（Callback Functions from Streams）

CUDA 提供了从 stream 内部在 host 上启动函数的机制。目前有两个用于此目的的函数：`cudaLaunchHostFunc()` 和 `cudaAddCallback()`。但 `cudaAddCallback()` 计划废弃，因此应用应使用 `cudaLaunchHostFunc()`。

### 使用 `cudaLaunchHostFunc()`

`cudaLaunchHostFunc()` 的签名如下：

```cpp
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*func)(void *), void *data);
```

参数含义：

- `stream`：把回调函数加入其中的 stream；
- `func`：要启动的回调函数；
- `data`：要传给回调函数的数据指针。

host 函数本身是一个简单的 C 函数，签名为：

```cpp
void hostFunction(void *data);
```

其中 `data` 参数指向用户定义的数据结构，函数可以解释该结构。使用这类回调函数时需要注意一些限制，尤其是 host 函数不能调用任何 CUDA API。

对于 unified memory，下面这些执行保证适用：

- 在函数执行期间，stream 被视为 idle。因此，例如，该函数始终可以使用它被加入的 stream 所附着的内存。
- 函数开始执行的效果等同于同步同一 stream 中紧邻函数之前记录的 event。因此，它会同步在该函数之前通过 event “连接”起来的 stream。
- 向任意 stream 添加 device 工作，不会让该 stream 变为 active，直到所有位于此前的 host 函数和 stream callback 都执行完。因此，例如，如果工作通过 event 排在函数调用之后，函数可以使用全局附着内存，即使另一个 stream 中已经添加了工作。
- 函数完成不会使 stream 变为 active，除非发生前面描述的情况。如果函数之后没有 device 工作，stream 会保持 idle；如果连续执行 host 函数或 stream callback 且中间没有 device 工作，stream 也会保持 idle。因此，例如可以在 stream 末尾的 host 函数中发信号，实现 stream 同步。

### 2.5.4.1 使用 `cudaStreamAddCallback()`

::: tip 注意
`cudaStreamAddCallback()` 计划废弃并移除；本节保留它是为了完整性，也因为它可能仍出现在已有代码中。应用应使用或迁移到 `cudaLaunchHostFunc()`。
:::

`cudaStreamAddCallback()` 的签名如下：

```cpp
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags);
```

参数含义：

- `stream`：把 callback 加入其中的 stream；
- `callback`：要启动的 callback 函数；
- `userData`：要传给 callback 函数的数据指针；
- `flags`：为了未来兼容，目前必须为 0。

`callback` 函数的签名与 `cudaLaunchHostFunc()` 的情况略有不同：

```cpp
void callbackFunction(cudaStream_t stream, cudaError_t status, void *userData);
```

此时函数会收到：

- `stream`：启动 callback 的 stream handle；
- `status`：触发 callback 的 stream 操作状态；
- `userData`：传给 callback 函数的数据指针。

特别是，`status` 参数包含 stream 的当前错误状态，该状态可能由此前的操作设置。与 `cudaLaunchHostFunc()` 类似，在 host function 完成前，stream 不会变为 active，也不会继续推进任务；callback 函数内部不能调用任何 CUDA 函数。

### 2.5.4.2 异步错误处理（Asynchronous Error Handling）

在 CUDA stream 中，错误可能来自 stream 中的任意操作，包括 kernel launch 和内存传输。直到 stream 被同步（例如等待 event 或调用 `cudaStreamSynchronize()`），这些错误才可能在运行时传播给用户。发现 stream 中可能发生错误有两种方式：

- 使用 `cudaGetLastError()`：返回并清除当前 context 中任何 stream 遇到的最后一个错误。如果两次调用之间没有其他错误，紧接着再次调用 `cudaGetLastError()` 会返回 `cudaSuccess`；
- 使用 `cudaPeekAtLastError()`：返回当前 context 中的最后一个错误，但不清除它。

这两个函数都以 `cudaError_t` 类型返回错误。可以使用 `cudaGetErrorName()` 和 `cudaGetErrorString()` 生成错误的可打印名称和描述。

下面是使用这两个函数的示例：

```cpp
// Some work occurs in streams.
cudaStreamSynchronize(stream);

// Look at the last error but do not clear it
cudaError_t err = cudaPeekAtLastError();
if (err != cudaSuccess) {
    printf("Error with name: %s\n", cudaGetErrorName(err));
    printf("Error description: %s\n", cudaGetErrorString(err));
}

// Look at the last error and clear it
cudaError_t err2 = cudaGetLastError();
if (err2 != cudaSuccess) {
    printf("Error with name: %s\n", cudaGetErrorName(err2));
    printf("Error description: %s\n", cudaGetErrorString(err2));
}

if (err2 != err) {
    printf("As expected, cudaPeekAtLastError() did not clear the error\n");
}

// Check again
cudaError_t err3 = cudaGetLastError();
if (err3 == cudaSuccess) {
    printf("As expected, cudaGetLastError() cleared the error\n");
}
```

::: tip 调试提示
当错误在同步操作处出现时，尤其是 stream 中包含许多操作时，往往很难准确定位错误在 stream 的哪里发生。一个有用的调试技巧是设置环境变量 `CUDA_LAUNCH_BLOCKING=1` 后运行应用。该变量会在每次 kernel launch 后同步，从而帮助定位是哪一个 kernel 或传输引发了错误。同步可能很昂贵；设置该变量后，应用可能明显变慢。
:::

## 2.5.5 CUDA Stream 的顺序（CUDA Stream Ordering）

在讨论完 stream、event 和 callback 函数的基本机制后，需要理解 stream 中异步操作的顺序语义。这些语义让应用程序员可以安全地思考 stream 中操作的顺序。在某些性能优化场景中，顺序语义可能被放宽，例如[程序化依赖 kernel launch](../04-cuda-features/programmatic-dependent-launch.html)可以通过特殊属性和 kernel launch 机制让两个 kernel 重叠；又如使用[异步批量内存复制函数](../03-advanced-cuda/advanced-apis-and-features.html)批量传输内存时，runtime 可以并发执行互不重叠的批量复制。

最重要的是，CUDA stream 是所谓的**有序流（in-order stream）**：stream 中操作的执行顺序与它们入队的顺序相同。一个操作不能跳过其他操作。runtime 会跟踪内存操作（例如复制），并始终在下一个操作之前完成这些操作，使依赖这些数据的 kernel 可以安全访问数据。

## 2.5.6 阻塞、非阻塞 Stream 与 Default Stream（Blocking and non-blocking streams and the default stream）

CUDA 中有两种 stream：阻塞 stream 和非阻塞 stream。名称可能有些误导，因为 blocking/non-blocking 语义只涉及 stream 如何与 default stream 同步。默认情况下，`cudaStreamCreate()` 创建的是 blocking stream。要创建 non-blocking stream，必须使用带 `cudaStreamNonBlocking` 标志的 `cudaStreamCreateWithFlags()`：

```cpp
cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
```

non-blocking stream 可以按通常方式用 `cudaStreamDestroy()` 销毁。

### 2.5.6.1 Legacy Default Stream

blocking stream 与 non-blocking stream 的关键区别在于它们如何与 **default stream** 同步。CUDA 提供了 legacy default stream，也称 NULL stream 或 stream ID 为 0 的 stream；没有在 kernel launch 或 blocking `cudaMemcpy()` 中指定 stream 时，会使用它。这个由所有 host 线程共享的 default stream 是 blocking stream。

当操作被加入 default stream 时，它会与所有其他 blocking stream 同步；换句话说，它会等待其他 blocking stream 完成后才能执行。

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

kernel1<<<grid, block, 0, stream1>>>(...);
kernel2<<<grid, block>>>(...);
kernel3<<<grid, block, 0, stream2>>>(...);

cudaDeviceSynchronize();
```

这意味着上面代码中的 `kernel2` 会等待 `kernel1` 完成，`kernel3` 会等待 `kernel2` 完成，即使原则上三个 kernel 可以并发执行。通过创建 non-blocking stream，可以避免这种同步行为。下面的代码创建两个 non-blocking stream；default stream 不再与它们同步，原则上三个 kernel 都可以并发执行。因此，不能假设这些 kernel 的执行顺序，应该进行显式同步（例如较重量级的 `cudaDeviceSynchronize()`），以确保 kernel 已完成。

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

kernel1<<<grid, block, 0, stream1>>>(...);
kernel2<<<grid, block>>>(...);
kernel3<<<grid, block, 0, stream2>>>(...);

cudaDeviceSynchronize();
```

### 2.5.6.2 每线程 Default Stream（Per-thread Default Stream）

从 CUDA 7 开始，CUDA 允许每个 host 线程拥有独立的 default stream，而不是共享 legacy default stream。要启用这一行为，可以使用 `nvcc` 编译选项 `--default-stream per-thread`，或者定义预处理宏 `CUDA_API_PER_THREAD_DEFAULT_STREAM`。

启用后，每个 host 线程都有自己的独立 default stream，它不会像 legacy default stream 那样与其他 stream 同步。在这种情况下，[legacy default stream 示例](#legacy-default-stream)会表现出与[non-blocking stream 示例](#legacy-default-stream)相同的同步行为。

## 2.5.7 显式同步（Explicit Synchronization）

有多种方式可以显式地让 stream 彼此同步：

- `cudaDeviceSynchronize()` 等待所有 host 线程的所有 stream 中此前发出的命令完成。
- `cudaStreamSynchronize()` 接受一个 stream 参数，等待给定 stream 中此前发出的命令完成。它可以让 host 与指定 stream 同步，同时允许 device 上的其他 stream 继续执行。
- `cudaStreamWaitEvent()` 接受 stream 和 event 参数（event 的说明见[CUDA Event](#253-cuda-event)），使调用之后加入给定 stream 的所有命令延迟执行，直到给定 event 完成。
- `cudaStreamQuery()` 让应用知道某个 stream 中此前发出的所有命令是否已经完成。

## 2.5.8 隐式同步（Implicit Synchronization）

如果两个不同 stream 之间插入了 NULL stream 上的 CUDA 操作，那么这两个 stream 中的操作不能并发运行；例外是两个 stream 都是 non-blocking stream（使用 `cudaStreamNonBlocking` 标志创建）。

应用应遵循以下原则，以提高 kernel 并发执行的潜力：

- 先发出所有相互独立的操作，再发出有依赖的操作；
- 尽可能延迟任何形式的同步。

## 2.5.9 其他与高级主题（Miscellaneous and Advanced topics）

### 2.5.9.1 Stream 优先级（Stream Prioritization）

前面提到，开发者可以为 CUDA stream 分配优先级。带优先级的 stream 必须使用 `cudaStreamCreateWithPriority()` 创建。该函数接受两个参数：stream handle 和优先级级别。一般规则是数值越小优先级越高。可以使用 `cudaDeviceGetStreamPriorityRange()` 查询指定 device 和 context 的优先级范围。stream 的默认优先级为 0。

```cpp
int minPriority, maxPriority;

// Query the priority range for the device
cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority);

// Create two streams with different priorities
// cudaStreamDefault indicates the stream should be created with default flags
// in other words they will be blocking streams with respect to the legacy default stream
// One could also use the option `cudaStreamNonBlocking` here to create a non-blocking streams
cudaStream_t stream1, stream2;
cudaStreamCreateWithPriority(&stream1, cudaStreamDefault, minPriority);  // Lowest priority
cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, maxPriority);  // Highest priority
```

需要注意，stream 优先级只是 runtime 的提示，通常主要应用于 kernel launch，对于内存传输可能不会生效。stream 优先级不会抢占已经执行的工作，也不保证任何特定执行顺序。

### 2.5.9.2 使用 Stream Capture 初识 CUDA Graph（Introduction to CUDA Graphs with Stream Capture）

CUDA stream 允许程序按顺序指定操作、kernel 或内存复制。使用多个 stream 和通过 `cudaStreamWaitEvent` 建立跨 stream 依赖，应用可以指定一个完整的有向无环图（DAG）。有些应用在执行过程中需要反复运行同一个操作序列或 DAG。

针对这种情况，CUDA 提供了称为 CUDA Graph 的功能。本节介绍 CUDA Graph，以及一种名为 *stream capture* 的创建机制。更详细的 CUDA Graph 讨论见 [CUDA Graph](../04-cuda-features/cuda-graphs.html)。捕获或创建图，可以减少 host 线程反复调用同一组 API 所产生的延迟和 CPU 开销。应用只需调用一次用于指定图操作的 API，之后就可以多次执行生成的图。

CUDA Graph 的工作方式如下：

1. 应用**捕获（captured）**图。第一次执行图时完成这一步；也可以使用 CUDA graph API 手动组合图。
2. 图被**实例化（instantiated）**。捕获图后执行一次这一步，用于设置执行图所需的各种 runtime 结构，使图组件可以尽可能快地启动。
3. 后续步骤中，预实例化的图按需要执行任意多次。因为执行图操作所需的 runtime 结构已经准备好，图执行的 CPU 开销会降到最低。

下面的代码来自 CUDA Developer Technical Blog（A. Gray，2019），展示使用 CUDA Graph 捕获、实例化和执行一个简单线性图的阶段：

```cpp
#define N 500000 // tuned such that kernel takes a few microseconds

// A very lightweight kernel
__global__ void shortKernel(float * out_d, float * in_d){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N) out_d[idx]=1.23*in_d[idx];
}

bool graphCreated=false;
cudaGraph_t graph;
cudaGraphExec_t instance;

// The graph will be executed NSTEP times
for(int istep=0; istep<NSTEP; istep++){
    if(!graphCreated){
        // Capture the graph
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        // Launch NKERNEL kernels
        for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
            shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
        }

        // End the capture
        cudaStreamEndCapture(stream, &graph);

        // Instantiate the graph
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        graphCreated=true;
    }

    // Launch the graph
    cudaGraphLaunch(instance, stream);

    // Synchronize the stream
    cudaStreamSynchronize(stream);
}
```

关于 CUDA Graph 的更多细节见 [CUDA Graph](../04-cuda-features/cuda-graphs.html)。

## 2.5.10 异步执行小结（Summary of Asynchronous Execution）

本节要点如下：

- 异步 API 允许表达任务的并发执行，从而可以表达不同操作的重叠；实际实现的并发程度取决于可用硬件资源和 compute capability。
- CUDA 中用于异步执行的关键抽象是 stream、event 和 callback 函数。
- 可以在 event、stream 和 device 层级进行同步。
- default stream 是 blocking stream，会与所有其他 blocking stream 同步，但不会与 non-blocking stream 同步。
- 可以通过 `--default-stream per-thread` 编译选项或 `CUDA_API_PER_THREAD_DEFAULT_STREAM` 预处理宏使用每线程 default stream，避免 default stream 的 legacy 行为。
- 可以创建不同优先级的 stream；优先级只是 runtime 的提示，可能不会对内存传输生效。
- CUDA 提供 CUDA Graph、批量内存传输和程序化依赖 kernel launch 等 API，以减少或重叠 kernel launch 与内存传输的开销。

## 本节导航

- [2.1 CUDA C++ 入门](./intro-to-cuda-cpp.html)
- [2.3 编写 SIMT Kernel](./writing-simt-kernels.html)
- [4.2 CUDA Graph](../04-cuda-features/cuda-graphs.html)
