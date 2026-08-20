---
title: 3.1 高级 CUDA API 与特性（Advanced CUDA APIs and Features）
description: 高级 CUDA API 与特性，包括 Kernel Launch、Cluster、Stream、PDL、批量内存传输和环境变量
---

# 3.1 高级 CUDA API 与特性（Advanced CUDA APIs and Features）

本节介绍更高级的 CUDA API 与特性。这些主题通常不要求修改 CUDA kernel，但仍然可以从 host 侧影响应用程序级行为，包括 GPU 工作的执行方式、GPU 性能以及 CPU 侧性能。

## 3.1.1 `cudaLaunchKernelEx`

在 CUDA 早期版本引入[三尖括号表示法](../02-programming-gpus/intro-to-cuda-cpp.html#2113-使用三尖括号启动-kernel-launching-kernels-with-triple-chevron)时，kernel 的[执行配置](../05-technical-appendices/cpp-language-extensions.html#执行配置-execution-configuration)只有四个可编程参数：

- thread block 维度；
- grid 维度；
- 动态 shared memory 大小（可选；未指定时为 0）；
- stream（未指定时使用 default stream）。

某些 CUDA 特性可以利用 kernel launch 时提供的额外属性和提示。`cudaLaunchKernelEx` 允许程序通过 `cudaLaunchConfig_t` 结构体设置上述执行配置参数。此外，`cudaLaunchConfig_t` 还允许程序传入零个或多个 `cudaLaunchAttribute`，用于控制或提示 kernel launch 的其他参数。例如，本章后文讨论的 [`cudaLaunchAttributePreferredSharedMemoryCarveout`](./advanced-kernel-programming.html#配置-l1shared-memory-平衡-configuring-l1shared-memory-balance) 就使用 `cudaLaunchKernelEx` 指定。后文讨论的 `cudaLaunchAttributeClusterDimension` 属性，则用于指定 kernel launch 所需的 cluster 大小。

支持的属性及其含义的完整列表，请参阅 [CUDA Runtime API Reference Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gfc5ed48085f05863b1aeebb14934b056)。

## 3.1.2 启动 Cluster（Launching Clusters）

[Thread block cluster](../01-introduction/programming-model.html#thread-block-clusters) 是前文介绍的一种可选 thread block 组织层级，在计算能力 9.0 及更高版本上可用。它允许应用保证：一个 cluster 中的 thread block 会同时在同一个 GPC 上执行。因此，超过单个 SM 容量的更大线程组也能够彼此交换数据并相互同步。

[2.1.10.1 节](../02-programming-gpus/intro-to-cuda-cpp.html#21101-使用三尖括号启动-cluster-launching-with-clusters-in-triple-chevron-notation)展示了如何使用三尖括号表示法指定并启动使用 cluster 的 kernel。在该节中，使用 `__cluster_dims__` 注解指定必须用于启动 kernel 的 cluster 维度。使用三尖括号表示法时，cluster 大小是隐式确定的。

### 3.1.2.1 使用 `cudaLaunchKernelEx` 启动 Cluster（Launching with Clusters using `cudaLaunchKernelEx`）

不同于[使用三尖括号表示法启动 Cluster](../02-programming-gpus/intro-to-cuda-cpp.html#21101-使用三尖括号启动-cluster-launching-with-clusters-in-triple-chevron-notation)，thread block cluster 的大小可以针对每次 launch 单独配置。下面的代码示例展示了如何使用 `cudaLaunchKernelEx` 启动一个 cluster kernel。

```cpp
// Kernel 定义
// 未在编译期为 kernel 附加属性
__global__ void cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // 使用运行时 cluster 大小调用 kernel
    {
        cudaLaunchConfig_t config = {0};
        // cluster launch 不会影响 grid 维度，grid 仍然使用 block 数量枚举。
        // grid 维度应当是 cluster 大小的整数倍。
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // X 维度上的 cluster 大小
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, cluster_kernel, input, output);
    }
}
```

与 thread block cluster 相关的 `cudaLaunchAttribute` 类型有两个：`cudaLaunchAttributeClusterDimension` 和 `cudaLaunchAttributePreferredClusterDimension`。

属性 ID `cudaLaunchAttributeClusterDimension` 指定执行 cluster 时必须使用的维度。该属性的值 `clusterDim` 是一个三维值。grid 的对应维度（x、y 和 z）必须分别能够被指定 cluster 维度的对应维度整除。设置该属性的效果类似于在编译期为 kernel 定义设置 `__cluster_dims__` 属性（见[使用三尖括号表示法启动 Cluster](../02-programming-gpus/intro-to-cuda-cpp.html#21101-使用三尖括号启动-cluster-launching-with-clusters-in-triple-chevron-notation)），但同一个 kernel 的不同 launch 可以在运行时使用不同设置。

在计算能力 10.0 及更高版本的 GPU 上，另一个属性 ID `cudaLaunchAttributePreferredClusterDimension` 允许应用额外指定 cluster 的首选维度。首选维度必须是 kernel 上 `__cluster_dims__` 属性，或通过 `cudaLaunchKernelEx` 传入的 `cudaLaunchAttributeClusterDimension` 属性所指定的最小 cluster 维度的整数倍。也就是说，指定首选 cluster 维度时，必须同时指定最小 cluster 维度。grid 的对应维度（x、y 和 z）必须分别能够被指定首选 cluster 维度的对应维度整除。

所有 thread block 都会以至少达到最小 cluster 维度的 cluster 执行。在可能的情况下会使用首选维度的 cluster，但不能保证所有 cluster 都会以首选维度执行。所有 thread block 都会以最小或首选 cluster 维度之一执行。使用首选 cluster 维度的 kernel 必须能够在最小和首选两种 cluster 维度下都正确工作。

### 3.1.2.2 将 Block 作为 Cluster（Blocks as Clusters）

当 kernel 使用 `__cluster_dims__` 注解定义时，grid 中的 cluster 数量是隐式的，可以通过 grid 大小除以指定的 cluster 大小计算得到。

```cpp
__cluster_dims__((2, 2, 2)) __global__ void foo();

// 8x8x8 个 cluster，每个 cluster 包含 2x2x2 个 thread block。
foo<<<dim3(16, 16, 16), dim3(1024, 1, 1)>>>();
```

在上面的示例中，kernel 以 16x16x16 个 thread block 组成的 grid 启动，因此使用的是 8x8x8 个 cluster 组成的 grid。

kernel 也可以使用 `__block_size__` 注解。该注解在定义 kernel 时同时指定所需的 block 大小和 cluster 大小。使用此注解后，三尖括号 launch 中的 grid 维度不再以 thread block 为单位，而是以 cluster 为单位，如下所示。

```cpp
// 每个 block 的线程数以及每个 cluster 的 block 数量，
// 都作为 kernel 的属性处理；这里仅展示其实现细节。
__block_size__((1024, 1, 1), (2, 2, 2)) __global__ void foo();

// 8x8x8 个 cluster。
foo<<<dim3(8, 8, 8)>>>();
```

`__block_size__` 需要两个字段，每个字段都是包含 3 个元素的 tuple。第一个 tuple 表示 block 维度，第二个 tuple 表示 cluster 大小。如果未传入第二个 tuple，则默认使用 `(1,1,1)`。当 kernel launch 还要指定动态 shared memory 大小和/或 stream 时，`<<<>>>` 中的第二个参数必须是占位值 `1`；任何其他值都会导致未定义行为。

注意，同时指定 `__block_size__` 的第二个 tuple 和 `__cluster_dims__` 是非法的。将 `__block_size__` 与空的 `__cluster_dims__` 一起使用同样是非法的。当指定 `__block_size__` 的第二个 tuple 时，就表示启用了“Blocks as Clusters”；编译器会将 `<<<>>>` 内的第一个参数识别为 cluster 数量，而不是 thread block 数量。

## 3.1.3 关于 Stream 和 Event 的更多内容（More on Streams and Events）

[CUDA Stream](../02-programming-gpus/asynchronous-execution.html#251-cuda-stream) 介绍了 CUDA stream 的基础知识。默认情况下，提交到给定 CUDA stream 的操作会串行执行：前一个操作完成后，后一个操作才能开始执行。唯一的例外是最近加入的[程序化依赖启动与同步](../04-cuda-features/programmatic-dependent-launch.html#程序化依赖启动与同步-programmatic-dependent-launch-and-synchronization)特性。使用多个 CUDA stream 可以实现并发执行；另一种方式是使用 [CUDA Graph](../04-cuda-features/cuda-graphs.html#cuda-graph)。两种方法也可以组合使用。

来自不同 CUDA stream 的工作在特定条件下可以并发执行，例如不存在 event 依赖、不存在隐式同步，并且有足够的资源等。

如果在来自不同 CUDA stream 的独立操作之间提交了任何 NULL stream 上的 CUDA 操作，那么这些操作不能并发执行，除非这些 stream 是 non-blocking CUDA stream。此类 stream 是通过 runtime API `cudaStreamCreateWithFlags()` 并传入 `cudaStreamNonBlocking` 标志创建的。为了提高 GPU 工作并发执行的可能性，建议创建 non-blocking CUDA stream。

此外，建议用户选择能够满足问题需求的、作用范围最小的同步选项。例如，如果要求 CPU 阻塞等待某个特定 CUDA stream 上的所有工作完成，那么对该 stream 使用 `cudaStreamSynchronize()` 会优于使用 `cudaDeviceSynchronize()`，因为后者会不必要地等待设备上所有 CUDA stream 的 GPU 工作完成。如果要求 CPU 等待但不阻塞，那么可以在轮询循环中调用 `cudaStreamQuery()` 并检查其返回值。

CUDA event（见 [CUDA Event](../02-programming-gpus/asynchronous-execution.html#252-cuda-event)）也可以实现类似的同步效果。例如，可以在该 stream 上记录一个 event，然后调用 `cudaEventSynchronize()`，以阻塞方式等待该 event 捕获的工作完成。这同样比使用 `cudaDeviceSynchronize()` 更聚焦。调用 `cudaEventQuery()` 并检查返回值（例如在轮询循环中）则是非阻塞替代方案。

如果同步操作处于应用程序的关键路径上，那么显式同步方法的选择尤其重要。[表 4](#表-4与-host-的显式同步选项摘要)从较高层次总结了各种与 host 同步的选项。

|  | 等待特定 stream | 等待特定 event | 等待设备上的全部工作 |
| --- | --- | --- | --- |
| 非阻塞（需要轮询循环） | `cudaStreamQuery()` | `cudaEventQuery()` | N/A |
| 阻塞 | `cudaStreamSynchronize()` | `cudaEventSynchronize()` | `cudaDeviceSynchronize()` |

**表 4：与 host 之间的显式同步选项摘要**

为了在 CUDA stream 之间进行同步，也就是表达依赖关系，建议使用不计时的 CUDA event，如 [CUDA Event](../02-programming-gpus/asynchronous-execution.html#252-cuda-event) 所述。可以调用 `cudaStreamWaitEvent()`，强制某个 stream 上之后提交的操作等待一个此前记录的 event 完成（例如该 event 记录在另一个 stream 上）。注意，对于任何等待或查询 event 的 CUDA API，用户都有责任确保已经调用过 `cudaEventRecord` API；未记录的 event 总是返回成功。

CUDA event 默认携带计时信息，因为它们可以用于调用 `cudaEventElapsedTime()` API。但是，如果 CUDA event 仅用于表达跨 stream 的依赖关系，就不需要计时信息。对于这种情况，建议创建禁用计时信息的 event，以改善性能。可以通过 `cudaEventCreateWithFlags()` API 配合 `cudaEventDisableTiming` 标志实现这一点。

### 3.1.3.1 Stream 优先级（Stream Priorities）

可以在创建 stream 时使用 `cudaStreamCreateWithPriority()` 指定 stream 的相对优先级。允许的优先级范围按“最高优先级、最低优先级”的顺序排列，可以通过 `cudaDeviceGetStreamPriorityRange()` 函数获得。在运行时，GPU 调度器使用 stream 优先级决定任务执行顺序，但这些优先级是提示而不是保证。选择待启动的工作时，高优先级 stream 中的待处理任务优先于低优先级 stream 中的任务。高优先级任务不会抢占已经运行的低优先级任务。GPU 不会在任务执行过程中重新评估工作队列，提高某个 stream 的优先级也不会中断正在执行的工作。Stream 优先级会影响任务执行，但不会强制严格排序；因此，用户可以利用它影响执行顺序，而不应依赖严格的排序保证。

下面的代码示例获取当前设备允许的优先级范围，并使用可用的最高和最低优先级创建两个 non-blocking CUDA stream。

```cpp
// 获取此设备的 stream 优先级范围
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

// 使用可用的最高和最低优先级创建 stream
cudaStream_t st_high, st_low;
cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, greatestPriority));
cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, leastPriority);
```

### 3.1.3.2 显式同步（Explicit Synchronization）

如前所述，stream 可以通过多种方式与其他 stream 同步。下面列出不同粒度下的常用方法：

- `cudaDeviceSynchronize()` 等待所有 host 线程的所有 stream 中，位于此前的全部命令完成。
- `cudaStreamSynchronize()` 接受一个 stream 作为参数，等待给定 stream 中位于此前的全部命令完成。它可以将 host 与特定 stream 同步，同时允许设备上的其他 stream 继续执行。
- `cudaStreamWaitEvent()` 接受一个 stream 和一个 event 作为参数（event 的说明见 [CUDA Event](../02-programming-gpus/asynchronous-execution.html#252-cuda-event)），使得在调用 `cudaStreamWaitEvent()` 后加入给定 stream 的所有命令，都要等到给定 event 完成后才执行。
- `cudaStreamQuery()` 为应用提供一种方式，用于判断某个 stream 中位于此前的所有命令是否已经完成。

### 3.1.3.3 隐式同步（Implicit Synchronization）

如果 host 线程在来自不同 stream 的两个命令之间发出以下任一操作，那么这两个命令不能并发执行：

- 分配 page-locked host memory；
- 分配 device memory；
- 设置 device memory；
- 在同一个 device memory 上的两个地址之间进行 memory copy；
- 向 NULL stream 提交任何 CUDA 命令；
- 切换 L1/shared memory 配置。

需要执行依赖检查的操作，还包括与被检查 launch 位于同一个 stream 中的任何其他命令，以及对该 stream 调用 `cudaStreamQuery()`。因此，为了提高 kernel 并发执行的可能性，应用应遵循以下准则：

- 所有独立操作都应在依赖操作之前提交；
- 尽可能延迟任何类型的同步。

## 3.1.4 程序化依赖 Kernel Launch（Programmatic Dependent Kernel Launch）

如前文所述，CUDA stream 的语义规定 kernel 按顺序执行。这样，如果两个 kernel 连续执行且第二个 kernel 依赖第一个 kernel 的结果，程序员可以确信：第二个 kernel 开始执行时，依赖数据已经可用。然而，可能出现这样的情况：第一个 kernel 已经把后续 kernel 所依赖的数据写入 global memory，但自身仍有更多工作要做；同样，依赖它的第二个 kernel 在真正需要第一个 kernel 的数据之前，也可能有一些独立工作。在这种情况下，如果硬件资源充足，就可以让两个 kernel 的执行部分重叠。第二个 kernel 的 launch 开销也可能与这段重叠执行重合。

除硬件资源是否充足外，可实现的重叠程度还取决于 kernel 的具体结构，例如：

- 第一个 kernel 在执行到哪个时刻完成第二个 kernel 所依赖的工作？
- 第二个 kernel 在执行到哪个时刻开始处理来自第一个 kernel 的数据？

由于这些问题高度依赖具体 kernel，完全自动化很困难。因此，CUDA 提供了一种机制，允许应用开发者指定两个 kernel 之间的同步点。这种技术称为“程序化依赖 Kernel Launch”。其情形如下图所示。

![图：程序化依赖 Kernel Launch（Programmatic Dependent Kernel Launch）](/images/chapter-03/pdl.png)

图示说明：第一个 primary kernel 在完成 secondary kernel 所需的工作后发出完成信号；secondary kernel 可以先执行独立工作，随后在依赖同步点等待 primary kernel 刷新结果，从而让两个 kernel 的可重叠部分并行推进。

PDL（Programmatic Dependent Launch）包含三个主要组成部分：

1. 第一个 kernel（即所谓的 *primary kernel*）需要调用特殊函数，表示它已经完成后续依赖 kernel（也称为 *secondary kernel*）所需的全部工作。该函数是 `cudaTriggerProgrammaticLaunchCompletion()`。
2. 依赖它的 secondary kernel 需要表示：自己已经执行到独立于 primary kernel 的部分，并正在等待 primary kernel 完成所依赖的工作。这通过 `cudaGridDependencySynchronize()` 函数实现。
3. 第二个 kernel 必须使用特殊属性 `cudaLaunchAttributeProgrammaticStreamSerialization` 启动，并将其 `programmaticStreamSerializationAllowed` 字段设置为 `1`。

下面的代码片段展示了实现方式。

**代码清单 3：使用两个 Kernel 的程序化依赖 Kernel Launch 示例**

```c
__global__ void primary_kernel() {
    // 在启动 secondary kernel 前应完成的初始工作

    // 触发 secondary kernel
    cudaTriggerProgrammaticLaunchCompletion();

    // 可以与 secondary kernel 同时进行的工作
}

__global__ void secondary_kernel()
{
    // 初始化、独立工作等

    // 等待 secondary kernel 所依赖的所有 primary kernel 完成，
    // 并将结果刷新到 global memory
    cudaGridDependencySynchronize();

    // 依赖工作
}

// 使用特殊属性启动 secondary kernel

// 设置属性
cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute[0].val.programmaticStreamSerializationAllowed = 1;

// 在 kernel launch 配置中设置属性
cudaLaunchConfig_t config = {0};

// 基本 launch 配置
config.gridDim = grid_dim;
config.blockDim = block_dim;
config.dynamicSmemBytes= 0;
config.stream = stream;

// 添加 PDL 专用属性
config.attrs = attribute;
config.numAttrs = 1;

// 启动 primary kernel
primary_kernel<<<grid_dim, block_dim, 0, stream>>>();

// 使用包含该属性的配置启动 secondary（依赖）kernel
cudaLaunchKernelEx(&config, secondary_kernel);
```

## 3.1.5 批量内存传输（Batched Memory Transfers）

CUDA 开发中的一个常见模式是使用 batching 技术。这里的 batching 是指：将多个（通常较小的）任务组合成一个（通常更大的）操作。批次中的组件不一定完全相同，尽管它们通常相同。cuBLAS 提供的批量矩阵乘法操作就是一个例子。

与 CUDA Graph 和 PDL 类似，batching 的目的通常是减少分别调度批次中各个任务所产生的开销。对于内存传输，启动一次传输会产生一定的 CPU 和 driver 开销。此外，当前形式的普通 `cudaMemcpyAsync()` 函数不一定会向 driver 提供足够的信息来优化传输，例如源和目标的位置提示。在 Tegra 平台上，可以选择使用 SM 或 Copy Engine（CE）执行传输；当前具体选择由 driver 中的启发式算法决定。这一点很重要，因为使用 SM 可能带来更快的传输，但会占用一部分可用计算能力。另一方面，使用 CE 可能使传输更慢，却能释放 SM 执行其他工作，从而获得更高的整体应用性能。

这些考虑促成了 `cudaMemcpyBatchAsync()` 函数及其对应的 `cudaMemcpyBatch3DAsync()` 函数的设计。这些函数允许优化批量内存传输。除了源指针和目标指针列表外，该 API 还使用 memory copy attribute 指定顺序预期，提供源和目标位置的提示，并提示是否希望传输与计算重叠（目前只有 Tegra 平台上的 CE 支持这一点）。

先考虑最简单的情况：从 pinned host memory 到 pinned device memory 的简单批量数据传输。

**代码清单 4：从 Pinned Host Memory 到 Pinned Device Memory 的同构批量内存传输示例**

```cpp
std::vector<void *> srcs(batch_size);
std::vector<void *> dsts(batch_size);
std::vector<size_t> sizes(batch_size);

// 分配源和目标缓冲区
// 使用 stream 编号初始化
for (size_t i = 0; i < batch_size; i++) {
    cudaMallocHost(&srcs[i], sizes[i]);
    cudaMalloc(&dsts[i], sizes[i]);
    cudaMemsetAsync(srcs[i], sizes[i], stream);
}

// 为这一批 copy 设置属性
cudaMemcpyAttributes attrs = {};
attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;

// 批次中的所有 copy 使用相同的 copy 属性。
size_t attrsIdxs = 0;  // 属性索引

// 启动批量内存传输
cudaMemcpyBatchAsync(&dsts[0], &srcs[0], &sizes[0], batch_size,
    &attrs, &attrsIdxs, 1 /*numAttrs*/, nullptr /*failIdx*/, stream);
```

`cudaMemcpyBatchAsync()` 的前几个参数比较直观：它们由包含源指针、目标指针和传输大小的数组组成。每个数组都必须有 `batch_size` 个元素。新的信息来自属性。该函数需要一个属性数组指针，以及一个与之对应的属性索引数组。原则上也可以传入一个 `size_t` 数组，在其中记录失败传输的索引；但这里传入 `nullptr` 是安全的，此时不会记录失败索引。

在这个例子中，传输是同构的，因此只使用一个属性，并将其应用于所有传输。这由 `attrIndex` 参数控制。原则上，`attrIndex` 可以是一个数组。数组的第 *i* 个元素包含属性数组第 *i* 个元素所适用的第一个传输的索引。在本例中，`attrIndex` 被当作只有一个元素的数组，其值为 `0`，这意味着 `attribute[0]` 会应用于索引从 0 开始的所有传输，也就是全部传输。

最后，注意本例将 `srcAccessOrder` 属性设置为 `cudaMemcpySrcAccessOrderStream`。这意味着源数据会按照正常的 stream 顺序访问。换句话说，在处理这些源指针和目标指针所涉及数据的此前 kernel 完成之前，memcpy 会阻塞。

下面考虑一个更复杂的异构批量传输。

**代码清单 5：使用部分临时 Host Memory 向 Pinned Device Memory 进行异构批量传输的示例**

```cpp
std::vector<void *> srcs(batch_size);
std::vector<void *> dsts(batch_size);
std::vector<size_t> sizes(batch_size);

// 分配源和目标缓冲区
for (size_t i = 0; i < batch_size - 10; i++) {
    cudaMallocHost(&srcs[i], sizes[i]);
    cudaMalloc(&dsts[i], sizes[i]);
}

int buffer[10];

for (size_t i = batch_size - 10; i < batch_size; i++) {
    srcs[i] = &buffer[10 - (batch_size - i];
    cudaMalloc(&dsts[i], sizes[i]);
}

// 为这一批 copy 设置属性
cudaMemcpyAttributes attrs[2] = {};
attrs[0].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
attrs[1].srcAccessOrder = cudaMemcpySrcAccessOrderDuringApiCall;

size_t attrsIdxs[2];
attrsIdxs[0] = 0;
attrsIdxs[1] = batch_size - 10;

// 启动批量内存传输
cudaMemcpyBatchAsync(&dsts[0], &srcs[0], &sizes[0], batch_size,
    &attrs, &attrsIdxs, 2 /*numAttrs*/, nullptr /*failIdx*/, stream);
```

这里有两类传输：`batch_size-10` 个从 pinned host memory 到 pinned device memory 的传输，以及 10 个从 host 数组到 pinned device memory 的传输。此外，`buffer` 数组不仅位于 host 上，而且只在当前作用域内存在；它的地址称为 *ephemeral pointer*。由于 API 是异步的，该指针在 API 调用完成后可能不再有效。要使用这类 ephemeral pointer 执行 copy，属性中的 `srcAccessOrder` 必须设置为 `cudaMemcpySrcAccessOrderDuringApiCall`。

现在有两个属性：第一个属性适用于索引从 0 开始且小于 `batch_size-10` 的所有传输；第二个属性适用于索引从 `batch_size-10` 开始且小于 `batch_size` 的所有传输。

如果不是从栈上分配 `buffer` 数组，而是使用 `malloc` 从堆上分配，那么该数据就不再是 ephemeral 的，而会一直有效到显式释放指针为止。在这种情况下，如何为 copy 选择最佳 staging 方式，取决于系统是否具有硬件管理的 memory，或者是否能够通过地址转换以一致方式访问 host memory。如果系统具备这些能力，最好使用 stream ordering；如果不具备，则立即 staging 传输更合理。在这种情况下，应将属性的 `srcAccessOrder` 使用值 `cudaMemcpyAccessOrderAny`。

`cudaMemcpyBatchAsync` 还允许程序员提供源和目标位置的提示。方法是设置 `cudaMemcpyAttributes` 结构体的 `srcLocation` 和 `dstLocation` 字段。这两个字段的类型都是 `cudaMemLocation`，该结构体包含位置类型和位置 ID。`cudaMemLocation` 也是 runtime 在使用 `cudaMemPrefetchAsync()` 时可以用来提供预取提示的结构体。下面的代码示例展示如何为从 device 到 host 特定 NUMA 节点的传输设置位置提示：

**代码清单 6：设置源和目标位置提示的示例**

```cpp
// 分配源和目标缓冲区
std::vector<void *> srcs(batch_size);
std::vector<void *> dsts(batch_size);
std::vector<size_t> sizes(batch_size);

// 用于提供位置提示的 cudaMemLocation 结构体
// Device device_id
cudaMemLocation srcLoc = {cudaMemLocationTypeDevice, dev_id};

// Host 上的 NUMA Node numa_id
cudaMemLocation dstLoc = {cudaMemLocationTypeHostNuma, numa_id};

// 分配源和目标缓冲区
for (size_t i = 0; i < batch_size; i++) {
    cudaMallocManaged(&srcs[i], sizes[i]);
    cudaMallocManaged(&dsts[i], sizes[i]);

    cudaMemPrefetchAsync(srcs[i], sizes[i], srcLoc, 0, stream);
    cudaMemPrefetchAsync(dsts[i], sizes[i], dstLoc, 0, stream);
    cudaMemsetAsync(srcs[i], sizes[i], stream);
}

// 为这一批 copy 设置属性
cudaMemcpyAttributes attrs = {};

// 这些是 managed memory 指针，因此适合使用 Stream Order
attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;

// 现在可以在这里指定位置提示。
attrs.srcLocHint = srcLoc;
attrs.dstlocHint = dstLoc;

// 批次中的所有 copy 使用相同的 copy 属性。
size_t attrsIdxs = 0;

// 启动批量内存传输
cudaMemcpyBatchAsync(&dsts[0], &srcs[0], &sizes[0], batch_size,
    &attrs, &attrsIdxs, 1 /*numAttrs*/, nullptr /*failIdx*/, stream);
```

最后需要介绍一个提示：是否希望使用 SM 或 CE 执行传输。对应字段是 `cudaMemcpyAttributes::flags`，可能的值包括：

- `cudaMemcpyFlagDefault`：默认行为；
- `cudaMemcpyFlagPreferOverlapWithCompute`：提示系统优先使用 CE 执行传输，使传输与计算重叠。该标志在非 Tegra 平台上会被忽略。

总的来说，关于 `cudaMemcpyBatchAsync`，要点如下：

- `cudaMemcpyBatchAsync` 函数及其 3D 变体允许程序员指定一批内存传输，从而摊销传输设置开销。
- 除源指针、目标指针和传输大小外，该函数还可以接受一个或多个 memory copy attribute，用于提供正在传输的内存类型及源指针对应的 stream ordering 行为，提示源和目标位置，并提示是否优先让传输与计算重叠（如果可能）或使用 SM 执行传输。
- runtime 可以根据上述信息，尽可能地优化传输。

## 3.1.6 环境变量（Environment Variables）

CUDA 提供了各种环境变量（见[第 5.2 节](../05-technical-appendices/environment-variables.html#cuda-环境变量-cuda-environment-variables)），这些变量会影响执行和性能。如果没有显式设置，CUDA 会为它们使用合理的默认值；但在具体场景中可能需要特殊处理，例如用于调试或获得更好的性能。

例如，增大环境变量 `CUDA_DEVICE_MAX_CONNECTIONS` 的值，可能有助于降低不同 CUDA stream 中的独立工作因虚假依赖而被串行化的可能性。使用相同底层资源时，可能引入这类虚假依赖。建议先使用默认值，只有在出现性能问题时才探索该环境变量的影响，例如不同 CUDA stream 中独立工作出现无法归因于其他因素（如可用 SM 资源不足）的意外串行化。需要注意的是，在 MPS 场景中，该环境变量的默认值不同，而且更低。

类似地，对于对延迟敏感的应用，将环境变量 `CUDA_MODULE_LOADING` 设置为 `EAGER` 可能更合适。这样可以把 module loading 的全部开销移到应用初始化阶段，移出关键路径。当前默认模式是 lazy module loading。在该默认模式下，也可以在应用初始化阶段为各种 kernel 添加“warm-up”调用，迫使 module loading 更早发生，从而达到类似 eager module loading 的效果。

有关各种 CUDA 环境变量的更多细节，请参阅 [CUDA Environment Variables](../05-technical-appendices/environment-variables.html#cuda-环境变量-cuda-environment-variables)。建议在启动应用**之前**将环境变量设置为新值；尝试在应用内部设置它们可能不会生效。
