---
title: 4.5 程序化依赖启动与同步
description: 程序化依赖启动和同步机制的完整中文翻译
---

# 4.5 程序化依赖启动与同步（Programmatic Dependent Launch and Synchronization）

*程序化依赖启动（Programmatic Dependent Launch）*机制允许有依赖关系的 *secondary* kernel 在其所依赖、且位于同一 CUDA stream 中的 *primary* kernel 执行完毕之前启动。从计算能力 9.0 的设备开始提供该机制。当 *secondary* kernel 中有大量工作不依赖 *primary* kernel 的结果时，这种技术可以带来性能收益。

## 4.5.1 背景（Background）

CUDA 应用通过在 GPU 上启动并执行多个 kernel 来使用 GPU。典型的 GPU 活动时间线如图 42 所示。

![图 42：GPU 活动时间线](/images/chapter-04/gpu-activity.png)

图 42：GPU 活动时间线。

这里，`secondary_kernel` 在 `primary_kernel` 完成执行后才启动。通常必须串行执行，因为 `secondary_kernel` 依赖 `primary_kernel` 产生的结果数据。如果 `secondary_kernel` 不依赖 `primary_kernel`，则可以使用 [CUDA Streams](../02-programming-gpus/asynchronous-execution.html#cuda-streams) 并发启动二者。即使 `secondary_kernel` 依赖 `primary_kernel`，也仍有一定的并发执行空间。例如，几乎所有 kernel 都有某种*前导（preamble）*部分，在此期间会执行清零缓冲区、加载常量值等任务。

![图 43：secondary_kernel 的前导部分](/images/chapter-04/secondary-kernel-preamble.png)

图 43 展示了 `secondary_kernel` 中可以并发执行而不影响应用程序的部分。注意，并发启动还可以把 `secondary_kernel` 的启动延迟隐藏在 `primary_kernel` 的执行过程中。

![图 44：primary_kernel 与 secondary_kernel 的并发执行](/images/chapter-04/preamble-overlap.png)

图 44 所示的 `secondary_kernel` 并发启动与执行可以通过*程序化依赖启动*实现。

*程序化依赖启动*会按下一节所述修改 CUDA kernel 启动 API。这些 API 至少需要计算能力 9.0，才能提供重叠执行。

## 4.5.2 API 说明（API Description）

在程序化依赖启动中，primary kernel 和 secondary kernel 在同一个 CUDA stream 中启动。当 primary kernel 已准备好让 secondary kernel 启动时，它应让所有 thread block 执行 `cudaTriggerProgrammaticLaunchCompletion`。secondary kernel 必须使用如下所示的可扩展启动 API 启动。

```cpp
__global__ void primary_kernel() {
   // 在 secondary kernel 启动前必须完成的初始工作

   // 触发 secondary kernel
   cudaTriggerProgrammaticLaunchCompletion();

   // 可以与 secondary kernel 同时进行的工作
}

__global__ void secondary_kernel()
{
   // 独立工作

   // 阻塞，直到 secondary kernel 所依赖的所有 primary kernel 完成，
   // 并将结果刷新到全局内存
   cudaGridDependencySynchronize();

   // 依赖工作
}

cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute[0].val.programmaticStreamSerializationAllowed = 1;
configSecondary.attrs = attribute;
configSecondary.numAttrs = 1;

primary_kernel<<<grid_dim, block_dim, 0, stream>>>();
cudaLaunchKernelEx(&configSecondary, secondary_kernel);
```

当使用 `cudaLaunchAttributeProgrammaticStreamSerialization` 属性启动 secondary kernel 时，CUDA 驱动可以安全地提前启动 secondary kernel，而不必等 primary kernel 完成并刷新内存后再启动它。

当 primary kernel 的所有 thread block 都已启动并执行 `cudaTriggerProgrammaticLaunchCompletion` 后，CUDA 驱动就可以启动 secondary kernel。如果 primary kernel 没有执行该触发器，则它会在 primary kernel 的所有 thread block 退出后隐式发生。

在这两种情况下，secondary kernel 的 thread block 都可能在 primary kernel 写入的数据可见之前启动。因此，当 secondary kernel 配置为使用*程序化依赖启动*时，它必须始终使用 `cudaGridDependencySynchronize` 或其他手段确认 primary kernel 的结果数据已经可用。

请注意，这些方法为 primary kernel 和 secondary kernel 并发执行提供了机会，但这种行为是机会性的，并不保证会导致 kernel 并发执行。以这种方式依赖并发执行是不安全的，可能导致死锁。

## 4.5.3 在 CUDA Graph 中使用（Use in CUDA Graphs）

程序化依赖启动可以通过 [stream capture](cuda-graphs.html#cuda-graphs-creating-a-graph-using-stream-capture) 或直接通过 [edge data](cuda-graphs.html#cuda-graphs-edge-data) 在 [CUDA Graphs](cuda-graphs.html#cuda-graphs) 中使用。若要在带边数据的 CUDA Graph 中编程使用此功能，请在连接两个 kernel 节点的边上使用 `cudaGraphDependencyType` 值 `cudaGraphDependencyTypeProgrammatic`。这种边类型会让上游 kernel 对下游 kernel 中的 `cudaGridDependencySynchronize()` 可见。该类型必须与 `cudaGraphKernelNodePortLaunchCompletion` 或 `cudaGraphKernelNodePortProgrammatic` 二者之一的 outgoing port 一起使用。

stream capture 产生的图等价关系如下：

| Stream 代码（缩略） | 生成的图边 |
|---|---|
| `cudaLaunchAttribute attribute; attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization; attribute.val.programmaticStreamSerializationAllowed = 1;` | `cudaGraphEdgeData edgeData; edgeData.type = cudaGraphDependencyTypeProgrammatic; edgeData.from_port = cudaGraphKernelNodePortProgrammatic;` |
| `cudaLaunchAttribute attribute; attribute.id = cudaLaunchAttributeProgrammaticEvent; attribute.val.programmaticEvent.triggerAtBlockStart = 0;` | `cudaGraphEdgeData edgeData; edgeData.type = cudaGraphDependencyTypeProgrammatic; edgeData.from_port = cudaGraphKernelNodePortProgrammatic;` |
| `cudaLaunchAttribute attribute; attribute.id = cudaLaunchAttributeProgrammaticEvent; attribute.val.programmaticEvent.triggerAtBlockStart = 1;` | `cudaGraphEdgeData edgeData; edgeData.type = cudaGraphDependencyTypeProgrammatic; edgeData.from_port = cudaGraphKernelNodePortLaunchCompletion;` |
