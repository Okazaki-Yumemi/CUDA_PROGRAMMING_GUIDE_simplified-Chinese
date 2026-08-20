---
title: 4.2 CUDA 图（CUDA Graphs）
description: CUDA Graph 的结构、构建、执行、更新、条件节点、图内存节点与设备端启动
---

# 4.2 CUDA 图（CUDA Graphs）

CUDA Graph 提供了 CUDA 中另一种工作提交模型。一个图由 Kernel 启动、数据移动等操作组成，操作之间通过依赖关系连接；图的定义与执行彼此分离，因此可以定义一次并反复启动。将图的定义和执行分开可以实现多种优化：首先，与 Stream 相比，CPU 启动成本更低，因为大量准备工作已经提前完成；其次，把完整工作流交给 CUDA 后，CUDA 可以执行 Stream 按片段提交机制无法实现的优化。

要理解 Graph 能实现的优化，可以看看 Stream 中发生的事情：当把一个 Kernel 放入 Stream 时，主机端驱动会执行一系列操作，为 GPU 执行该 Kernel 做准备。这些设置和启动 Kernel 所必需的操作，会成为每次提交 Kernel 都必须支付的额外开销。对于执行时间很短的 GPU Kernel，这项开销可能占端到端总执行时间的很大比例。创建一个包含将被多次启动的完整工作流的 CUDA Graph 后，这些开销只需在实例化期间为整个图支付一次；之后就可以以很低的开销反复启动图本身。

## 4.2.1 图结构（Graph Structure）

操作在图中构成节点，操作之间的依赖关系构成边。这些依赖关系约束了操作的执行顺序。

当某个操作所依赖的节点全部完成后，该操作就可以在任意时刻被调度。具体调度时机由 CUDA 系统决定。

### 4.2.1.1 节点类型（Node Types）

图节点可以是以下类型之一：

- Kernel
- CPU 函数调用
- 内存复制
- `memset`
- 空节点
- 等待一个 [CUDA Event](../02-programming-gpus/asynchronous-execution.html#cuda-events)
- 记录一个 [CUDA Event](../02-programming-gpus/asynchronous-execution.html#cuda-events)
- 发出一个[外部信号量](cuda-interoperability.html#external-resource-interoperability)
- 等待一个[外部信号量](cuda-interoperability.html#external-resource-interoperability)
- [条件节点](#4-2-4-条件图节点-conditional-graph-nodes)
- [内存节点](#4-2-5-图内存节点-graph-memory-nodes)
- 子图：用于执行单独的嵌套图，如下图所示。

![子图示例](/images/chapter-04/child-graph.png)

图 24：子图示例。

### 4.2.1.2 边数据（Edge Data）

CUDA 12.3 为 CUDA Graph 引入了边数据。目前，非默认边数据唯一的用途是启用[程序化依赖启动](../04-cuda-features/programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization)。

一般来说，边数据会修改由某条边指定的依赖关系，由三部分组成：出端口、入端口和类型。出端口指定关联边何时触发；入端口指定节点的哪一部分依赖于关联边；类型则修改两个端点之间的关系。

端口值取决于节点类型和方向，边类型也可能仅限于特定节点类型。在所有情况下，零初始化的边数据都表示默认行为：出端口 0 等待整个任务，入端口 0 阻塞整个任务，边类型 0 表示带有内存同步行为的完整依赖。

在不同的 Graph API 中，可以通过与关联节点并行的数组可选地指定边数据。如果输入参数中省略了边数据，则使用零初始化数据。如果作为输出（查询）参数省略了边数据，当被忽略的边数据全部为零初始化时 API 可以接受该调用；如果该调用会丢弃信息，则返回 `cudaErrorLossyQuery`。

部分 Stream capture API 也支持边数据：`cudaStreamBeginCaptureToGraph()`、`cudaStreamGetCaptureInfo()` 和 `cudaStreamUpdateCaptureDependencies()`。在这些情况下，下游节点尚不存在。边数据会关联到一条悬空边（半边），该边最终要么连接到后续捕获的节点，要么在 Stream capture 结束时被丢弃。注意，某些边类型不等待上游节点完全完成。在判断 Stream capture 是否已经完全重新加入源 Stream 时会忽略这些边，并且在 capture 结束时不能丢弃它们。参见 [Stream Capture](#4-2-2-1-2-stream-capture)。

没有节点类型定义额外的入端口，只有 Kernel 节点定义了额外的出端口。存在一种非默认依赖类型 `cudaGraphDependencyTypeProgrammatic`，它用于在两个 Kernel 节点之间启用[程序化依赖启动](../04-cuda-features/programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization)。

## 4.2.2 构建和运行图（Building and Running Graphs）

使用 Graph 提交工作分为三个不同阶段：定义、实例化和执行。

- 在**定义**或**创建**阶段，程序创建图中操作的描述，以及操作之间的依赖关系。
- **实例化**会获取图模板的快照、验证它，并执行大量工作设置和初始化，目标是尽量减少启动时需要完成的工作。生成的实例称为*可执行图*（executable graph）。
- **可执行图**可以被启动到 Stream 中，与其他 CUDA 工作类似。它可以启动任意次数，而不必重复实例化。

### 4.2.2.1 图创建（Graph Creation）

#### 4.2.2.1.1 Graph API（Graph APIs）

可以直接使用 Graph API 创建图。以下示例使用 `cudaGraphAddKernelNode` 将一个 Kernel 节点添加到图中。该 API 接收一个指向图的指针、依赖节点列表及其数量、Kernel 节点参数，并返回新创建节点的句柄。

```cpp
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

cudaGraphNode_t kernelNode;
cudaKernelNodeParams kernelParams = {0};
kernelParams.func = (void*)kernel;
kernelParams.gridDim = dim3(1, 1, 1);
kernelParams.blockDim = dim3(1, 1, 1);
kernelParams.sharedMemBytes = 0;
kernelParams.kernelParams = nullptr;
cudaGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelParams);
```

#### 4.2.2.1.2 Stream Capture（Stream Capture）

Stream capture 将 Stream 中的工作序列记录为 Graph。以下示例展示了如何使用 Stream capture 创建与上一节等价的图：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernel<<<1, 1, 0, stream>>>();
cudaStreamEndCapture(stream, &graph);
```

在 capture 期间提交到 Stream 的操作不会立即执行，而是被记录到图中。`cudaStreamEndCapture` 返回捕获得到的图；随后可以实例化并反复启动该图。

Stream capture 可以跨多个 Stream 进行，并能通过 Event 记录和等待表达跨 Stream 依赖。

##### 4.2.2.1.2.1 跨 Stream 依赖和 Event（Cross-stream Dependencies and Events）

在 capture 期间，一个 Stream 可以等待另一个正在 capture 的 Stream 记录的 Event。这样可以在最终 Graph 中创建跨 Stream 的依赖边。例如：

```cpp
cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
kernel1<<<1, 1, 0, stream1>>>();
cudaEventRecord(event, stream1);

cudaStreamBeginCapture(stream2, cudaStreamCaptureModeGlobal);
cudaStreamWaitEvent(stream2, event);
kernel2<<<1, 1, 0, stream2>>>();

cudaStreamEndCapture(stream1, &graph1);
cudaStreamEndCapture(stream2, &graph2);
```

若多个 Stream capture 被连接，最终需要将它们重新加入同一个源 Stream；否则 capture 会失效。`cudaStreamGetCaptureInfo` 和 `cudaStreamGetCaptureInfo_v2` 可以用于查询 capture 状态及其依赖信息。

##### 4.2.2.1.2.2 禁止和未处理的操作（Prohibited and Unhandled Operations）

并非所有 CUDA 操作都可以在 Stream capture 期间调用。禁止的操作通常包括会改变 capture 所在 Stream 状态的操作，例如在正在 capture 的 Stream 上调用 `cudaStreamBeginCapture`，或销毁该 Stream。某些操作可能会导致 capture 无效；此时 `cudaStreamEndCapture` 会返回 `cudaErrorStreamCaptureInvalidated`，并返回空图。

调用可能在 capture 期间改变全局状态的 API 时，应查阅该 API 的文档以确认其行为。不能处理的操作并不一定会立即报错，而可能在执行时产生未定义结果，因此应避免在 capture 范围内使用不受支持的 API。

##### 4.2.2.1.2.3 失效（Invalidation）

如果 capture 期间发生禁止操作，或某个操作使 capture 无效，则整个 capture 会被标记为失效。失效的 capture 不能生成可执行图；必须调用 `cudaStreamEndCapture` 清理 capture 状态，然后重新开始 capture。

##### 4.2.2.1.2.4 Capture 内省（Capture Introspection）

`cudaStreamIsCapturing` 可以判断 Stream 当前是否处于 capture 状态。`cudaStreamGetCaptureInfo` 可以返回 capture 模式、捕获 ID、依赖节点等信息。使用这些 API 时，调用方应根据返回状态处理 `cudaStreamCaptureStatusActive`、`cudaStreamCaptureStatusInvalidated` 和 `cudaStreamCaptureStatusNone`。

#### 4.2.2.1.3 汇总示例（Putting It All Together）

下面的示例使用 Stream capture 构建一个包含两个 Kernel 和一个内存复制操作的 Graph。示例中的代码按照“开始 capture、提交工作、结束 capture、实例化、启动并同步”的顺序执行。

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernel1<<<grid, block, 0, stream>>>(d_data);
kernel2<<<grid, block, 0, stream>>>(d_data);
cudaMemcpyAsync(h_data, d_data, bytes, cudaMemcpyDeviceToHost, stream);

cudaGraph_t graph;
cudaStreamEndCapture(stream, &graph);

cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
cudaGraphLaunch(graphExec, stream);
```

![创建 Graph 的流程示意图](/images/chapter-04/create-a-graph.png)

图 25：创建 Graph。

该简化示例旨在从概念上展示一个小型 Graph。在实际使用 CUDA Graph 的应用中，无论使用 Graph API 还是 Stream capture，都会更复杂。下面的代码并列展示了如何使用 Graph API 和 Stream capture 创建 CUDA Graph，以执行一个简单的两阶段归约算法。

![使用两阶段归约 Kernel 的 CUDA Graph 示例](/images/chapter-04/cuda_graph_reduction.png)

图 26：使用两阶段归约 Kernel 的 CUDA Graph 示例。

### 4.2.5.4 性能注意事项（Performance Considerations）

当多个 Graph 启动到同一个 Stream 时，CUDA 会尝试为它们分配相同的物理内存，因为这些 Graph 不可能重叠执行。作为避免重新映射开销的优化，Graph 的物理映射会在多次启动之间保留。如果之后某个 Graph 被启动到可能与其他 Graph 重叠的执行环境中（例如启动到另一个 Stream），CUDA 必须执行一些重新映射，因为并发 Graph 需要不同内存以避免数据损坏。

通常，以下操作可能导致 CUDA 对 Graph 内存重新映射：

- 改变 Graph 启动到的 Stream；
- 对 Graph 内存池执行 trim 操作，显式释放未使用内存（见[物理内存占用](#4-2-5-5-物理内存占用-physical-memory-footprint)）；
- 当另一个 Graph 的未释放分配映射到相同内存时重新启动 Graph；重新启动前会重新映射内存。

重新映射必须按执行顺序发生，但要等待该 Graph 之前的执行完成（否则仍在使用的内存可能被取消映射）。由于这种排序依赖，而且映射操作属于 OS 调用，映射可能相对昂贵。应用可以将包含分配内存节点的 Graph 始终启动到同一个 Stream，以避免这项开销。

#### 4.2.5.4.1 首次启动 / `cudaGraphUpload`（First Launch / cudaGraphUpload）

由于无法预先知道 Graph 将在哪个 Stream 执行，Graph 实例化期间不能分配或映射物理内存；映射会在 Graph 启动期间完成。调用 `cudaGraphUpload` 可以立即完成该 Graph 的全部映射，并将 Graph 与上传 Stream 关联，从而把分配成本从启动成本中分离出来。如果随后 Graph 启动到同一个 Stream，就不需要额外的重新映射。

使用不同的 Stream 上传和启动 Graph，其行为类似于切换 Stream，可能导致重新映射操作。此外，不相关的内存池管理可能从空闲 Stream 获取内存，这会抵消上传的影响。

### 4.2.5.5 物理内存占用（Physical Memory Footprint）

异步分配的内存池管理行为意味着，销毁一个包含内存节点的 Graph（即使其分配已经释放）也不会立即把物理内存返还给 OS 供其他进程使用。要显式将内存释放回 OS，应用应使用 `cudaDeviceGraphMemTrim` API。

`cudaDeviceGraphMemTrim` 会取消映射并释放图内存节点保留、但当前未使用的物理内存。仍未释放的分配以及已排队或正在运行的 Graph 被视为正在使用物理内存，不会受到影响。使用 trim API 会使物理内存可供其他分配 API、其他应用或进程使用，但当被 trim 的 Graph 下次启动时，CUDA 需要重新分配并重新映射内存。注意，`cudaDeviceGraphMemTrim` 操作的是与 `cudaMemPoolTrimTo()` 不同的内存池；图内存池不会暴露给 Stream 有序内存分配器。

应用可以通过 `cudaDeviceGetGraphMemAttribute` API 查询 Graph 的内存占用。查询属性 `cudaGraphMemAttrReservedMemCurrent` 会返回当前进程中驱动为图分配保留的物理内存量；查询 `cudaGraphMemAttrUsedMemCurrent` 会返回当前至少被一个 Graph 映射的物理内存量。任一属性都可用于跟踪 CUDA 是否为了分配 Graph 获取了新的物理内存。这两个属性也适合检查内存共享机制节省了多少内存。

### 4.2.5.6 对等访问（Peer Access）

可以配置图分配，使其能够从多个 GPU 访问；此时 CUDA 会按需将分配映射到对等 GPU。CUDA 允许具有不同映射要求的图分配复用同一虚拟地址。当这种情况发生时，地址范围会映射到不同分配所需的全部 GPU。因此，某个分配有时可能允许比创建时请求更多的对等访问；不过，依赖这些额外映射仍然是错误的。

#### 4.2.5.6.1 使用 Graph 节点 API 的对等访问（Peer Access with Graph Node APIs）

`cudaGraphAddNode` API 接受分配节点参数结构 `accessDescs` 数组字段中的映射请求。嵌入的 `poolProps.location` 结构指定分配的驻留设备。默认认为需要从分配 GPU 访问，因此应用不需要在 `accessDescs` 数组中为驻留设备指定条目。

```cpp
cudaGraphNodeParams allocNodeParams = { cudaGraphNodeTypeMemAlloc };
allocNodeParams.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
allocNodeParams.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
// specify device 1 as the resident device
allocNodeParams.alloc.poolProps.location.id = 1;
allocNodeParams.alloc.bytesize = size;

// allocate an allocation resident on device 1 accessible from device 1
cudaGraphAddNode(&allocNode, graph, NULL, NULL, 0, &allocNodeParams);

accessDescs[2];
// boilerplate for the access descs (only ReadWrite and Device access supported by the add node api)
accessDescs[0].flags = cudaMemAccessFlagsProtReadWrite;
accessDescs[0].location.type = cudaMemLocationTypeDevice;
accessDescs[1].flags = cudaMemAccessFlagsProtReadWrite;
accessDescs[1].location.type = cudaMemLocationTypeDevice;

// access being requested for device 0 & 2. Device 1 access requirement left implicit.
accessDescs[0].location.id = 0;
accessDescs[1].location.id = 2;

// access request array has 2 entries.
allocNodeParams.accessDescCount = 2;
allocNodeParams.accessDescs = accessDescs;

// allocate an allocation resident on device 1 accessible from devices 0, 1 and 2. (0 & 2 from the descriptors, 1 from it being the resident device).
cudaGraphAddNode(&allocNode, graph, NULL, NULL, 0, &allocNodeParams);
```

#### 4.2.5.6.2 使用 Stream capture 的对等访问（Peer Access with Stream Capture）

对于 Stream capture，分配节点会记录 capture 时分配内存池的对等可访问性。在捕获 `cudaMallocFromPoolAsync` 调用后再改变分配池的对等可访问性，不会影响 Graph 为该分配建立的映射。

```cpp
// boilerplate for the access descs (only ReadWrite and Device access supported by the add node api)
accessDesc.flags = cudaMemAccessFlagsProtReadWrite;
accessDesc.location.type = cudaMemLocationTypeDevice;
accessDesc.location.id = 1;

// let memPool be resident and accessible on device 0

cudaStreamBeginCapture(stream);
cudaMallocAsync(&dptr1, size, memPool, stream);
cudaStreamEndCapture(stream, &graph1);

cudaMemPoolSetAccess(memPool, &accessDesc, 1);

cudaStreamBeginCapture(stream);
cudaMallocAsync(&dptr2, size, memPool, stream);
cudaStreamEndCapture(stream, &graph2);

//The graph node allocating dptr1 would only have the device 0 accessibility even though memPool now has device 1 accessibility.
//The graph node allocating dptr2 will have device 0 and device 1 accessibility, since that was the pool accessibility at the time of the cudaMallocAsync call.
```

## 4.2.3 更新已实例化的图（Updating Instantiated Graphs）

当工作流发生变化时，Graph 会过期，必须修改它。图结构的重大变化（例如拓扑或节点类型变化）需要重新实例化，因为与拓扑相关的优化必须重新应用。不过，常见情况是只有节点参数（例如 Kernel 参数和内存地址）变化，而图拓扑保持不变。针对这种情况，CUDA 提供了轻量级的“Graph 更新”机制，可以原地修改某些节点参数，而无需重建整个图；这比重新实例化高效得多。

更新会在下一次启动 Graph 时生效，因此不会影响此前的 Graph 启动，即使此前的启动在更新时仍在运行也是如此。Graph 可以反复更新和重新启动，所以可以在一个 Stream 上排队多个更新和启动操作。

CUDA 提供两种更新已实例化 Graph 参数的机制：整图更新和单节点更新。整图更新允许用户提供一个拓扑相同的 `cudaGraph_t` 对象，其中的节点包含更新后的参数。单节点更新允许用户显式更新单个节点的参数。当需要更新大量节点，或者调用方不知道图拓扑（例如图来自对库调用的 Stream capture）时，使用更新后的 `cudaGraph_t` 更方便。当变化数量较少且用户持有需要更新节点的句柄时，建议使用单节点更新。单节点更新会跳过对未改变节点的拓扑检查和比较，因此在很多情况下更高效。

CUDA 还提供了启用和禁用单个节点的机制，且不会影响节点当前参数。

下面分别介绍这些方法。

### 4.2.3.1 整图更新（Whole Graph Update）

`cudaGraphExecUpdate()` 允许使用拓扑相同的图（“更新图”）中的参数，更新一个已实例化的图（“原始图”）。更新图的拓扑必须与用于实例化 `cudaGraphExec_t` 的原始图完全相同。此外，指定依赖关系的顺序也必须匹配。最后，CUDA 需要对汇节点（没有依赖节点的节点）保持一致的顺序；CUDA 依靠特定 API 调用的顺序实现一致的汇节点排序。

更具体地说，遵循以下规则可以使 `cudaGraphExecUpdate()` 确定性地配对原始图和更新图中的节点：

1. 对于任意正在 capture 的 Stream，在该 Stream 上操作的 API 调用必须保持相同顺序，包括 Event wait 以及不直接对应节点创建的其他 API 调用。
2. 直接操作某个图节点入边的 API 调用（包括捕获的 Stream API、节点添加 API 以及边添加/删除 API）必须保持相同顺序。此外，当这些 API 通过数组指定依赖关系时，数组内部指定依赖的顺序也必须匹配。
3. 汇节点必须保持一致的顺序。汇节点是 `cudaGraphExecUpdate()` 调用时最终图中没有依赖节点/出边的节点。以下操作会影响汇节点排序（如果存在），并且必须作为一个整体按相同顺序执行：
   - 产生汇节点的节点添加 API。
   - 删除边后使某节点成为汇节点。
   - `cudaStreamUpdateCaptureDependencies()`：如果它从 capture Stream 的依赖集合中删除汇节点。
   - `cudaStreamEndCapture()`。

下面的示例展示了如何使用 API 更新已实例化的图：

```cpp
cudaGraphExec_t graphExec = NULL;

for (int i = 0; i < 10; i++) {
    cudaGraph_t graph;
    cudaGraphExecUpdateResult updateResult;
    cudaGraphNode_t errorNode;

    // In this example we use stream capture to create the graph.
    // You can also use the Graph API to produce a graph.
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // Call a user-defined, stream based workload, for example
    do_cuda_work(stream);

    cudaStreamEndCapture(stream, &graph);

    // If we've already instantiated the graph, try to update it directly
    // and avoid the instantiation overhead
    if (graphExec != NULL) {
        // If the graph fails to update, errorNode will be set to the
        // node causing the failure and updateResult will be set to a
        // reason code.
        cudaGraphExecUpdate(graphExec, graph, &errorNode, &updateResult);
    }

    // Instantiate during the first iteration or whenever the update
    // fails for any reason
    if (graphExec == NULL || updateResult != cudaGraphExecUpdateSuccess) {

        // If a previous update failed, destroy the cudaGraphExec_t
        // before re-instantiating it
        if (graphExec != NULL) {
            cudaGraphExecDestroy(graphExec);
        }
        // Instantiate graphExec from graph. The error node and
        // error message parameters are unused here.
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    }

    cudaGraphDestroy(graph);
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
}
```

典型工作流是先使用 Stream capture 或 Graph API 创建初始 `cudaGraph_t`，然后按通常方式实例化并启动它。第一次启动后，使用与创建初始图相同的方法创建新的 `cudaGraph_t`，并调用 `cudaGraphExecUpdate()`。如果更新成功（由上例中的 `updateResult` 参数指示），就启动更新后的 `cudaGraphExec_t`。如果更新因任何原因失败，则调用 `cudaGraphExecDestroy()` 和 `cudaGraphInstantiate()`，销毁原来的 `cudaGraphExec_t` 并实例化一个新的实例。

也可以直接更新 `cudaGraph_t` 的节点（例如使用 `cudaGraphKernelNodeSetParams()`），然后更新 `cudaGraphExec_t`；不过，使用下一节介绍的显式节点更新 API 更高效。

条件句柄标志和默认值会作为整图更新的一部分被更新。

有关用法和当前限制，请参阅 [Graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH)。

### 4.2.3.2 单节点更新（Individual Node Update）

可以直接更新已实例化图的节点参数。这样既消除了实例化开销，也消除了创建新 `cudaGraph_t` 的开销。当需要更新的节点数相对于图中节点总数较少时，逐个更新节点更好。以下方法可用于更新 `cudaGraphExec_t` 节点：

| API | 节点类型 |
| --- | --- |
| `cudaGraphExecKernelNodeSetParams()` | Kernel 节点 |
| `cudaGraphExecMemcpyNodeSetParams()` | 内存复制节点 |
| `cudaGraphExecMemsetNodeSetParams()` | 内存设置节点 |
| `cudaGraphExecHostNodeSetParams()` | Host 节点 |
| `cudaGraphExecChildGraphNodeSetParams()` | 子图节点 |
| `cudaGraphExecEventRecordNodeSetEvent()` | Event 记录节点 |
| `cudaGraphExecEventWaitNodeSetEvent()` | Event 等待节点 |
| `cudaGraphExecExternalSemaphoresSignalNodeSetParams()` | 外部信号量发送节点 |
| `cudaGraphExecExternalSemaphoresWaitNodeSetParams()` | 外部信号量等待节点 |

表 8：单节点更新 API。

有关用法和当前限制，请参阅 [Graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH)。

### 4.2.3.3 单节点启用（Individual Node Enable）

已实例化 Graph 中的 Kernel、memset 和 memcpy 节点可以使用 `cudaGraphNodeSetEnabled()` API 启用或禁用。这样可以创建一个包含所需功能超集的图，并针对每次启动进行定制。可以使用 `cudaGraphNodeGetEnabled()` API 查询节点的启用状态。

禁用的节点在重新启用之前，功能上等价于空节点。启用或禁用节点不会影响节点参数。单节点更新或使用 `cudaGraphExecUpdate()` 的整图更新不会改变启用状态。节点被禁用期间进行的参数更新，会在节点重新启用时生效。

有关用法和当前限制，请参阅 [Graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH)。

### 4.2.3.4 Graph 更新限制（Graph Update Limitations）

Kernel 节点：

- 函数的所属 context 不能改变。
- 原本不使用 CUDA Dynamic Parallelism 的节点，不能更新为使用 CUDA Dynamic Parallelism 的函数。

`cudaMemset` 和 `cudaMemcpy` 节点：

- 操作数所分配/映射到的 CUDA 设备不能改变。
- 源/目标内存必须从与原始源/目标内存相同的 context 中分配。
- 只能改变一维 `cudaMemset`/`cudaMemcpy` 节点。

额外的 memcpy 节点限制：

- 不支持改变源或目标内存类型（例如 `cudaPitchedPtr`、`cudaArray_t` 等），也不支持改变传输类型（例如 `cudaMemcpyKind`）。

外部信号量等待节点和记录节点：

- 不支持改变信号量数量。

条件节点：

- 图之间的句柄创建和赋值顺序必须匹配。
- 不支持改变节点参数（例如条件图中的图数量、节点 context 等）。
- 条件主体图内节点参数的改变，受上述规则约束。

内存节点：

- 如果 `cudaGraph_t` 当前已作为另一个 `cudaGraphExec_t` 实例化，则不能使用它更新一个 `cudaGraphExec_t`。

Host 节点、Event 记录节点和 Event 等待节点的更新没有限制。

## 4.2.4 条件图节点（Conditional Graph Nodes）

条件节点允许对条件节点中包含的图执行条件分支和循环。这样可以把动态的、迭代式的工作流完整表示在 Graph 内部，并释放 Host CPU，以便它并行执行其他工作。

当条件节点的依赖满足后，条件值会在设备端求值。条件节点可以是以下类型之一：

- 条件 **IF 节点**：当节点执行时条件值非零，就执行一次主体图。如果条件值为零，还可以提供第二个主体图；此时第二个主体图执行一次。
- 条件 **WHILE 节点**：当节点执行时条件值非零，就执行主体图；之后只要条件值不为零，就继续执行主体图。
- 条件 **SWITCH 节点**：当条件值等于 `n` 时，执行以零为索引的第 `n` 个主体图一次。如果条件值不对应某个主体图，则不启动任何主体图。

条件值通过[条件句柄](#4-2-4-1-条件句柄-conditional-handles)访问，句柄必须在节点创建前创建。设备代码可以使用 `cudaGraphSetConditional()` 设置条件值。也可以在创建句柄时指定默认值，该默认值会在每次 Graph 启动时应用。

创建条件节点时会创建一个空图，并将句柄返回给用户，以便用户填充该图。条件主体图可以使用 [Graph API](#4-2-2-1-1-graph-api-graph-apis) 或 `cudaStreamBeginCaptureToGraph()` 填充。

条件节点可以嵌套。

### 4.2.4.1 条件句柄（Conditional Handles）

条件值由 `cudaGraphConditionalHandle` 表示，并通过 `cudaGraphConditionalHandleCreate()` 创建。

句柄必须与单个条件节点关联。句柄不能销毁，因此不需要跟踪它们。

如果创建句柄时指定了 `cudaGraphCondAssignDefault`，则每次图执行开始时，条件值都会初始化为指定的默认值。如果没有提供该标志，则每次图执行开始时条件值都是未定义的，代码不应假定条件值会在多次执行之间保留。

与句柄关联的默认值和标志会在[整图更新](#4-2-3-1-整图更新-whole-graph-update)期间更新。

### 4.2.4.2 条件节点主体图要求（Conditional Node Body Graph Requirements）

一般要求：

- 图中的所有节点必须位于同一设备上。
- 图只能包含 Kernel 节点、空节点、memcpy 节点、memset 节点、子图节点和条件节点。

Kernel 节点：

- 不允许图中的 Kernel 使用 CUDA Dynamic Parallelism 或 Device Graph Launch。
- 只要未使用 MPS，就允许 cooperative launch。

Memcpy/memset 节点：

- 只允许涉及设备内存和/或固定、设备映射的 Host 内存的复制/设置操作。
- 不允许涉及 CUDA array 的复制/设置操作。
- 实例化时两个操作数都必须能被当前设备访问。注意：复制操作会由图所在设备执行，即使目标内存位于另一台设备上。

### 4.2.4.3 条件 IF 节点（Conditional IF Nodes）

当 IF 节点执行时条件值非零，主体图会执行一次。下图展示了一个包含 3 个节点的图，其中中间节点 B 是条件节点：

![条件 IF 节点](/images/chapter-04/conditional-if-node.png)

图 27：条件 IF 节点。

下面的代码展示了如何创建包含 IF 条件节点的图。条件默认值由上游 Kernel 设置。条件主体使用 [Graph API](#4-2-2-1-1-graph-api-graph-apis) 填充。

```cpp
__global__ void setHandle(cudaGraphConditionalHandle handle, int value)
{
    ...
    // Set the condition value to the value passed to the kernel
    cudaGraphSetConditional(handle, value);
    ...
}

void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[2];
    int value = 1;

    // Create the graph
    cudaGraphCreate(&graph, 0);

    // Create the conditional handle; because no default value is provided, the condition value is undefined at the start of each graph execution
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);

    // Use a kernel upstream of the conditional to set the handle value
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)setHandle;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    kernelArgs[1] = &value;
    cudaGraphAddNode(&node, graph, NULL, 0, &params);

    // Create and add the conditional node
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeIf;
    cParams.conditional.size   = 1; // There is only an "if" body graph
    cudaGraphAddNode(&node, graph, &node, 1, &cParams);

    // Get the body graph of the conditional node
    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

    // Populate the body graph of the IF conditional node
    ...
    cudaGraphAddNode(&node, bodyGraph, NULL, 0, &params);

    // Instantiate and launch the graph
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    // Clean up
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}
```

IF 节点还可以有一个可选的第二主体图。当节点执行时条件值为零，第二主体图会执行一次。

```cpp
void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[2];
    int value = 1;

    // Create the graph
    cudaGraphCreate(&graph, 0);

    // Create the conditional handle; because no default value is provided, the condition value is undefined at the start of each graph execution
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);

    // Use a kernel upstream of the conditional to set the handle value
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)setHandle;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    kernelArgs[1] = &value;
    cudaGraphAddNode(&node, graph, NULL, 0, &params);

    // Create and add the IF conditional node
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeIf;
    cParams.conditional.size   = 2; // There is both an "if" and an "else" body graph
    cudaGraphAddNode(&node, graph, &node, 1, &cParams);

    // Get the body graphs of the conditional node
    cudaGraph_t ifBodyGraph = cParams.conditional.phGraph_out[0];
    cudaGraph_t elseBodyGraph = cParams.conditional.phGraph_out[1];

    // Populate the body graphs of the IF conditional node
    ...
    cudaGraphAddNode(&node, ifBodyGraph, NULL, 0, &params);
    ...
    cudaGraphAddNode(&node, elseBodyGraph, NULL, 0, &params);

    // Instantiate and launch the graph
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    // Clean up
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}
```

### 4.2.4.4 条件 WHILE 节点（Conditional WHILE Nodes）

当 WHILE 节点的条件非零时，主体图会持续执行。条件会在节点执行时以及主体图完成后求值。下图展示了一个包含 3 个节点的图，其中中间节点 B 是条件节点：

![条件 WHILE 节点](/images/chapter-04/conditional-while-node.png)

图 28：条件 WHILE 节点。

下面的代码展示了如何创建包含 WHILE 条件节点的图。句柄使用 `cudaGraphCondAssignDefault` 创建，因此不需要上游 Kernel。条件主体使用 [Graph API](#4-2-2-1-1-graph-api-graph-apis) 填充。

```cpp
__global__ void loopKernel(cudaGraphConditionalHandle handle, char *dPtr)
{
   // Decrement the value of dPtr and set the condition value to 0 once dPtr is 0
   if (--(*dPtr) == 0) {
      cudaGraphSetConditional(handle, 0);
   }
}

void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[2];

    // Allocate a byte of device memory to use as input
    char *dPtr;
    cudaMalloc((void **)&dPtr, 1);

    // Create the graph
    cudaGraphCreate(&graph, 0);

    // Create the conditional handle with a default value of 1
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault);

    // Create and add the WHILE conditional node
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeWhile;
    cParams.conditional.size   = 1;
    cudaGraphAddNode(&node, graph, NULL, 0, &cParams);

    // Get the body graph of the conditional node
    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

    // Populate the body graph of the conditional node
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)loopKernel;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    kernelArgs[1] = &dPtr;
    cudaGraphAddNode(&node, bodyGraph, NULL, 0, &params);

    // Initialize device memory, instantiate, and launch the graph
    cudaMemset(dPtr, 10, 1); // Set dPtr to 10; the loop will run until dPtr is 0
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    // Clean up
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(dPtr);
}

```

### 4.2.4.5 条件 SWITCH 节点（Conditional SWITCH Nodes）

当 SWITCH 节点执行时，如果条件值等于 `n`，则以零为索引的第 `n` 个主体图会执行一次。下图展示了一个包含 3 个节点的图，其中中间节点 B 是条件节点：

![条件 SWITCH 节点](/images/chapter-04/conditional-switch-node.png)

图 29：条件 SWITCH 节点。

下面的代码展示了如何创建包含 SWITCH 条件节点的图。条件值由上游 Kernel 设置。条件主体使用 [Graph API](#4-2-2-1-1-graph-api-graph-apis) 填充。

```cpp
__global__ void setHandle(cudaGraphConditionalHandle handle, int value)
{
    ...
    // Set the condition value to the value passed to the kernel
    cudaGraphSetConditional(handle, value);
    ...
}

void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[2];
    int value = 1;

    // Create the graph
    cudaGraphCreate(&graph, 0);

    // Create the conditional handle; because no default value is provided, the condition value is undefined at the start of each graph execution
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);

    // Use a kernel upstream of the conditional to set the handle value
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)setHandle;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    kernelArgs[1] = &value;
    cudaGraphAddNode(&node, graph, NULL, 0, &params);

    // Create and add the conditional SWITCH node
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeSwitch;
    cParams.conditional.size   = 5;
    cudaGraphAddNode(&node, graph, &node, 1, &cParams);

    // Get the body graphs of the conditional node
    cudaGraph_t *bodyGraphs = cParams.conditional.phGraph_out;

    // Populate the body graphs of the SWITCH conditional node
    ...
    cudaGraphAddNode(&node, bodyGraphs[0], NULL, 0, &params);
    ...
    cudaGraphAddNode(&node, bodyGraphs[4], NULL, 0, &params);

    // Instantiate and launch the graph
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    // Clean up
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}
```

## 4.2.5 图内存节点（Graph Memory Nodes）

### 4.2.5.1 简介（Introduction）

CUDA Graph Memory Nodes 使 Graph 能够管理设备内存的分配和释放，并在执行图时复用这些分配。它们把内存生命周期与 Graph 的生命周期关联起来，从而让 CUDA 在已知整个工作流的情况下优化内存复用。

图内存节点具有由 GPU 排序的生命周期语义，规定设备何时可以访问内存。这些 GPU 排序的生命周期语义支持由驱动管理的内存复用，并且与 Stream 有序分配 API `cudaMallocAsync` 和 `cudaFreeAsync` 的语义一致；创建 Graph 时可以捕获后两个 API。

在 Graph 的整个生命周期内（包括反复实例化和启动），图分配都具有固定地址。因此，图中的其他操作可以直接引用这些内存，而不需要 Graph update，即使 CUDA 改变了背后的物理内存。图内，生命周期不重叠的分配可以使用相同的底层物理内存。

CUDA 还可以在多个 Graph 之间复用相同物理内存，并根据 GPU 排序的生命周期语义对虚拟地址映射建立别名。例如，当不同 Graph 启动到同一个 Stream 时，如果分配只具有单图生命周期，CUDA 可以对同一物理内存使用虚拟别名来满足这些分配的需求。

### 4.2.5.2 API 基础（API Fundamentals）

图内存节点 API 允许在图中添加分配节点和释放节点。分配节点产生的指针可以作为后续节点的参数；释放节点必须依赖所有仍使用该指针的节点。

图内存分配由图实例管理。不同图之间默认不能直接访问由某个图分配的内存，除非通过图内存节点 API 或 Stream capture 建立适当的访问和依赖关系。

图分配在每次 Graph 运行时都被视为重新创建。图分配的生命周期不同于节点生命周期：它从 GPU 执行到达分配图节点时开始，在以下任一事件发生时结束：

- GPU 执行到达释放图节点；
- GPU 执行到达释放该内存的 `cudaFreeAsync()` Stream 调用；
- 调用 `cudaFree()` 释放内存时立即结束。

> **注意：** 销毁 Graph 不会自动释放任何仍存活的图分配内存，尽管它会结束分配节点的生命周期。之后必须在另一个 Graph 中释放该内存，或使用 `cudaFreeAsync()`/`cudaFree()` 释放。

与其他[图结构](#4-2-1-图结构-graph-structure)一样，图内存节点通过依赖边在图中排序。程序必须保证访问图内存的操作：

- 排序在分配节点之后；
- 排序在释放内存的操作之前。

图分配的生命周期通常依据 GPU 执行开始和结束，而不是 API 调用开始和结束。GPU 顺序指工作在 GPU 上实际运行的顺序，而不是工作入队或描述的顺序。因此，图分配被称为“GPU 排序”的分配。

#### 4.2.5.2.1 Graph 节点 API（Graph Node APIs）

可使用 `cudaGraphAddMemAllocNode()` 在图中添加内存分配节点，使用 `cudaGraphAddMemFreeNode()` 添加内存释放节点。分配节点的输出参数包含 `cudaMemAllocNodeParams`，其中的 `dptr` 是分配得到的设备指针，`bytesize` 是请求的字节数，`poolProps` 描述内存池属性。

也可以使用节点创建 API `cudaGraphAddNode` 显式创建图内存节点。添加 `cudaGraphNodeTypeMemAlloc` 节点时分配的地址，会通过传入的 `cudaGraphNodeParams` 结构的 `alloc::dptr` 字段返回给用户。分配图中所有使用图分配的操作，都必须排序在分配节点之后；同样，释放节点必须排序在图内所有使用该分配的操作之后。释放节点通过 `cudaGraphAddNode` 创建，并将节点类型设为 `cudaGraphNodeTypeMemFree`。

下图展示了一个包含分配节点和释放节点的示例图。Kernel 节点 **a**、**b** 和 **c** 都排序在分配节点之后、释放节点之前，因此可以访问该分配。Kernel 节点 **e** 没有排序在分配节点之后，因此不能安全访问内存。Kernel 节点 **d** 没有排序在释放节点之前，因此也不能安全访问内存。

![Kernel 节点](/images/chapter-04/kernel-nodes.png)

图 30：Kernel 节点。

典型的 Graph API 代码如下：

```cpp
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

cudaGraphNode_t allocNode;
cudaMemAllocNodeParams allocParams = {};
allocParams.bytesize = bytes;
allocParams.poolProps.allocType = cudaMemAllocationTypePinned;
allocParams.poolProps.location.type = cudaMemLocationTypeDevice;
allocParams.poolProps.location.id = device;
cudaGraphAddMemAllocNode(&allocNode, graph, nullptr, 0, &allocParams);

cudaGraphNode_t kernelNode;
cudaKernelNodeParams kernelParams = {};
kernelParams.func = (void *)kernel;
kernelParams.gridDim = grid;
kernelParams.blockDim = block;
kernelParams.kernelParams = args;
cudaGraphAddKernelNode(&kernelNode, graph, &allocNode, 1, &kernelParams);

cudaGraphNode_t freeNode;
cudaGraphAddMemFreeNode(&freeNode, graph, &kernelNode, 1, allocParams.dptr);
```

使用图内存节点时，分配节点必须在所有使用分配内存的节点之前，释放节点必须在这些使用者节点之后。依赖关系由图决定，内存分配和释放本身不会替代正确的执行依赖。

下面的代码建立图 30 中的图：

```cpp
// Create the graph - it starts out empty
cudaGraphCreate(&graph, 0);

// parameters for a basic allocation
cudaGraphNodeParams params = { cudaGraphNodeTypeMemAlloc };
params.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
params.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
// specify device 0 as the resident device
params.alloc.poolProps.location.id = 0;
params.alloc.bytesize = size;

cudaGraphAddNode(&allocNode, graph, NULL, NULL, 0, &params);

// create a kernel node that uses the graph allocation
cudaGraphNodeParams nodeParams = { cudaGraphNodeTypeKernel };
nodeParams.kernel.kernelParams[0] = params.alloc.dptr;
// ...set other kernel node parameters...

// add the kernel node to the graph
cudaGraphAddNode(&a, graph, &allocNode, NULL, 1, &nodeParams);
cudaGraphAddNode(&b, graph, &a, NULL, 1, &nodeParams);
cudaGraphAddNode(&c, graph, &a, NULL, 1, &nodeParams);
cudaGraphNode_t dependencies[2];
// kernel nodes b and c are using the graph allocation, so the freeing node must depend on them. Since the dependency of node b on node a establishes an indirect dependency, the free node does not need to explicitly depend on node a.
dependencies[0] = b;
dependencies[1] = c;
cudaGraphNodeParams freeNodeParams = { cudaGraphNodeTypeMemFree };
freeNodeParams.free.dptr = params.alloc.dptr;
cudaGraphAddNode(&freeNode, graph, dependencies, NULL, 2, freeNodeParams);
// free node does not depend on kernel node d, so it must not access the freed graph allocation.
cudaGraphAddNode(&d, graph, &c, NULL, 1, &nodeParams);

// node e does not depend on the allocation node, so it must not access the allocation. This would be true even if the freeNode depended on kernel node e.
cudaGraphAddNode(&e, graph, NULL, NULL, 0, &nodeParams);
```

#### 4.2.5.2.2 Stream Capture（Stream Capture）

Stream capture 期间的 `cudaMallocAsync()` 和 `cudaFreeAsync()` 可以被捕获为 Graph Memory Nodes。这样，已有使用异步内存分配器的 Stream 工作流就能转换为包含图内存生命周期的 Graph。

```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
void *ptr = nullptr;
cudaMallocAsync(&ptr, bytes, stream);
kernel<<<grid, block, 0, stream>>>((float *)ptr);
cudaFreeAsync(ptr, stream);
cudaStreamEndCapture(stream, &graph);
```

Stream capture 会保留分配和释放在 Stream 中出现的顺序。若同一分配的生命周期跨越多个 Stream，必须使用 Event 和跨 Stream 依赖保证 capture 能够正确合并。

图内存节点也可以通过捕获对应的 Stream 有序分配和释放调用 `cudaMallocAsync` 与 `cudaFreeAsync` 创建。在这种情况下，被捕获的分配 API 返回的虚拟地址可以由图中的其他操作使用。由于 Stream 有序依赖会被捕获到图中，只要 Stream 代码正确编写，Stream 有序分配 API 的排序要求就能保证图内存节点相对于被捕获的 Stream 操作正确排序。

为清晰起见，忽略 Kernel 节点 **d** 和 **e**，下面的代码使用 Stream capture 创建图 30 中的图：

```cpp
cudaMallocAsync(&dptr, size, stream1);
kernel_A<<< ..., stream1 >>>(dptr, ...);

// Fork into stream2
cudaEventRecord(event1, stream1);
cudaStreamWaitEvent(stream2, event1);

kernel_B<<< ..., stream1 >>>(dptr, ...);
// event dependencies translated into graph dependencies, so the kernel node created by the capture of kernel C will depend on the allocation node created by capturing the cudaMallocAsync call.
kernel_C<<< ..., stream2 >>>(dptr, ...);

// Join stream2 back to origin stream (stream1)
cudaEventRecord(event2, stream2);
cudaStreamWaitEvent(stream1, event2);

// Free depends on all work accessing the memory.
cudaFreeAsync(dptr, stream1);

// End capture in the origin stream
cudaStreamEndCapture(stream1, &graph);
```

#### 4.2.5.2.3 在分配图之外访问和释放图内存（Accessing and Freeing Graph Memory Outside of the Allocating Graph）

从一个 Graph Memory Node 分配出的内存，可以在分配图之外访问，但必须满足 API 定义的依赖和生命周期要求。访问内存的工作必须在分配完成后执行，释放必须在所有访问结束后执行；不要在对应 Graph 仍可能使用内存时提前释放它。

如果需要在另一个 Graph 中使用该指针，应通过 Graph 的依赖关系或 Stream capture 将两个工作流连接起来。仅仅把指针值传给另一个 Graph 并不能建立执行顺序。

图分配不一定要由分配它的 Graph 释放。当 Graph 不释放某个分配时，该分配会在 Graph 执行结束后继续存在，并可由后续 CUDA 操作访问。只要访问操作通过 CUDA Event 或其他 Stream 排序机制排在分配之后，这些分配就可以在另一个 Graph 中访问，或者直接通过 Stream 操作访问。之后可以通过普通的 `cudaFree`、`cudaFreeAsync` 调用释放，也可以通过启动带有相应释放节点的另一个 Graph 释放，或者通过再次启动分配 Graph 释放（前提是该 Graph 使用了 [cudaGraphInstantiateFlagAutoFreeOnLaunch](#4-2-5-2-4-cudagraphinstantiateflagautofreeonlaunch) 标志实例化）。在内存已被释放后访问它是非法的；释放操作必须通过 Graph 依赖、CUDA Event 和其他 Stream 排序机制排在所有访问内存的操作之后。

> **注意：** 由于图分配可能共享底层物理内存，释放操作必须排在所有设备操作完成之后。带外同步（例如计算 Kernel 内部基于内存的同步）不足以建立内存写入与释放操作之间的排序。更多信息请参阅关于一致性和相干性的[虚拟别名支持](../04-cuda-features/virtual-memory-management.html#virtual-aliasing-support)规则。

下面三个代码片段分别展示如何在分配 Graph 外访问图分配，并通过单个 Stream、Stream 之间的 Event、以及内置在分配和释放 Graph 中的 Event 正确建立排序。

使用单个 Stream 建立排序：

```cpp
// Contents of allocating graph
void *dptr;
cudaGraphNodeParams params = { cudaGraphNodeTypeMemAlloc };
params.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
params.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
params.alloc.bytesize = size;
cudaGraphAddNode(&allocNode, allocGraph, NULL, NULL, 0, &params);
dptr = params.alloc.dptr;

cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);

cudaGraphLaunch(allocGraphExec, stream);
kernel<<< ..., stream >>>(dptr, ...);
cudaFreeAsync(dptr, stream);
```

通过记录和等待 CUDA Event 建立排序：

```cpp
// Contents of allocating graph
void *dptr;

// Contents of allocating graph
cudaGraphAddNode(&allocNode, allocGraph, NULL, NULL, 0, &allocNodeParams);
dptr = allocNodeParams.alloc.dptr;

// contents of consuming/freeing graph
kernelNodeParams.kernel.kernelParams[0] = allocNodeParams.alloc.dptr;
cudaGraphAddNode(&freeNode, freeGraph, NULL, NULL, 1, dptr);

cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);
cudaGraphInstantiate(&freeGraphExec, freeGraph, NULL, NULL, 0);

cudaGraphLaunch(allocGraphExec, allocStream);

// establish the dependency of stream2 on the allocation node
// note: the dependency could also have been established with a stream synchronize operation
cudaEventRecord(allocEvent, allocStream);
cudaStreamWaitEvent(stream2, allocEvent);

kernel<<< ..., stream2 >>> (dptr, ...);

// establish the dependency between the stream 3 and the allocation use
cudaStreamRecordEvent(streamUseDoneEvent, stream2);
cudaStreamWaitEvent(stream3, streamUseDoneEvent);

// it is now safe to launch the freeing graph, which may also access the memory
cudaGraphLaunch(freeGraphExec, stream3);
```

使用 Graph 外部 Event 节点建立排序：

```cpp
// Contents of allocating graph
void *dptr;
cudaEvent_t allocEvent; // event indicating when the allocation will be ready for use.
cudaEvent_t streamUseDoneEvent; // event indicating when the stream operations are done with the allocation.

// Contents of allocating graph with event record node
cudaGraphAddNode(&allocNode, allocGraph, NULL, NULL, 0, &allocNodeParams);
dptr = allocNodeParams.alloc.dptr;
// note: this event record node depends on the alloc node

cudaGraphNodeParams allocEventNodeParams = { cudaGraphNodeTypeEventRecord };
allocEventNodeParams.eventRecord.event = allocEvent;
cudaGraphAddNode(&recordNode, allocGraph, &allocNode, NULL, 1, allocEventNodeParams);
cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);

// contents of consuming/freeing graph with event wait nodes
cudaGraphNodeParams streamWaitEventNodeParams = { cudaGraphNodeTypeEventWait };
streamWaitEventNodeParams.eventWait.event = streamUseDoneEvent;
cudaGraphAddNode(&streamUseDoneEventNode, waitAndFreeGraph, NULL, NULL, 0, streamWaitEventNodeParams);

cudaGraphNodeParams allocWaitEventNodeParams = { cudaGraphNodeTypeEventWait };
allocWaitEventNodeParams.eventWait.event = allocEvent;
cudaGraphAddNode(&allocReadyEventNode, waitAndFreeGraph, NULL, NULL, 0, allocWaitEventNodeParams);

kernelNodeParams->kernelParams[0] = allocNodeParams.alloc.dptr;

// The allocReadyEventNode provides ordering with the alloc node for use in a consuming graph.
cudaGraphAddNode(&kernelNode, waitAndFreeGraph, &allocReadyEventNode, NULL, 1, &kernelNodeParams);

// The free node has to be ordered after both external and internal users.
// Thus the node must depend on both the kernelNode and the streamUseDoneEventNode.
dependencies[0] = kernelNode;
dependencies[1] = streamUseDoneEventNode;

cudaGraphNodeParams freeNodeParams = { cudaGraphNodeTypeMemFree };
freeNodeParams.free.dptr = dptr;
cudaGraphAddNode(&freeNode, waitAndFreeGraph, &dependencies, NULL, 2, freeNodeParams);
cudaGraphInstantiate(&waitAndFreeGraphExec, waitAndFreeGraph, NULL, NULL, 0);

cudaGraphLaunch(allocGraphExec, allocStream);

// establish the dependency of stream2 on the event node satisfies the ordering requirement
cudaStreamWaitEvent(stream2, allocEvent);
kernel<<< ..., stream2 >>> (dptr, ...);
cudaStreamRecordEvent(streamUseDoneEvent, stream2);

// the event wait node in the waitAndFreeGraphExec establishes the dependency on the "readyForFreeEvent" that is needed to prevent the kernel running in stream two from accessing the allocation after the free node in execution order.
cudaGraphLaunch(waitAndFreeGraphExec, stream3);
```

#### 4.2.5.2.4 `cudaGraphInstantiateFlagAutoFreeOnLaunch`（cudaGraphInstantiateFlagAutoFreeOnLaunch）

使用 `cudaGraphInstantiateFlagAutoFreeOnLaunch` 实例化 Graph 时，Graph Memory Nodes 分配的内存会在 Graph 启动时根据该标志自动管理。该机制适合不需要在 Graph 启动之间保留分配内容的场景；如果需要跨启动保留数据，应根据 API 语义选择合适的生命周期和释放方式。

在正常情况下，如果 Graph 有未释放的内存分配，CUDA 会阻止再次启动该 Graph，因为多个位于相同地址的分配会造成内存泄漏。使用 `cudaGraphInstantiateFlagAutoFreeOnLaunch` 标志实例化 Graph 后，即使仍有未释放的分配，也可以重新启动 Graph；此时，每次启动都会自动插入对未释放分配的异步释放操作。

自动启动释放适用于单生产者、多消费者算法。每次迭代中，生产者 Graph 创建多个分配；根据运行时条件，一组数量可变的消费者访问这些分配。由于后续消费者可能仍需要访问这些内存，消费者无法释放分配。自动启动释放使启动循环不需要跟踪生产者分配，而是将这部分信息隔离在生产者的创建和销毁逻辑中。通常，自动启动释放可以简化原本必须在每次重新启动前释放 Graph 所有分配的算法。

> **注意：** `cudaGraphInstantiateFlagAutoFreeOnLaunch` 不会改变 Graph 销毁行为。即使 Graph 使用该标志实例化，应用仍必须显式释放未释放的内存，以避免内存泄漏。下面的代码展示了如何使用该标志简化单生产者/多消费者算法：

```cpp
// Create producer graph which allocates memory and populates it with data
cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
cudaMallocAsync(&data1, blocks * threads, cudaStreamPerThread);
cudaMallocAsync(&data2, blocks * threads, cudaStreamPerThread);
produce<<<blocks, threads, 0, cudaStreamPerThread>>>(data1, data2);
...
cudaStreamEndCapture(cudaStreamPerThread, &graph);
cudaGraphInstantiateWithFlags(&producer,
                              graph,
                              cudaGraphInstantiateFlagAutoFreeOnLaunch);
cudaGraphDestroy(graph);

// Create first consumer graph by capturing an asynchronous library call
cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
consumerFromLibrary(data1, cudaStreamPerThread);
cudaStreamEndCapture(cudaStreamPerThread, &graph);
cudaGraphInstantiateWithFlags(&consumer1, graph, 0); //regular instantiation
cudaGraphDestroy(graph);

// Create second consumer graph
cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
consume2<<<blocks, threads, 0, cudaStreamPerThread>>>(data2);
...
cudaStreamEndCapture(cudaStreamPerThread, &graph);
cudaGraphInstantiateWithFlags(&consumer2, graph, 0);
cudaGraphDestroy(graph);

// Launch in a loop
bool launchConsumer2 = false;
do {
    cudaGraphLaunch(producer, myStream);
    cudaGraphLaunch(consumer1, myStream);
    if (launchConsumer2) {
        cudaGraphLaunch(consumer2, myStream);
    }
} while (determineAction(&launchConsumer2));

cudaFreeAsync(data1, myStream);
cudaFreeAsync(data2, myStream);

cudaGraphExecDestroy(producer);
cudaGraphExecDestroy(consumer1);
cudaGraphExecDestroy(consumer2);
```

#### 4.2.5.2.5 子图中的内存节点（Memory Nodes in Child Graphs）

子图可以包含图内存分配和释放节点，但内存的生命周期必须在所属图的依赖范围内保持有效。父图不能假定子图外的节点可以在子图释放内存后继续访问该指针；需要通过依赖边明确表达所有使用关系。

CUDA 12.9 引入了将子图所有权移动到父图的能力。被移动到父图的子图可以包含内存分配和释放节点。这使得包含分配或释放节点的子图可以在加入父图之前独立构建。

移动后，子图受到以下限制：

- 不能独立实例化或销毁；
- 不能作为另一个父图的子图添加；
- 不能作为 `cuGraphExecUpdate` 的参数使用；
- 不能再添加内存分配或释放节点。

```cpp
// Create the child graph
cudaGraphCreate(&child, 0);

// parameters for a basic allocation
cudaGraphNodeParams allocNodeParams = { cudaGraphNodeTypeMemAlloc };
allocNodeParams.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
allocNodeParams.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
// specify device 0 as the resident device
allocNodeParams.alloc.poolProps.location.id = 0;
allocNodeParams.alloc.bytesize = size;

cudaGraphAddNode(&allocNode, child, NULL, NULL, 0, &allocNodeParams);
// Additional nodes using the allocation could be added here
cudaGraphNodeParams freeNodeParams = { cudaGraphNodeTypeMemFree };
freeNodeParams.free.dptr = allocNodeParams.alloc.dptr;
cudaGraphAddNode(&freeNode, child, &allocNode, NULL, 1, freeNodeParams);

// Create the parent graph
cudaGraphCreate(&parent, 0);

// Move the child graph to the parent graph
cudaGraphNodeParams childNodeParams = { cudaGraphNodeTypeGraph };
childNodeParams.graph.graph = child;
childNodeParams.graph.ownership = cudaGraphChildGraphOwnershipMove;
cudaGraphAddNode(&parentNode, parent, NULL, NULL, 0, &childNodeParams);
```

### 4.2.5.3 优化的内存复用（Optimized Memory Reuse）

CUDA 通过两种方式复用内存：

- Graph 内的虚拟和物理内存复用基于虚拟地址分配，类似 Stream 有序分配器；
- Graph 之间的物理内存复用使用虚拟别名：不同 Graph 可以把同一物理内存映射到各自唯一的虚拟地址。

#### 4.2.5.3.1 图内地址复用（Address Reuse within a Graph）

CUDA 可以把不重叠生命周期的不同分配放到相同虚拟地址范围，从而在 Graph 内复用内存。由于虚拟地址可能复用，生命周期不相交的不同分配对应的指针不保证唯一。

下图展示了添加一个新的分配节点（2），它可以复用依赖节点（1）释放的地址。

![添加新的分配节点 2](/images/chapter-04/new-alloc-node.png)

图 31：添加新的分配节点 2。

下图展示了添加新的分配节点（4）。新的分配节点不依赖释放节点（2），因此不能复用关联分配节点（2）的地址。如果分配节点（2）使用了释放节点（1）释放的地址，那么新的分配节点 3 就需要新地址。

![添加新的分配节点 3](/images/chapter-04/adding-new-alloc-nodes.png)

图 32：添加新的分配节点 3。

#### 4.2.5.3.2 物理内存管理和共享（Physical Memory Management and Sharing）

CUDA 负责在 GPU 顺序到达分配节点之前，将物理内存映射到虚拟地址。为了优化内存占用和映射开销，如果多个 Graph 不会同时运行，它们可以对不同分配使用相同物理内存；不过，如果物理页同时绑定到多个正在执行的 Graph，或者绑定到一个仍未释放的图分配，则不能复用这些物理页。

CUDA 可以在 Graph 实例化、启动或执行期间随时更新物理内存映射。CUDA 也可能在未来的 Graph 启动之间引入同步，以防止仍存活的图分配引用同一物理内存。与任何“分配-释放-分配”模式一样，如果程序在分配生命周期之外访问指针，错误访问可能静默读写另一个分配拥有的活动数据（即使该分配的虚拟地址唯一）。使用 Compute Sanitizer 工具可以捕获此错误。

下图展示了在同一 Stream 中顺序启动的 Graph。此例中每个 Graph 都释放它分配的全部内存。由于同一 Stream 中的 Graph 从不并发运行，CUDA 可以、也应当使用同一物理内存满足所有分配。

![顺序启动的 Graph](/images/chapter-04/sequentially-launched-graphs.png)

图 33：顺序启动的 Graph。

#### Graph API

```cpp
void cudaGraphsManual(float  *inputVec_h,
                      float  *inputVec_d,
                      double *outputVec_d,
                      double *result_d,
                      size_t  inputSize,
                      size_t  numOfBlocks)
{
   cudaStream_t                 streamForGraph;
   cudaGraph_t                  graph;
   std::vector<cudaGraphNode_t> nodeDependencies;
   cudaGraphNode_t              memcpyNode, kernelNode, memsetNode;
   double                       result_h = 0.0;

   cudaStreamCreate(&streamForGraph);

   cudaKernelNodeParams kernelNodeParams = {0};
   cudaMemcpy3DParms    memcpyParams     = {0};
   cudaMemsetParams     memsetParams     = {0};

   memcpyParams.srcArray = NULL;
   memcpyParams.srcPos   = make_cudaPos(0, 0, 0);
   memcpyParams.srcPtr   = make_cudaPitchedPtr(inputVec_h, sizeof(float) * inputSize, inputSize, 1);
   memcpyParams.dstArray = NULL;
   memcpyParams.dstPos   = make_cudaPos(0, 0, 0);
   memcpyParams.dstPtr   = make_cudaPitchedPtr(inputVec_d, sizeof(float) * inputSize, inputSize, 1);
   memcpyParams.extent   = make_cudaExtent(sizeof(float) * inputSize, 1, 1);
   memcpyParams.kind     = cudaMemcpyHostToDevice;

   memsetParams.dst         = (void *)outputVec_d;
   memsetParams.value       = 0;
   memsetParams.pitch       = 0;
   memsetParams.elementSize = sizeof(float); // elementSize can be max 4 bytes
   memsetParams.width       = numOfBlocks * 2;
   memsetParams.height      = 1;

   cudaGraphCreate(&graph, 0);
   cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams);
   cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams);

   nodeDependencies.push_back(memsetNode);
   nodeDependencies.push_back(memcpyNode);

   void *kernelArgs[4] = {(void *)&inputVec_d, (void *)&outputVec_d, &inputSize, &numOfBlocks};

   kernelNodeParams.func           = (void *)reduce;
   kernelNodeParams.gridDim        = dim3(numOfBlocks, 1, 1);
   kernelNodeParams.blockDim       = dim3(THREADS_PER_BLOCK, 1, 1);
   kernelNodeParams.sharedMemBytes = 0;
   kernelNodeParams.kernelParams   = (void **)kernelArgs;
   kernelNodeParams.extra          = NULL;

   cudaGraphAddKernelNode(
      &kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams);

   nodeDependencies.clear();
   nodeDependencies.push_back(kernelNode);

   memset(&memsetParams, 0, sizeof(memsetParams));
   memsetParams.dst         = result_d;
   memsetParams.value       = 0;
   memsetParams.elementSize = sizeof(float);
   memsetParams.width       = 2;
   memsetParams.height      = 1;
   cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams);

   nodeDependencies.push_back(memsetNode);

   memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
   kernelNodeParams.func           = (void *)reduceFinal;
   kernelNodeParams.gridDim        = dim3(1, 1, 1);
   kernelNodeParams.blockDim       = dim3(THREADS_PER_BLOCK, 1, 1);
   kernelNodeParams.sharedMemBytes = 0;
   void *kernelArgs2[3]            = {(void *)&outputVec_d, (void *)&result_d, &numOfBlocks};
   kernelNodeParams.kernelParams   = kernelArgs2;
   kernelNodeParams.extra          = NULL;

   cudaGraphAddKernelNode(
      &kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams);

   nodeDependencies.clear();
   nodeDependencies.push_back(kernelNode);

   memset(&memcpyParams, 0, sizeof(memcpyParams));

   memcpyParams.srcArray = NULL;
   memcpyParams.srcPos   = make_cudaPos(0, 0, 0);
   memcpyParams.srcPtr   = make_cudaPitchedPtr(result_d, sizeof(double), 1, 1);
   memcpyParams.dstArray = NULL;
   memcpyParams.dstPos   = make_cudaPos(0, 0, 0);
   memcpyParams.dstPtr   = make_cudaPitchedPtr(&result_h, sizeof(double), 1, 1);
   memcpyParams.extent   = make_cudaExtent(sizeof(double), 1, 1);
   memcpyParams.kind     = cudaMemcpyDeviceToHost;

   cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(), nodeDependencies.size(), &memcpyParams);
   nodeDependencies.clear();
   nodeDependencies.push_back(memcpyNode);

   cudaGraphNode_t    hostNode;
   cudaHostNodeParams hostParams = {0};
   hostParams.fn                 = myHostNodeCallback;
   callBackData_t hostFnData;
   hostFnData.data     = &result_h;
   hostFnData.fn_name  = "cudaGraphsManual";
   hostParams.userData = &hostFnData;

   cudaGraphAddHostNode(&hostNode, graph, nodeDependencies.data(), nodeDependencies.size(), &hostParams);
}
```

#### Stream Capture

```cpp
void cudaGraphsUsingStreamCapture(float  *inputVec_h,
                      float  *inputVec_d,
                      double *outputVec_d,
                      double *result_d,
                      size_t  inputSize,
                      size_t  numOfBlocks)
{
   cudaStream_t stream1, stream2, stream3, streamForGraph;
   cudaEvent_t  forkStreamEvent, memsetEvent1, memsetEvent2;
   cudaGraph_t  graph;
   double       result_h = 0.0;

   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);
   cudaStreamCreate(&stream3);
   cudaStreamCreate(&streamForGraph);

   cudaEventCreate(&forkStreamEvent);
   cudaEventCreate(&memsetEvent1);
   cudaEventCreate(&memsetEvent2);

   cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

   cudaEventRecord(forkStreamEvent, stream1);
   cudaStreamWaitEvent(stream2, forkStreamEvent, 0);
   cudaStreamWaitEvent(stream3, forkStreamEvent, 0);

   cudaMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * inputSize, cudaMemcpyDefault, stream1);

   cudaMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks, stream2);

   cudaEventRecord(memsetEvent1, stream2);

   cudaMemsetAsync(result_d, 0, sizeof(double), stream3);
   cudaEventRecord(memsetEvent2, stream3);

   cudaStreamWaitEvent(stream1, memsetEvent1, 0);

   reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream1>>>(inputVec_d, outputVec_d, inputSize, numOfBlocks);

   cudaStreamWaitEvent(stream1, memsetEvent2, 0);

   reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream1>>>(outputVec_d, result_d, numOfBlocks);
   cudaMemcpyAsync(&result_h, result_d, sizeof(double), cudaMemcpyDefault, stream1);

   callBackData_t hostFnData = {0};
   hostFnData.data           = &result_h;
   hostFnData.fn_name        = "cudaGraphsUsingStreamCapture";
   cudaHostFn_t fn           = myHostNodeCallback;
   cudaLaunchHostFunc(stream1, fn, &hostFnData);
   cudaStreamEndCapture(stream1, &graph);
}
```

### 4.2.2.2 图实例化（Graph Instantiation）

无论是通过 Graph API 还是 Stream capture 创建了 Graph，都必须先将其实例化以创建可执行图，之后才能启动。假设 `cudaGraph_t graph` 已成功创建，下面的代码将实例化该图并创建 `cudaGraphExec_t graphExec`：

```cpp
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
```

### 4.2.2.3 图执行（Graph Execution）

创建 Graph 并将其实例化为可执行图后，就可以启动它。假设 `cudaGraphExec_t graphExec` 已成功创建，下面的代码会将图启动到指定的 Stream 中：

```cpp
cudaGraphLaunch(graphExec, stream);
```

将前面的步骤组合起来，并使用 [Stream Capture](#4-2-2-1-2-stream-capture) 示例，下面的代码会创建、实例化并启动一个图：

```cpp
cudaGraph_t graph;

cudaStreamBeginCapture(stream);

kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);

cudaStreamEndCapture(stream, &graph);

cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
cudaGraphLaunch(graphExec, stream);
```

## 4.2.6 设备端 Graph 启动（Device Graph Launch）

许多工作流需要在运行时根据数据相关的决策选择并执行不同操作。与其把决策过程交给 Host（这可能需要从设备往返一次），用户可能更希望在设备上完成决策。为此，CUDA 提供了从设备启动 Graph 的机制。

设备端 Graph 启动为从设备执行动态控制流提供了方便方式，既可以实现简单循环，也可以实现复杂的设备端工作调度器。

后文把可以从设备启动的 Graph 称为**设备 Graph**，把不能从设备启动的 Graph 称为**Host Graph**。

设备 Graph 既可以从 Host 启动，也可以从设备启动；Host Graph 只能从 Host 启动。与 Host 启动不同，如果设备端在某个设备 Graph 的前一次启动仍在运行时再次启动该 Graph，会返回 `cudaErrorInvalidValue`；因此，设备 Graph 不能同时被设备端启动两次。从 Host 和设备端同时启动同一个设备 Graph 会产生未定义行为。

### 4.2.6.1 设备端 Graph 创建（Device Graph Creation）

要从设备启动 Graph，必须显式地为设备端启动实例化它：向 `cudaGraphInstantiate()` 调用传递 `cudaGraphInstantiateFlagDeviceLaunch` 标志即可。与 Host Graph 一样，设备 Graph 的结构在实例化时固定，若要更新必须重新实例化，而且实例化只能在 Host 上执行。要将 Graph 实例化为可供设备端启动的 Graph，它必须满足多项要求。

#### 4.2.6.1.1 设备 Graph 要求（Device Graph Requirements）

一般要求：

- 图中的所有节点必须位于同一设备上。
- 图只能包含 Kernel 节点、memcpy 节点、memset 节点和子图节点。

Kernel 节点：

- 不允许图中的 Kernel 使用 CUDA Dynamic Parallelism。
- 只要未使用 MPS，就允许 cooperative launch。

Memcpy 节点：

- 只允许涉及设备内存和/或固定、设备映射 Host 内存的复制。
- 不允许涉及 CUDA array 的复制。
- 实例化时两个操作数都必须能被当前设备访问。注意：复制操作会由图所在设备执行，即使目标内存位于另一台设备上。

#### 4.2.6.1.2 设备 Graph 上传（Device Graph Upload）

要在设备上启动 Graph，必须先将 Graph 上传到设备，以准备必要的设备资源。有两种方式可以完成上传。

第一种方式是显式上传：调用 `cudaGraphUpload()`，或者在实例化时通过 `cudaGraphInstantiateWithParams()` 请求上传。

另一种方式是先从 Host 启动 Graph；Host 启动会隐式执行上传步骤。

下面给出了三种方法的示例：

```cpp
// Explicit upload after instantiation
cudaGraphInstantiate(&deviceGraphExec1, deviceGraph1, cudaGraphInstantiateFlagDeviceLaunch);
cudaGraphUpload(deviceGraphExec1, stream);

// Explicit upload as part of instantiation
cudaGraphInstantiateParams instantiateParams = {0};
instantiateParams.flags = cudaGraphInstantiateFlagDeviceLaunch | cudaGraphInstantiateFlagUpload;
instantiateParams.uploadStream = stream;
cudaGraphInstantiateWithParams(&deviceGraphExec2, deviceGraph2, &instantiateParams);

// Implicit upload via host launch
cudaGraphInstantiate(&deviceGraphExec3, deviceGraph3, cudaGraphInstantiateFlagDeviceLaunch);
cudaGraphLaunch(deviceGraphExec3, stream);
```

#### 4.2.6.1.3 设备 Graph 更新（Device Graph Update）

设备 Graph 只能从 Host 更新；可执行 Graph 更新后，必须重新上传到设备，修改才能生效。可以使用[设备 Graph 上传](#4-2-6-1-2-设备-graph-上传-device-graph-upload)一节介绍的相同方法完成重新上传。与 Host Graph 不同，在更新应用期间从设备启动设备 Graph 会产生未定义行为。

### 4.2.6.2 设备端启动（Device Launch）

设备 Graph 可以通过 `cudaGraphLaunch()` 从 Host 或设备端启动；该函数在设备端和 Host 端具有相同签名。设备 Graph 在 Host 和设备端使用同一个句柄。设备端启动时，设备 Graph 必须从另一个 Graph 中启动。

设备端 Graph 启动是按线程进行的，多个线程可以同时启动，因此用户需要选择一个线程来启动给定 Graph。

与 Host 启动不同，设备 Graph 不能启动到普通 CUDA Stream，只能启动到互不相同的命名 Stream；每个命名 Stream 表示特定的启动模式。可用启动模式如下：

| Stream | 启动模式 |
| --- | --- |
| `cudaStreamGraphFireAndForget` | Fire-and-forget 启动 |
| `cudaStreamGraphTailLaunch` | Tail 启动 |
| `cudaStreamGraphFireAndForgetAsSibling` | Sibling 启动 |

表 9：仅设备端 Graph 启动 Stream。

#### 4.2.6.2.1 Fire-and-forget 启动（Fire and Forget Launch）

顾名思义，Fire-and-forget 启动会立即提交到 GPU，并独立于启动它的 Graph 运行。在 Fire-and-forget 场景中，启动 Graph 是父 Graph，被启动的 Graph 是子 Graph。

![Fire-and-forget 启动](/images/chapter-04/fire-and-forget-simple.png)

图 34：Fire-and-forget 启动。

上图可以由以下示例代码生成：

```cpp
__global__ void launchFireAndForgetGraph(cudaGraphExec_t graph) {
    cudaGraphLaunch(graph, cudaStreamGraphFireAndForget);
}

void graphSetup() {
    cudaGraphExec_t gExec1, gExec2;
    cudaGraph_t g1, g2;

    // Create, instantiate, and upload the device graph.
    create_graph(&g2);
    cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
    cudaGraphUpload(gExec2, stream);

    // Create and instantiate the launching graph.
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launchFireAndForgetGraph<<<1, 1, 0, stream>>>(gExec2);
    cudaStreamEndCapture(stream, &g1);
    cudaGraphInstantiate(&gExec1, g1);

    // Launch the host graph, which will in turn launch the device graph.
    cudaGraphLaunch(gExec1, stream);
}
```

一个 Graph 在其执行过程中最多可以总共启动 120 个 Fire-and-forget Graph。同一个父 Graph 的两次启动之间，这个总数会重置。

##### 4.2.6.2.1.1 Graph 执行环境（Graph Execution Environments）

要完整理解设备端同步模型，首先需要理解执行环境的概念。

当从设备启动 Graph 时，它会启动到自己的执行环境中。给定 Graph 的执行环境封装了该 Graph 中的所有工作，以及由其生成的所有 Fire-and-forget 工作。当 Graph 已执行完成且所有生成的子工作也完成时，可以认为该 Graph 已完成。

下图展示了前一节 Fire-and-forget 示例代码产生的执行环境封装：

![带执行环境的 Fire-and-forget 启动](/images/chapter-04/fire-and-forget-environments.png)

图 35：带执行环境的 Fire-and-forget 启动。

这些环境也是分层的，因此一个 Graph 环境可以包含由 Fire-and-forget 启动产生的多层子环境。

![嵌套的 Fire-and-forget 环境](/images/chapter-04/fire-and-forget-nested-environments.png)

图 36：嵌套的 Fire-and-forget 环境。

当从 Host 启动 Graph 时，会存在一个 Stream 环境作为被启动 Graph 执行环境的父环境。Stream 环境封装整体启动过程中生成的所有工作。当整体 Stream 环境被标记为完成时，Stream 启动才算完成（即下游依赖工作此时可以运行）。

![Stream 环境可视化](/images/chapter-04/device-graph-stream-environment.png)

图 37：Stream 环境可视化。

#### 4.2.6.2.2 Tail 启动（Tail Launch）

与 Host 不同，设备端不能通过 `cudaDeviceSynchronize()` 或 `cudaStreamSynchronize()` 等传统方法从 GPU 同步设备 Graph。为了支持串行工作依赖，CUDA 提供了另一种启动模式——Tail 启动，以实现类似功能。

当一个 Graph 的环境被视为完成时，Tail 启动就会执行，即 Graph 及其所有子 Graph 都完成时。一个 Graph 完成后，Tail 启动列表中下一个 Graph 的环境会替换已完成的环境，成为父环境的子环境。与 Fire-and-forget 启动一样，一个 Graph 可以为 Tail 启动排队多个 Graph。

![简单 Tail 启动](/images/chapter-04/tail-launch-simple.png)

图 38：简单 Tail 启动。

上面的执行流可以由以下代码生成：

```cpp
__global__ void launchTailGraph(cudaGraphExec_t graph) {
    cudaGraphLaunch(graph, cudaStreamGraphTailLaunch);
}

void graphSetup() {
    cudaGraphExec_t gExec1, gExec2;
    cudaGraph_t g1, g2;

    // Create, instantiate, and upload the device graph.
    create_graph(&g2);
    cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
    cudaGraphUpload(gExec2, stream);

    // Create and instantiate the launching graph.
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launchTailGraph<<<1, 1, 0, stream>>>(gExec2);
    cudaStreamEndCapture(stream, &g1);
    cudaGraphInstantiate(&gExec1, g1);

    // Launch the host graph, which will in turn launch the device graph.
    cudaGraphLaunch(gExec1, stream);
}
```

给定 Graph 排队的 Tail 启动会按照排队顺序逐个执行。因此，最先入队的 Graph 先运行，之后依次运行其他 Graph。

![Tail 启动排序](/images/chapter-04/tail-launch-ordering-simple.png)

图 39：Tail 启动排序。

由 Tail Graph 排队的 Tail 启动，会先于 Tail 启动列表中此前 Graph 排队的 Tail 启动执行。这些新的 Tail 启动会按入队顺序执行。

![多个 Graph 排队时的 Tail 启动排序](/images/chapter-04/tail-launch-ordering-complex.png)

图 40：多个 Graph 排队时的 Tail 启动排序。

一个 Graph 最多可以有 255 个待处理 Tail 启动。

##### 4.2.6.2.2.1 Tail 自启动（Tail Self-launch）

设备 Graph 可以为自身排队一个 Tail 启动，但给定 Graph 同时只能有一个自身启动处于排队状态。为了查询当前正在运行、从而可以重新启动的设备 Graph，新增了以下设备端函数：

```cpp
cudaGraphExec_t cudaGetCurrentGraphExec();
```

如果当前运行的 Graph 是设备 Graph，该函数返回当前 Graph 的句柄。如果当前执行的 Kernel 不是设备 Graph 中的节点，该函数返回 `NULL`。

下面的示例代码展示了如何使用该函数实现重新启动循环：

```cpp
__device__ int relaunchCount = 0;

__global__ void relaunchSelf() {
    int relaunchMax = 100;

    if (threadIdx.x == 0) {
        if (relaunchCount < relaunchMax) {
            cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
        }

        relaunchCount++;
    }
}
```

#### 4.2.6.2.3 Sibling 启动（Sibling Launch）

Sibling 启动是 Fire-and-forget 启动的一种变体：Graph 不作为启动 Graph 执行环境的子环境启动，而是作为启动 Graph 父环境的子环境启动。Sibling 启动等价于从启动 Graph 的父环境执行 Fire-and-forget 启动。

![简单 Sibling 启动](/images/chapter-04/sibling-launch-simple.png)

图 41：简单 Sibling 启动。

上图可以由以下示例代码生成：

```cpp
__global__ void launchSiblingGraph(cudaGraphExec_t graph) {
    cudaGraphLaunch(graph, cudaStreamGraphFireAndForgetAsSibling);
}

void graphSetup() {
    cudaGraphExec_t gExec1, gExec2;
    cudaGraph_t g1, g2;

    // Create, instantiate, and upload the device graph.
    create_graph(&g2);
    cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
    cudaGraphUpload(gExec2, stream);

    // Create and instantiate the launching graph.
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launchSiblingGraph<<<1, 1, 0, stream>>>(gExec2);
    cudaStreamEndCapture(stream, &g1);
    cudaGraphInstantiate(&gExec1, g1);

    // Launch the host graph, which will in turn launch the device graph.
    cudaGraphLaunch(gExec1, stream);
}
```

由于 Sibling 启动不会启动到启动 Graph 的执行环境中，因此它们不会阻塞由启动 Graph 排队的 Tail 启动。

## 4.2.7 使用 Graph API（Using Graph APIs）

`cudaGraph_t` 对象不是线程安全的。用户有责任确保多个线程不会并发访问同一个 `cudaGraph_t`。

`cudaGraphExec_t` 不能与自身并发运行。一次 `cudaGraphExec_t` 启动会排在同一个可执行图的此前启动之后。

Graph 执行在 Stream 中完成，以便与其他异步工作排序。不过，Stream 仅用于排序；它既不约束 Graph 内部的并行性，也不影响 Graph 节点在哪个位置执行。

参见 [Graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html)。

## 4.2.8 CUDA User Object（CUDA User Objects）

CUDA User Object 可用于帮助管理 CUDA 异步工作所使用资源的生命周期。该功能尤其适用于 [CUDA Graph](#4-2-cuda-图-cuda-graphs) 和 [Stream capture](#4-2-2-1-2-stream-capture)。

许多资源管理方案与 CUDA Graph 不兼容。例如，基于 Event 的资源池，或者同步创建、异步销毁的方案。

```cpp
// Library API with pool allocation
void libraryWork(cudaStream_t stream) {
    auto &resource = pool.claimTemporaryResource();
    resource.waitOnReadyEventInStream(stream);
    launchWork(stream, resource);
    resource.recordReadyEvent(stream);
}
```

```cpp
// Library API with asynchronous resource deletion
void libraryWork(cudaStream_t stream) {
    Resource *resource = new Resource(...);
    launchWork(stream, resource);
    cudaLaunchHostFunc(
        stream,
        [](void *resource) {
            delete static_cast<Resource *>(resource);
        },
        resource,
        0);
    // Error handling considerations not shown
}
```

这些方案在 CUDA Graph 中难以使用，因为资源的指针或句柄不是固定的，需要间接访问或 Graph update，而且每次提交工作都需要同步 CPU 代码。如果这些细节对库调用方隐藏，它们也无法与 Stream capture 配合，因为 capture 期间会使用不允许的 API。可以采用多种解决方案，例如把资源暴露给调用方；CUDA User Object 提供了另一种方法。

CUDA User Object 将用户指定的析构回调与内部引用计数关联起来，类似 C++ 的 `shared_ptr`。引用可以由 CPU 上的用户代码以及 CUDA Graph 持有。注意，与 C++ 智能指针不同，用户持有的引用没有对应对象；用户必须手动跟踪自己持有的引用。一个典型用法是在创建 User Object 后立即将唯一的用户引用转移给 CUDA Graph。

当引用与 CUDA Graph 关联时，CUDA 会自动管理 Graph 操作。克隆的 `cudaGraph_t` 会保留源 `cudaGraph_t` 所拥有的每个引用的副本，且引用数量相同。实例化的 `cudaGraphExec_t` 会保留源 `cudaGraph_t` 中每个引用的副本。如果 `cudaGraphExec_t` 在未同步的情况下销毁，这些引用会一直保留到执行完成。

下面是一个使用示例：

```cpp
cudaGraph_t graph;  // Preexisting graph

Object *object = new Object;  // C++ object with possibly nontrivial destructor
cudaUserObject_t cuObject;
cudaUserObjectCreate(
    &cuObject,
    object,  // Here we use a CUDA-provided template wrapper for this API,
             // which supplies a callback to delete the C++ object pointer
    1,  // Initial refcount
    cudaUserObjectNoDestructorSync  // Acknowledge that the callback cannot be
                                    // waited on via CUDA
);
cudaGraphRetainUserObject(
    graph,
    cuObject,
    1,  // Number of references
    cudaGraphUserObjectMove  // Transfer a reference owned by the caller (do
                             // not modify the total reference count)
);
// No more references owned by this thread; no need to call release API
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);  // Will retain a
                                                               // new reference
cudaGraphDestroy(graph);  // graphExec still owns a reference
cudaGraphLaunch(graphExec, 0);  // Async launch has access to the user objects
cudaGraphExecDestroy(graphExec);  // Launch is not synchronized; the release
                                  // will be deferred if needed
cudaStreamSynchronize(0);  // After the launch is synchronized, the remaining
                           // reference is released and the destructor will
                           // execute. Note this happens asynchronously.
// If the destructor callback had signaled a synchronization object, it would
// be safe to wait on it at this point.
```

子图节点中由 Graph 持有的引用与子图关联，而不是与父图关联。如果子图被更新或删除，引用也会相应变化。如果可执行 Graph 或子图通过 `cudaGraphExecUpdate` 或 `cudaGraphExecChildGraphNodeSetParams` 更新，新源图中的引用会被克隆并替换目标图中的引用。无论哪种情况，如果此前的启动尚未同步，所有将被释放的引用都会保留到这些启动执行完毕。

目前没有通过 CUDA API 等待 User Object 析构函数的机制。用户可以在析构代码中手动向同步对象发送信号。此外，类似 `cudaLaunchHostFunc` 的限制，析构函数中不能调用 CUDA API。这是为了避免阻塞 CUDA 内部共享线程并阻止向前进展。如果依赖是单向的，且执行调用的线程不会阻塞 CUDA 工作向前推进，那么可以向另一个线程发送信号，让它执行 API 调用。

User Object 使用 `cudaUserObjectCreate` 创建；这是浏览相关 API 的良好起点。
