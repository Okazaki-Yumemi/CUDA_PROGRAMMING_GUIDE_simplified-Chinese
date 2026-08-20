---
title: 4.3 按 Stream 顺序的内存分配器
description: 按 Stream 顺序的内存分配器、内存池、IPC 与调优的完整中文翻译
---

# 4.3 按 Stream 顺序的内存分配器（Stream-Ordered Memory Allocator）

## 4.3.1 介绍（Introduction）

使用 `cudaMalloc` 和 `cudaFree` 管理内存分配，会导致 GPU 在所有正在执行的 CUDA stream 之间进行同步。按 stream 顺序的内存分配器允许应用将内存分配和释放与 CUDA stream 中启动的其他工作（例如 kernel 启动和异步拷贝）排序。该分配器利用 stream 顺序语义复用内存分配，从而改善应用的内存使用。它还允许应用控制分配器的内存缓存行为。设置合适的释放阈值后，当应用表示愿意接受更大的内存占用时，缓存行为可以避免向操作系统发出代价高昂的调用。该分配器还支持在进程之间轻松、安全地共享分配。

按 Stream 顺序的内存分配器具有以下作用：

- 减少对自定义内存管理抽象的需求，使需要高性能自定义内存管理的应用更容易实现。
- 允许多个库共享由驱动管理的公共内存池，从而减少额外的内存消耗。
- 允许驱动根据对分配器和其他 stream 管理 API 的了解执行优化。

> **注意**
>
> 从 CUDA 11.3 开始，Nsight Compute 和新一代 CUDA 调试器已经能够识别该分配器。

## 4.3.2 内存管理（Memory Management）

`cudaMallocAsync` 和 `cudaFreeAsync` 是实现按 stream 顺序内存管理的 API。`cudaMallocAsync` 返回一个分配，`cudaFreeAsync` 释放一个分配。两个 API 都接受 stream 参数，用来定义分配何时开始可用以及何时停止可用。这些函数可以将内存操作绑定到特定 CUDA stream，使操作能够在不阻塞主机或其他 stream 的情况下进行。通过避免 `cudaMalloc` 和 `cudaFree` 可能产生的昂贵同步，可以提升应用性能。

这些 API 还可以通过内存池进一步优化性能。内存池管理并复用大块内存，使分配和释放更高效。在频繁分配内存的场景中，内存池可以减少开销、防止碎片并改善性能。

### 4.3.2.1 分配内存（Allocating Memory）

`cudaMallocAsync` 函数在 GPU 上触发与特定 CUDA stream 关联的异步内存分配。它允许分配在不妨碍主机或其他 stream 的情况下进行，从而不需要昂贵的同步。

> **注意**
>
> `cudaMallocAsync` 在确定分配所在位置时会忽略当前 device/context。相反，它会根据指定的内存池或传入的 stream 确定适当的设备。

下面的代码展示基本使用模式：在同一个 stream 中分配、使用内存，然后释放回该 stream。

```cpp
void *ptr;
size_t size = 512;
cudaMallocAsync(&ptr, size, cudaStreamPerThread);
// 使用该分配执行工作
kernel<<<..., cudaStreamPerThread>>>(ptr, ...);
// 可以指定异步释放，而不必同步 CPU 和 GPU
cudaFreeAsync(ptr, cudaStreamPerThread);
```

> **注意**
>
> 当从创建该分配的 stream 以外的 stream 访问它时，用户必须保证访问发生在分配操作之后，否则行为未定义。

### 4.3.2.2 释放内存（Freeing Memory）

`cudaFreeAsync()` 以按 stream 顺序的方式异步释放设备内存。这意味着释放操作被分配到特定 CUDA stream，并且不会阻塞主机或其他 stream。

用户必须保证释放操作发生在分配操作以及对该分配的所有使用之后。释放操作开始后继续使用该分配会导致未定义行为。

应使用 event 和/或 stream 同步操作，保证来自其他 stream 的访问在释放开始前已经完成，示例如下：

```cpp
cudaMallocAsync(&ptr, size, stream1);
cudaEventRecord(event1, stream1);
// stream2 在访问前必须等待分配就绪
cudaStreamWaitEvent(stream2, event1);
kernel<<<..., stream2>>>(ptr, ...);
cudaEventRecord(event2, stream2);
// stream3 在释放分配前必须等待 stream2 完成访问
cudaStreamWaitEvent(stream3, event2);
cudaFreeAsync(ptr, stream3);
```

使用 `cudaMalloc()` 分配的内存也可以用 `cudaFreeAsync()` 释放。同样，所有内存访问都必须在释放开始前完成。

```cpp
cudaMalloc(&ptr, size);
kernel<<<..., stream>>>(ptr, ...);
cudaFreeAsync(ptr, stream);
```

同理，使用 `cudaMallocAsync` 分配的内存也可以用 `cudaFree()` 释放。通过 `cudaFree()` API 释放此类分配时，驱动假设对该分配的所有访问都已完成，不再执行额外同步。用户可以使用 `cudaStreamQuery`、`cudaStreamSynchronize`、`cudaEventQuery`、`cudaEventSynchronize` 或 `cudaDeviceSynchronize`，确保相应异步工作已经完成，并确保 GPU 不会继续访问该分配。

```cpp
cudaMallocAsync(&ptr, size,stream);
kernel<<<..., stream>>>(ptr, ...);
// 需要同步，以免过早释放内存
cudaStreamSynchronize(stream);
cudaFree(ptr);
```

## 4.3.3 内存池（Memory Pools）

内存池封装虚拟地址和物理内存资源，并依据池的属性和性质对它们进行分配与管理。内存池最主要的特征是它管理的内存种类和位置。

所有 `cudaMallocAsync` 调用都使用内存池中的资源。如果没有指定内存池，`cudaMallocAsync` 会使用所传入 stream 所属设备的当前内存池。可以用 `cudaDeviceSetMempool` 设置设备的当前内存池，用 `cudaDeviceGetMempool` 查询它。每个设备都有一个默认内存池；如果没有调用 `cudaDeviceSetMempool`，该默认池处于活动状态。

`cudaMallocFromPoolAsync` 以及 `cudaMallocAsync` 的 C++ 重载允许用户指定某次分配使用的池，而不用把该池设置为当前池。`cudaDeviceGetDefaultMempool` 和 `cudaMemPoolCreate` API 返回内存池句柄；`cudaMemPoolSetAttribute` 和 `cudaMemPoolGetAttribute` 控制内存池属性。

> **注意**
>
> 设备的当前内存池属于该设备本地。因此，不指定内存池的分配总是产生在 stream 所属设备本地的分配。

### 4.3.3.1 默认/隐式池（Default/Implicit Pools）

调用 `cudaDeviceGetDefaultMempool` 可以取得设备的默认内存池。从设备默认内存池分配的内存是位于该设备上的不可迁移设备分配，并且始终可从该设备访问。可以用 `cudaMemPoolSetAccess` 修改默认池的可访问性，用 `cudaMemPoolGetAccess` 查询。由于默认池不需要显式创建，因此有时也称为隐式池。设备默认内存池不支持 IPC。

### 4.3.3.2 显式池（Explicit Pools）

`cudaMemPoolCreate` 创建显式池。它允许应用为分配请求默认/隐式池无法提供的属性，例如 IPC 能力、最大池大小，以及在受支持平台上让分配驻留于特定 CPU NUMA 节点等。

```cpp
// 创建一个类似设备 0 隐式池的内存池
int device = 0;
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = device;
poolProps.location.type = cudaMemLocationTypeDevice;

cudaMemPoolCreate(&memPool, &poolProps));
```

下面的代码展示如何在有效的 CPU NUMA 节点上创建具有 IPC 能力的内存池。

```cpp
// 创建驻留于支持 IPC 共享（通过文件描述符）的 CPU NUMA 节点上的池
int cpu_numa_id = 0;
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = cpu_numa_id;
poolProps.location.type = cudaMemLocationTypeHostNuma;
poolProps.handleType = cudaMemHandleTypePosixFileDescriptor;

cudaMemPoolCreate(&ipcMemPool, &poolProps));
```

### 4.3.3.3 多 GPU 支持下的设备可访问性（Device Accessibility for Multi-GPU Support）

与虚拟内存管理 API 控制的分配可访问性类似，内存池分配的可访问性不遵循 `cudaDeviceEnablePeerAccess` 或 `cuCtxEnablePeerAccess`。对于内存池，`cudaMemPoolSetAccess` API 修改哪些设备可以访问池中的分配。默认情况下，分配只可从其所在设备访问，且这种访问不能撤销。若要启用其他设备的访问，访问设备必须与内存池所在设备具备 peer 能力，可用 `cudaDeviceCanAccessPeer` 验证。如果未检查 peer 能力，设置访问可能失败并返回 `cudaErrorInvalidDevice`。不过，如果池中尚未创建分配，即使设备不具备 peer 能力，`cudaMemPoolSetAccess` 调用也可能成功；此时下一次从该池分配时会失败。

需要注意，`cudaMemPoolSetAccess` 会影响内存池中的所有分配，而不只是未来的分配。同样，`cudaMemPoolGetAccess` 报告的可访问性也适用于池中的所有分配，而不只是未来的分配。不建议频繁更改某个 GPU 对某池的可访问性设置：一旦某个池对某 GPU 可访问，在该池的生命周期内就应保持可访问。

```cpp
// 展示 cudaMemPoolSetAccess 的用法
cudaError_t setAccessOnDevice(cudaMemPool_t memPool, int residentDevice,
              int accessingDevice) {
    cudaMemAccessDesc accessDesc = {};
    accessDesc.location.type = cudaMemLocationTypeDevice;
    accessDesc.location.id = accessingDevice;
    accessDesc.flags = cudaMemAccessFlagsProtReadWrite;

    int canAccess = 0;
    cudaError_t error = cudaDeviceCanAccessPeer(&canAccess, accessingDevice,
              residentDevice);
    if (error != cudaSuccess) {
        return error;
    } else if (canAccess == 0) {
        return cudaErrorPeerAccessUnsupported;
    }

    // 使地址可访问
    return cudaMemPoolSetAccess(memPool, &accessDesc, 1);
}
```

### 4.3.3.4 为 IPC 启用内存池（Enabling Memory Pools for IPC）

可以为进程间通信（IPC）启用内存池，以便在进程之间轻松、高效且安全地共享 GPU 内存。CUDA 的 IPC 内存池提供与 CUDA 虚拟内存管理 API 相同的安全性收益。

使用内存池在进程间共享内存有两个步骤：进程首先共享对内存池的访问权限，然后共享该池中的特定分配。第一步建立并强制实施安全性；第二步协调每个进程使用的虚拟地址，以及导入进程中映射何时必须有效。

#### 4.3.3.4.1 创建和共享 IPC 内存池（Creating and Sharing IPC Memory Pools）

共享对内存池的访问包括：使用 `cudaMemPoolExportToShareableHandle()` 获取池的操作系统原生句柄；使用操作系统原生 IPC 机制将句柄传给导入进程；然后通过 `cudaMemPoolImportFromShareableHandle()` API 创建导入的内存池。要使 `cudaMemPoolExportToShareableHandle` 成功，内存池必须在池属性结构中指定了所请求的句柄类型。

可以参考 [CUDA Samples 的 streamOrderedAllocationIPC 示例](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/streamOrderedAllocationIPC)，了解在进程间传输操作系统原生句柄的 IPC 机制。其余流程如下。

```cpp
// 导出进程
// 在设备 0 上创建可导出的、具有 IPC 能力的池
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = 0;
poolProps.location.type = cudaMemLocationTypeDevice;

// 将 handleTypes 设置为非零值会使池可导出（具备 IPC 能力）
poolProps.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

cudaMemPoolCreate(&memPool, &poolProps));

// 基于 FD 的句柄是整数类型
int fdHandle = 0;

// 获取池的操作系统原生句柄。
// 注意这里传入的是句柄内存的指针。
cudaMemPoolExportToShareableHandle(&fdHandle,
             memPool,
             CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
             0);

// 必须使用适当的操作系统专用 API 将句柄发送给导入进程。
```

```cpp
// 导入进程
int fdHandle;
// 需要使用适当的操作系统专用 API 从导出进程取得句柄。
// 根据可共享句柄创建导入池。
// 注意这里按值传入句柄。
cudaMemPoolImportFromShareableHandle(&importedMemPool,
          (void*)fdHandle,
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
          0);
```

#### 4.3.3.4.2 在导入进程中设置访问权限（Set Access in the Importing Process）

导入的内存池初始时只能从其驻留设备访问。导入的内存池不会继承导出进程设置的任何可访问性。导入进程需要对计划访问该内存的每个 GPU 调用 `cudaMemPoolSetAccess` 来启用访问。

如果导入的内存池属于导入进程不可见的设备，用户必须使用 `cudaMemPoolSetAccess` API，从实际使用这些分配的 GPU 启用访问。请参见[多 GPU 支持下的设备可访问性](#4333-多-gpu-支持下的设备可访问性-device-accessibility-for-multi-gpu-support)。

#### 4.3.3.4.3 从导出池创建和共享分配（Creating and Sharing Allocations from an Exported Pool）

共享池后，导出进程中使用该池的 `cudaMallocAsync()` 创建的分配，可以与已导入该池的进程共享。由于池级别已经建立并验证了安全策略，操作系统不需要再为特定池分配维护额外的安全记录。换言之，导入池分配所需的不透明 `cudaMemPoolPtrExportData` 可以通过任意机制发送给导入进程。

虽然导出和导入分配时不必以任何方式同步分配所在的 stream，但导入进程访问该分配时必须遵守与导出进程相同的规则。具体来说，访问必须发生在分配操作于分配 stream 中执行之后。以下代码通过 IPC event 保证导入进程不会在分配就绪前访问它，展示 `cudaMemPoolExportPointer()` 和 `cudaMemPoolImportPointer()` 如何共享分配。

```cpp
// 在导出进程中准备分配
cudaMemPoolPtrExportData exportData;
cudaEvent_t readyIpcEvent;
cudaIpcEventHandle_t readyIpcEventHandle;

// 用于进程间协调的 IPC event
// cudaEventInterprocess 标志使 event 成为 IPC event
// 出于性能原因设置 cudaEventDisableTiming
cudaEventCreate(&readyIpcEvent, cudaEventDisableTiming | cudaEventInterprocess)

// 从导出内存池分配
cudaMallocAsync(&ptr, size,exportMemPool, stream);

// 用于共享分配就绪状态的 event
cudaEventRecord(readyIpcEvent, stream);
cudaMemPoolExportPointer(&exportData, ptr);
cudaIpcGetEventHandle(&readyIpcEventHandle, readyIpcEvent);

// 使用任意机制将 IPC event 和指针导出数据共享给导入进程。
// 这里将数据复制到共享内存
shmem->ptrData = exportData;
shmem->readyIpcEventHandle = readyIpcEventHandle;
// 通知消费者数据已经就绪
```

```cpp
// 导入分配
cudaMemPoolPtrExportData *importData = &shmem->prtData;
cudaEvent_t readyIpcEvent;
cudaIpcEventHandle_t *readyIpcEventHandle = &shmem->readyIpcEventHandle;

// 需要使用任意机制从导出进程取得 IPC event 句柄和导出数据。
// 这里使用共享内存，只需要同步以确保共享内存已经填充。
cudaIpcOpenEventHandle(&readyIpcEvent, readyIpcEventHandle);

// 导入分配。该操作不会阻塞等待分配就绪。
cudaMemPoolImportPointer(&ptr, importedMemPool, importData);

// 在导入进程使用分配之前，等待分配 stream 中的先前操作完成。
cudaStreamWaitEvent(stream, readyIpcEvent);
kernel<<<..., stream>>>(ptr, ...);
```

释放分配时，必须先在导入进程中释放，再在导出进程中释放。下面的代码使用 CUDA IPC event，在两个进程的 `cudaFreeAsync` 操作之间提供所需同步。导入进程中的释放操作显然会限制该进程对分配的访问。需要注意，在两个进程中都可以用 `cudaFree` 释放分配，也可以使用其他 stream 同步 API 代替 CUDA IPC event。

```cpp
// 导入进程必须先于导出进程释放
kernel<<<..., stream>>>(ptr, ...);

// 导入进程中的最后一次访问
cudaFreeAsync(ptr, stream);

// 导入进程释放后，不允许再从导入进程访问
cudaIpcEventRecord(finishedIpcEvent, stream);
```

```cpp
// 导出进程
// 导出进程需要将自己的释放与导入进程的 stream 顺序释放协调起来。
cudaStreamWaitEvent(stream, finishedIpcEvent);
kernel<<<..., stream>>>(ptrInExportingProcess, ...);

// 导入进程的释放不会阻止导出进程使用该分配。
cudFreeAsync(ptrInExportingProcess,stream);
```

#### 4.3.3.4.4 IPC 导出池的限制（IPC Export Pool Limitations）

当前 IPC 池不支持将物理块释放回操作系统。因此，`cudaMemPoolTrimTo` API 不起作用，`cudaMemPoolAttrReleaseThreshold` 也会被有效忽略。该行为由驱动而不是 runtime 控制，未来的驱动更新可能会改变它。

#### 4.3.3.4.5 IPC 导入池的限制（IPC Import Pool Limitations）

不允许从导入池分配；具体来说，导入池不能被设置为当前池，也不能用于 `cudaMallocFromPoolAsync` API。因此，分配复用策略属性对这些池没有意义。

IPC 导入池与 IPC 导出池一样，目前不支持将物理块释放回操作系统。

资源使用统计属性的查询只反映导入到当前进程的分配及其关联的物理内存。

## 4.3.4 最佳实践与调优（Best Practices and Tuning）

### 4.3.4.1 查询支持情况（Query for Support）

应用可以调用 `cudaDeviceGetAttribute()`，并传入设备属性 `cudaDevAttrMemoryPoolsSupported`，判断设备是否支持按 stream 顺序的内存分配器。

可以使用设备属性 `cudaDevAttrMemoryPoolSupportedHandleTypes` 查询 IPC 内存池支持情况。该属性在 CUDA 11.3 中加入；查询该属性时，旧驱动会返回 `cudaErrorInvalidValue`。

```cpp
int driverVersion = 0;
int deviceSupportsMemoryPools = 0;
int poolSupportedHandleTypes = 0;
cudaDriverGetVersion(&driverVersion);
if (driverVersion >= 11020) {
    cudaDeviceGetAttribute(&deviceSupportsMemoryPools,
                           cudaDevAttrMemoryPoolsSupported, device);
}
if (deviceSupportsMemoryPools != 0) {
    // `device` 支持按 Stream 顺序的内存分配器
}

if (driverVersion >= 11030) {
    cudaDeviceGetAttribute(&poolSupportedHandleTypes,
              cudaDevAttrMemoryPoolSupportedHandleTypes, device);
}
if (poolSupportedHandleTypes & cudaMemHandleTypePosixFileDescriptor) {
   // 指定设备上的池可以使用基于 POSIX 文件描述符的 IPC 创建
}
```

在查询前检查驱动版本，可以避免在尚未定义该属性的驱动上触发 `cudaErrorInvalidValue`。也可以使用 `cudaGetLastError` 清除错误，而不必规避该错误。

### 4.3.4.2 物理页缓存行为（Physical Page Caching Behavior）

默认情况下，分配器会尝试使池占用的物理内存最少。为减少分配和释放物理内存时调用操作系统的次数，应用必须为每个池配置内存占用上限；可以使用释放阈值属性 `cudaMemPoolAttrReleaseThreshold` 完成配置。

释放阈值是池在尝试将内存释放回操作系统前应保留的内存量（以字节为单位）。当内存池持有的内存超过释放阈值时，分配器会在下一次 stream、event 或 device 同步调用时尝试将内存释放回操作系统。将释放阈值设为 `UINT64_MAX`，可以阻止驱动在每次同步后尝试收缩池。

```cpp
Cuuint64_t setVal = UINT64_MAX;
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);
```

将 `cudaMemPoolAttrReleaseThreshold` 设置得足够大、实际上禁用内存池收缩的应用，可能希望显式收缩内存池占用。`cudaMemPoolTrimTo` 允许应用完成此操作。收缩内存池占用时，`minBytesToKeep` 参数允许应用保留指定数量的内存，例如应用预计在后续执行阶段需要的内存量。

```cpp
Cuuint64_t setVal = UINT64_MAX;
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);

// 应用阶段：需要从按 Stream 顺序的分配器获得大量内存
for (i=0; i<10; i++) {
    for (j=0; j<10; j++) {
        cudaMallocAsync(&ptrs[j],size[j], stream);
    }
    kernel<<<...,stream>>>(ptrs,...);
    for (j=0; j<10; j++) {
        cudaFreeAsync(ptrs[j], stream);
    }
}

// 下一阶段不需要这么多内存。
// 同步，使 trim 操作知道这些分配已不再使用。
cudaStreamSynchronize(stream);
cudaMemPoolTrimTo(mempool, 0);

// 其他进程/分配机制现在可以使用 trim 操作释放的物理内存。
```

### 4.3.4.3 资源使用统计（Resource Usage Statistics）

查询池的 `cudaMemPoolAttrReservedMemCurrent` 属性，会报告池当前消耗的 GPU 物理内存总量。查询池的 `cudaMemPoolAttrUsedMemCurrent` 属性，会返回从池中分配且当前不可复用的所有内存的总大小。

`cudaMemPoolAttr*MemHigh` 属性是水位线，记录自上次重置以来相应 `cudaMemPoolAttr*MemCurrent` 属性达到的最大值。使用 `cudaMemPoolSetAttribute` API，可以将水位线重置为当前值。

```cpp
// 批量获取使用统计的辅助函数示例
struct usageStatistics {
    cuuint64_t reserved;
    cuuint64_t reservedHigh;
    cuuint64_t used;
    cuuint64_t usedHigh;
};

void getUsageStatistics(cudaMemoryPool_t memPool, struct usageStatistics *statistics)
{
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, statistics->reserved);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, statistics->reservedHigh);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, statistics->used);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, statistics->usedHigh);
}

// 重置水位线后，它们会取当前值。
void resetStatistics(cudaMemoryPool_t memPool)
{
    cuuint64_t value = 0;
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, &value);
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, &value);
}
```

### 4.3.4.4 内存复用策略（Memory Reuse Policies）

为了满足分配请求，驱动会在尝试从操作系统分配更多内存之前，先尝试复用之前通过 `cudaFreeAsync()` 释放的内存。例如，在一个 stream 中释放的内存可以立即用于同一 stream 上后续的分配请求。当一个 stream 与 CPU 同步后，该 stream 之前释放的内存就可以用于任何 stream 上的分配。复用策略同时适用于默认内存池和显式内存池。

按 stream 顺序的分配器提供了几个可控的分配策略。池属性 `cudaMemPoolReuseFollowEventDependencies`、`cudaMemPoolReuseAllowOpportunistic` 和 `cudaMemPoolReuseAllowInternalDependencies` 控制这些策略，具体如下。可以通过调用 `cudaMemPoolSetAttribute` 启用或禁用它们。升级到更新的 CUDA 驱动后，复用策略枚举可能发生变化、增强、扩展和/或重新排序。

#### 4.3.4.4.1 `cudaMemPoolReuseFollowEventDependencies`

在分配更多 GPU 物理内存之前，分配器会检查由 CUDA event 建立的依赖信息，并尝试使用在另一个 stream 中释放的内存。

```cpp
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);
cudaEventRecord(event,originalStream);

// 等待捕获了另一个 stream 中释放操作的 event，
// 可以让分配器在另一个 stream 的新分配请求中复用内存，
// 前提是启用了 cudaMemPoolReuseFollowEventDependencies。
cudaStreamWaitEvent(otherStream, event);
cudaMallocAsync(&ptr2, size, otherStream);
```

#### 4.3.4.4.2 `cudaMemPoolReuseAllowOpportunistic`

启用 `cudaMemPoolReuseAllowOpportunistic` 策略后，分配器会检查已释放的分配，判断释放操作的 stream 顺序语义是否已经满足，例如该 stream 是否已经越过释放操作所指示的执行点。禁用该策略后，分配器仍会复用 stream 与 CPU 同步后变得可用的内存。禁用该策略不会阻止 `cudaMemPoolReuseFollowEventDependencies` 生效。

```cpp
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);

// 经过一段时间后，kernel 完成运行
wait(10);

// 启用 cudaMemPoolReuseAllowOpportunistic 时，
// 分配器可以根据 originalStream 的进度使用之前的分配满足请求。
cudaMallocAsync(&ptr2, size, otherStream);
```

#### 4.3.4.4.3 `cudaMemPoolReuseAllowInternalDependencies`

如果无法从操作系统分配并映射更多物理内存，驱动会查找可用性依赖于另一个 stream 尚未完成的进度的内存。如果找到这样的内存，驱动会把所需依赖插入分配 stream，并复用该内存。

```cpp
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);

// 启用 cudaMemPoolReuseAllowInternalDependencies，且驱动无法分配更多物理内存时，
// 驱动可能等效于在分配 stream 中执行 cudaStreamWaitEvent，
// 确保 futureStream 中的未来工作发生在 originalStream 中
// 仍可能访问原始分配的工作之后。
cudaMallocAsync(&ptr2, size, otherStream);
```

#### 4.3.4.4.4 禁用复用策略（Disabling Reuse Policies）

虽然可控复用策略可以改善内存复用，但用户可能希望禁用它们。允许机会性复用（例如 `cudaMemPoolReuseAllowOpportunistic`）会根据 CPU 和 GPU 执行的交错方式，使不同运行之间的分配模式产生差异。内部依赖插入（例如 `cudaMemPoolReuseAllowInternalDependencies`）可能以用户没有预期的、潜在非确定性的方式串行化工作，而用户可能更希望在分配失败时显式同步 event 或 stream。

### 4.3.4.5 同步 API 的行为（Synchronization API Actions）

分配器属于 CUDA 驱动所带来的优化之一，是它与同步 API 的集成。当用户请求 CUDA 驱动执行同步时，驱动会等待异步工作完成。返回前，驱动会判断哪些释放操作已由该同步保证完成。对应的分配无论指定了什么 stream、无论复用策略是否禁用，都可以重新用于分配。驱动还会在这里检查 `cudaMemPoolAttrReleaseThreshold`，并释放能够释放的多余物理内存。

## 4.3.5 补充说明（Addendums）

### 4.3.5.1 `cudaMemcpyAsync` 的当前 Context/Device 敏感性（cudaMemcpyAsync Current Context/Device Sensitivity）

在当前 CUDA 驱动中，任何涉及 `cudaMallocAsync` 内存的异步 `memcpy`，都应使用指定 stream 所属的 context 作为调用线程的当前 context。`cudaMemcpyPeerAsync` 不需要这样做，因为该 API 会引用 API 参数中指定的设备 primary context，而不是当前 context。

### 4.3.5.2 `cudaPointerGetAttributes` 查询（cudaPointerGetAttributes Query）

在对某个分配调用 `cudaFreeAsync` 后，再对该分配调用 `cudaPointerGetAttributes` 会导致未定义行为。具体来说，即使该分配仍可从某个 stream 访问，行为仍然未定义。

### 4.3.5.3 `cudaGraphAddMemsetNode`

`cudaGraphAddMemsetNode` 不能用于按 stream 顺序的分配器分配的内存。不过，可以对这些分配执行 memset 的 stream capture。

### 4.3.5.4 指针属性（Pointer Attributes）

`cudaPointerGetAttributes` 查询适用于按 stream 顺序的分配。由于这类分配不与 context 关联，查询 `CU_POINTER_ATTRIBUTE_CONTEXT` 会成功，但 `*data` 返回 NULL。可以使用 `CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL` 属性确定分配的位置；在选择 context、使用 `cudaMemcpyPeerAsync` 执行 p2h2p 拷贝时，这很有用。属性 `CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE` 在 CUDA 11.3 中加入，可用于调试，也可在执行 IPC 前确认分配来自哪个池。

### 4.3.5.5 CPU 虚拟内存（CPU Virtual Memory）

使用 CUDA 按 stream 顺序的内存分配器 API 时，避免使用 `ulimit -v` 设置 VRAM 限制，因为该配置不受支持。
