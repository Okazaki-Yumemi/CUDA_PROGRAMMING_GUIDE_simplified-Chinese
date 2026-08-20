---
title: 4.17 扩展 GPU 内存（Extended GPU Memory）
description: 使用 NVLink-C2C 和 EGM 接口访问单节点及多节点系统中的系统内存。
---

# 4.17 扩展 GPU 内存（Extended GPU Memory）

扩展 GPU 内存（Extended GPU Memory，EGM）功能利用高带宽 NVLink-C2C，使 GPU 能够在单节点和多节点系统中高效访问全部系统内存。EGM 面向 CPU-GPU 集成式 NVIDIA 系统，允许分配物理内存，并使配置中的任意 GPU 线程都能访问这部分内存。EGM 确保所有 GPU 都能以 GPU-GPU NVLink 或 NVLink-C2C 的速度访问其资源。

![EGM 示意图](/images/chapter-04/egm-c2c-intro.png)

在这种配置中，内存访问通过本地高带宽 NVLink-C2C 进行。对于远程内存访问，将使用 GPU NVLink，并在某些情况下使用 NVLink-C2C。借助 EGM，GPU 线程可以通过 NVSwitch fabric 访问所有可用内存资源，包括 CPU 连接的内存和 HBM3。

## 4.17.1 前置知识（Preliminaries）

在介绍 EGM 功能的 API 变化之前，先说明当前支持的拓扑、标识符分配、虚拟内存管理的前置条件，以及 EGM 使用的 CUDA 类型。

### 4.17.1.1 EGM 平台：系统拓扑（EGM Platforms: System topology）

目前可以在以下几类平台上启用 EGM：**（1）单节点、单 GPU**：由基于 Arm 的 CPU、CPU 连接的内存和一个 GPU 组成；CPU 与 GPU 之间通过高带宽 C2C（Chip-to-Chip，芯片到芯片）互连。**（2）单节点、多 GPU**：由多个基于 ARM 的 CPU 组成，每个 CPU 都有连接的内存；多个 GPU 通过基于 NVLink 的网络互连。**（3）多节点、多 GPU**：由两个或更多单节点系统组成，每个系统的结构如上面的（1）或（2），这些系统再通过基于 NVLink 的网络连接。

::: note

使用 `cgroups` 限制可用设备会阻止通过 EGM 进行路由，并造成性能问题。请改用 `CUDA_VISIBLE_DEVICES`。

:::

### 4.17.1.2 Socket 标识符：它们是什么，如何访问？（Socket Identifiers: What are they? How to access them?）

NUMA（Non-Uniform Memory Access，非统一内存访问）是多处理器计算机系统使用的一种内存架构：系统内存被划分为多个节点，每个节点都有自己的处理器和内存。在这种系统中，NUMA 将系统划分为多个节点，并为每个节点分配一个唯一标识符（`numaID`）。

EGM 使用由操作系统分配的 NUMA 节点标识符。请注意，该标识符不同于设备的 ordinal，并且与距离最近的主机节点关联。除现有方法外，用户还可以通过调用 `cuDeviceGetAttribute` 并将属性类型设为 `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` 来获取主机节点的标识符（`numaID`）：

```cpp
int numaId;
cuDeviceGetAttribute(&numaId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, deviceOrdinal);
```

### 4.17.1.3 分配器与 EGM 支持（Allocators and EGM support）

将系统内存映射为 EGM 不会导致性能问题。事实上，访问映射为 EGM 的远程 socket 系统内存会更快，因为 EGM 流量保证通过 NVLink 路由。目前，`cuMemCreate` 和 `cudaMemPoolCreate` 分配器在使用适当的位置类型和 NUMA 标识符时受到支持。

### 4.17.1.4 当前 API 的内存管理扩展（Memory management extensions to current APIs）

目前，可以使用虚拟内存（`cuMemCreate`）或流排序内存（`cudaMemPoolCreate`）分配器映射 EGM 内存。用户负责分配物理内存，并将其映射到所有 socket 上的虚拟内存地址空间。

::: note

多节点、多 GPU 平台需要进程间通信。因此建议阅读 [4.15 进程间通信（Interprocess Communication）](./interprocess-communication.html)。

:::

::: note

建议阅读 CUDA Programming Guide 的 [4.16 虚拟内存管理（Virtual Memory Management）](./virtual-memory-management.html) 和 [4.3 流排序内存分配器（Stream-Ordered Memory Allocator）](./stream-ordered-memory-allocator.html)，以便更好地理解相关内容。

:::

API 新增了 CUDA 属性类型，使这些方法可以使用类似 NUMA 节点的标识符了解分配的位置：

| CUDA 类型 | 用于 |
| --- | --- |
| `CU_MEM_LOCATION_TYPE_HOST_NUMA` | `cuMemCreate` 的 `CUmemAllocationProp` |
| `cudaMemLocationTypeHostNuma` | `cudaMemPoolCreate` 的 `cudaMemPoolProps` |

::: note

有关 NUMA 专用 CUDA 类型的更多信息，请参阅 [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html) 和 [CUDA Runtime 数据类型](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html)。

:::

## 4.17.2 使用 EGM 接口（Using the EGM Interface）

### 4.17.2.1 单节点、单 GPU（Single-Node, Single-GPU）

可以使用现有的任意 CUDA 主机分配器以及系统分配的内存，从高带宽 C2C 中受益。对用户而言，本地访问的行为与当前主机分配的行为一致。

::: note

有关内存分配器和页大小的更多信息，请参阅调优指南。

:::

### 4.17.2.2 单节点、多 GPU（Single-Node, Multi-GPU）

在多 GPU 系统中，用户必须提供主机信息以确定内存放置位置。如前所述，表达这类信息的一种自然方式是使用 NUMA 节点 ID，EGM 也采用这种方法。因此，用户可以使用 `cuDeviceGetAttribute` 函数获取距离最近的 NUMA 节点 ID（参见[“Socket 标识符：它们是什么，如何访问？”](#41712-socket-标识符它们是什么如何访问)）。然后，用户可以使用 VMM（Virtual Memory Management，虚拟内存管理）API 或 CUDA Memory Pool 分配和管理 EGM 内存。

#### 4.17.2.2.1 使用 VMM API（Using VMM APIs）

使用虚拟内存管理 API 分配内存的第一步，是创建一个为分配提供后备存储的物理内存块。更多细节请参阅 CUDA Programming Guide 的[“虚拟内存管理”](./virtual-memory-management.html)一节。在 EGM 分配中，用户必须显式提供 `CU_MEM_LOCATION_TYPE_HOST_NUMA` 作为位置类型，并提供 `numaID` 作为位置标识符。此外，EGM 分配必须按照平台适用的粒度对齐。下面的代码片段展示了如何使用 `cuMemCreate` 分配物理内存：

```cpp
CUmemAllocationProp prop{};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
prop.location.id = numaId;
size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop, MEM_ALLOC_GRANULARITY_MINIMUM);
size_t padded_size = ROUND_UP(size, granularity);
CUmemGenericAllocationHandle allocHandle;
cuMemCreate(&allocHandle, padded_size, &prop, 0);
```

分配物理内存后，需要保留地址空间并将它映射到一个指针。这些步骤没有 EGM 专用的变化：

```cpp
CUdeviceptr dptr;
cuMemAddressReserve(&dptr, padded_size, 0, 0, 0);
cuMemMap(dptr, padded_size, 0, allocHandle, 0);
```

最后，用户必须显式保护已映射的虚拟地址范围，否则访问已映射空间会导致崩溃。与内存分配类似，用户必须提供 `CU_MEM_LOCATION_TYPE_HOST_NUMA` 作为位置类型，并提供 `numaId` 作为位置标识符。下面的代码片段为主机节点和 GPU 创建访问描述符，使二者都能对映射内存进行读写：

```cpp
CUmemAccessDesc accessDesc[2]{{}};
accessDesc[0].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
accessDesc[0].location.id = numaId;
accessDesc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
accessDesc[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
accessDesc[1].location.id = currentDev;
accessDesc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
cuMemSetAccess(dptr, size, accessDesc, 2);
```

#### 4.17.2.2.2 使用 CUDA Memory Pool（Using CUDA Memory Pool）

要定义 EGM，用户可以在某个节点上创建内存池，并向对等端授予访问权限。在这种情况下，用户必须显式将 `cudaMemLocationTypeHostNuma` 指定为位置类型，并提供 `numaId` 作为位置标识符。下面的代码片段展示了如何使用 `cudaMemPoolCreate` 创建内存池：

```cpp
cudaSetDevice(homeDevice);
cudaMemPoolProps props{};
props.allocType = cudaMemAllocationTypePinned;
props.location.type = cudaMemLocationTypeHostNuma;
props.location.id = numaId;
cudaMemPoolCreate(&memPool, &props);
```

此外，对于直接连接的对等访问，还可以使用已有的对等访问 API `cudaMemPoolSetAccess`。下面的代码片段展示了如何为 `accessingDevice` 设置访问权限：

```cpp
cudaMemAccessDesc desc{};
desc.flags = cudaMemAccessFlagsProtReadWrite;
desc.location.type = cudaMemLocationTypeDevice;
desc.location.id = accessingDevice;
cudaMemPoolSetAccess(memPool, &desc, 1);
```

创建内存池并授予访问权限后，用户可以将创建的内存池设置给 `residentDevice`，并使用 `cudaMallocAsync` 开始分配内存：

```cpp
cudaDeviceSetMemPool(residentDevice, memPool);
cudaMallocAsync(&ptr, size, memPool, stream);
```

::: note

EGM 使用 2 MB 页进行映射。因此，访问非常大的分配时，用户可能会遇到更多 TLB miss。

:::

### 4.17.2.3 多节点、多 GPU（Multi-Node, Multi-GPU）

除内存分配之外，远程对等访问没有 EGM 专用修改，而是遵循 CUDA 进程间通信（IPC）协议。有关 IPC 的更多细节，请参阅 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#allocating-physical-memory)。

用户应使用 `cuMemCreate` 分配内存，并再次显式提供 `CU_MEM_LOCATION_TYPE_HOST_NUMA` 作为位置类型、`numaID` 作为位置标识符。此外，应将 `CU_MEM_HANDLE_TYPE_FABRIC` 指定为请求的句柄类型。下面的代码片段展示了如何在节点 A 上分配物理内存：

```cpp
CUmemAllocationProp prop{};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
prop.location.id = numaId;
size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop,
                              MEM_ALLOC_GRANULARITY_MINIMUM);
size_t padded_size = ROUND_UP(size, granularity);
size_t page_size = ...;
assert(padded_size % page_size == 0);
CUmemGenericAllocationHandle allocHandle;
cuMemCreate(&allocHandle, padded_size, &prop, 0);
```

使用 `cuMemCreate` 创建分配句柄后，用户可以调用 `cuMemExportToShareableHandle` 将该句柄导出到另一个节点 B：

```cpp
cuMemExportToShareableHandle(&fabricHandle, allocHandle,
                             CU_MEM_HANDLE_TYPE_FABRIC, 0);
// 此时应通过 TCP/IP 将 fabricHandle 发送到节点 B。
```

在节点 B 上，可以使用 `cuMemImportFromShareableHandle` 导入句柄，并像处理其他 fabric 句柄一样处理它：

```cpp
// 此时应已通过 TCP/IP 从节点 A 接收到 fabricHandle。
CUmemGenericAllocationHandle allocHandle;
cuMemImportFromShareableHandle(&allocHandle, &fabricHandle,
                               CU_MEM_HANDLE_TYPE_FABRIC);
```

在节点 B 导入句柄后，用户可以按常规方式保留地址空间并在本地映射它：

```cpp
size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop,
                              MEM_ALLOC_GRANULARITY_MINIMUM);
size_t padded_size = ROUND_UP(size, granularity);
size_t page_size = ...;
assert(padded_size % page_size == 0);
CUdeviceptr dptr;
cuMemAddressReserve(&dptr, padded_size, 0, 0, 0);
cuMemMap(dptr, padded_size, 0, allocHandle, 0);
```

最后一步是为节点 B 上的每个本地 GPU 授予适当的访问权限。下面的代码片段为 8 个本地 GPU 授予读写权限：

```cpp
// 让本地的全部 8 个 GPU 访问位于节点 A、已导出的 EGM 内存。
CUmemAccessDesc accessDesc[8];
for (int i = 0; i < 8; i++) {
   accessDesc[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
   accessDesc[i].location.id = i;
   accessDesc[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}
cuMemSetAccess(dptr, size, accessDesc, 8);
```
