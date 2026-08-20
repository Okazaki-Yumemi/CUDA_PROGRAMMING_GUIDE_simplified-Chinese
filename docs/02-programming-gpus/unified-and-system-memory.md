---
title: 2.6 统一内存与系统内存
description: 统一虚拟地址空间、统一内存范式、HMM、ATS、页锁定与映射内存
---

# 2.6 统一内存与系统内存（Unified and System Memory）

> 本页按 NVIDIA CUDA Programming Guide Release 13.3 的官方网页逐段翻译。原文页面：[2.6. Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)；官方页面标注的更新时间为 2026-05-27。

异构系统包含多个可存储数据的物理内存。host CPU 有与之连接的 DRAM，系统中的每个 GPU 也有自己连接的 DRAM。当数据驻留在访问它的处理器所对应的内存中时，性能最佳。CUDA 提供 API[显式管理内存放置](./intro-to-cuda-cpp.html)，但这会使代码冗长，并使软件设计复杂化。CUDA 还提供了一些功能和能力，用于简化数据在不同物理内存之间的分配、放置和迁移。

本章介绍并解释这些功能，以及它们对应用开发者在功能和性能方面的含义。统一内存有多种表现形式，取决于操作系统、驱动版本和所使用的 GPU。本章会说明如何确定适用哪一种统一内存范式，以及统一内存的功能在各范式下如何表现。后面的[统一内存专题](../04-cuda-features/unified-memory.html)会更详细地解释统一内存。

本章会定义和解释以下概念：

- [统一虚拟地址空间](#261-统一虚拟地址空间-unified-virtual-address-space)：CPU 内存和每个 GPU 的内存在同一个虚拟地址空间中拥有不同的地址范围；
- [统一内存](#262-统一内存-unified-memory)：CUDA 的一项功能，允许 managed memory 在 CPU 与 GPU 之间自动迁移；
  - [有限统一内存](#2623-有限统一内存支持-limited-unified-memory-support)：具有一些限制的统一内存范式；
  - [完整统一内存](#2622-完整统一内存功能支持-full-unified-memory-feature-support)：完整支持统一内存功能；
  - [硬件一致性的完整统一内存](#26221-具有硬件一致性的完整统一内存-full-unified-memory-with-hardware-coherency)：利用硬件能力支持完整统一内存；
  - [统一内存提示](#2624-内存建议与预取-memory-advise-and-prefetch)：用于指导特定分配的统一内存行为的 API；
- [页锁定 Host 内存](#263-页锁定-host-内存-page-locked-host-memory)：不可分页的系统内存，一些 CUDA 操作必须使用它；
  - [映射内存](#2631-映射内存-mapped-memory)：一种不同于统一内存的机制，允许 kernel 直接访问 host memory。

此外，本章还引入讨论统一内存和系统内存时使用的术语：

- [异构内存管理（HMM）](#26222-hmm具有软件一致性的完整统一内存-hmm-full-unified-memory-with-software-coherency)：Linux kernel 的一项功能，为完整统一内存提供软件一致性；
- [地址转换服务（ATS）](#26221-具有硬件一致性的完整统一内存-full-unified-memory-with-hardware-coherency)：当 GPU 通过 NVLink Chip-to-Chip（C2C）互连连接到 CPU 时可用的硬件功能，为完整统一内存提供硬件一致性。

## 2.6.1 统一虚拟地址空间（Unified Virtual Address Space）

在单个 OS 进程中，系统中所有 host memory 和所有 GPU 上的 global memory 共用一个虚拟地址空间。host 和所有 device 上的内存分配都位于这个地址空间内。无论分配使用 CUDA API（例如 `cudaMalloc`、`cudaMallocHost`），还是使用系统分配 API（例如 `new`、`malloc`、`mmap`），都遵循这一点。CPU 和每张 GPU 在统一虚拟地址空间中都有唯一的地址范围。

这意味着：

- 可以使用 `cudaPointerGetAttributes()`，根据指针的值确定一块内存的位置（CPU 内存，或它所在的 GPU 内存）；
- `cudaMemcpy*()` 的 `cudaMemcpyKind` 参数可以设置为 `cudaMemcpyDefault`，让 CUDA 根据指针自动确定复制类型。

## 2.6.2 统一内存（Unified Memory）

*统一内存（unified memory）* 是 CUDA 的一种内存功能，允许称为 *managed memory* 的内存分配由 CPU 或 GPU 上运行的代码访问。[CUDA C++ 入门](./intro-to-cuda-cpp.html)已经展示过统一内存。所有 CUDA 支持的系统都提供统一内存。

在一些系统上，managed memory 必须显式分配。在 CUDA 中，可以通过几种方式显式分配：

- CUDA API `cudaMallocManaged`；
- CUDA API `cudaMallocFromPoolAsync`，并使用 `allocType` 为 `cudaMemAllocationTypeManaged` 创建的内存池；
- 使用 `__managed__` 说明符的全局变量（见[内存空间说明符](../05-technical-appendices/cpp-language-extensions.html)）。

在具有 [HMM](#26222-hmm具有软件一致性的完整统一内存-hmm-full-unified-memory-with-software-coherency) 或 [ATS](#26221-具有硬件一致性的完整统一内存-full-unified-memory-with-hardware-coherency) 的系统上，无论如何分配，所有系统内存都隐式成为 managed memory，不需要特殊分配。

### 2.6.2.1 统一内存范式（Unified Memory Paradigms）

统一内存的功能和行为会因操作系统、Linux kernel 版本、GPU 硬件以及 GPU-CPU 互连而不同。可以使用 `cudaDeviceGetAttribute` 查询几个属性，确定可用的统一内存形式：

- `cudaDevAttrConcurrentManagedAccess`：为 1 表示完整统一内存支持，为 0 表示有限支持；
- `cudaDevAttrPageableMemoryAccess`：为 1 表示所有系统内存都是完整支持的统一内存；为 0 表示只有显式分配为 managed memory 的内存是完整支持的统一内存；
- `cudaDevAttrPageableMemoryAccessUsesHostPageTables`：表示 CPU/GPU 一致性的机制，为 1 表示硬件一致性，为 0 表示软件一致性。

![图 21：统一内存范式流程图](/images/chapter-02/unified-memory-explainer.png)

<div class="figure-caption"><strong>图 21：统一内存范式流程图。</strong> 原 PDF 第 102 页。先检查 `cudaDevAttrConcurrentManagedAccess` 是否提供完整支持，再检查 `cudaDevAttrPageableMemoryAccess` 判断是否所有系统内存都统一，最后用 `cudaDevAttrPageableMemoryAccessUsesHostPageTables` 区分硬件一致性和软件一致性。</div>

统一内存有四种操作范式：

- 显式 managed memory 分配的完整支持；
- 所有分配的软件一致性完整支持；
- 所有分配的硬件一致性完整支持；
- 有限统一内存支持。

有完整支持时，系统可能要求显式分配，也可能把所有系统内存隐式视为统一内存。如果所有内存都隐式统一，一致性机制可以是软件或硬件。Windows 和部分 Tegra 设备只提供有限统一内存支持。

### 统一内存范式概览表

| 统一内存范式 | 设备属性 | 完整文档 |
| --- | --- | --- |
| 有限统一内存支持 | `cudaDevAttrConcurrentManagedAccess` 为 0 | [Windows、WSL 和 Tegra 上的统一内存](../04-cuda-features/unified-memory.html)；[CUDA for Tegra Memory Management](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management) |
| 显式 managed memory 分配的完整支持 | `cudaDevAttrPageableMemoryAccess` 为 0，且 `cudaDevAttrConcurrentManagedAccess` 为 1 | [仅支持 CUDA Managed Memory 的设备](../04-cuda-features/unified-memory.html) |
| 所有分配的软件一致性完整支持 | `cudaDevAttrPageableMemoryAccessUsesHostPageTables` 为 0，`cudaDevAttrPageableMemoryAccess` 为 1，`cudaDevAttrConcurrentManagedAccess` 为 1 | [完整 CUDA 统一内存支持的设备](../04-cuda-features/unified-memory.html) |
| 所有分配的硬件一致性完整支持 | `cudaDevAttrPageableMemoryAccessUsesHostPageTables` 为 1，`cudaDevAttrPageableMemoryAccess` 为 1，`cudaDevAttrConcurrentManagedAccess` 为 1 | [完整 CUDA 统一内存支持的设备](../04-cuda-features/unified-memory.html) |

#### 2.6.2.1.1 统一内存范式：代码示例（Unified Memory Paradigm: Code Example）

下面的代码对系统中的每张 GPU 查询设备属性，并按照[图 21](#2621-统一内存范式-unified-memory-paradigms)的逻辑确定统一内存范式。

```cpp
void queryDevices()
{
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    for(int i=0; i<numDevices; i++)
    {
        cudaSetDevice(i);
        cudaInitDevice(0, 0, 0);
        int deviceId = i;

        int concurrentManagedAccess = -1;
        cudaDeviceGetAttribute (&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, deviceId);
        int pageableMemoryAccess = -1;
        cudaDeviceGetAttribute (&pageableMemoryAccess, cudaDevAttrPageableMemoryAccess, deviceId);
        int pageableMemoryAccessUsesHostPageTables = -1;
        cudaDeviceGetAttribute (&pageableMemoryAccessUsesHostPageTables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, deviceId);

        printf("Device %d has ", deviceId);
        if(concurrentManagedAccess){
            if(pageableMemoryAccess){
                printf("full unified memory support");
                if( pageableMemoryAccessUsesHostPageTables)
                    { printf(" with hardware coherency\n");  }
                else
                    { printf(" with software coherency\n"); }
            }
            else
                { printf("full unified memory support for CUDA-made managed allocations\n"); }
        }
        else
        {   printf("limited unified memory support: Windows, WSL, or Tegra\n");  }
    }
}
```

### 2.6.2.2 完整统一内存功能支持（Full Unified Memory Feature Support）

大多数 Linux 系统都完整支持统一内存。如果设备属性 `cudaDevAttrPageableMemoryAccess` 为 1，那么无论通过 CUDA API 还是系统 API 分配，所有系统内存都作为具有完整功能支持的统一内存运行。这也包括通过 `mmap` 创建的文件后备内存分配。

如果 `cudaDevAttrPageableMemoryAccess` 为 0，那么只有由 CUDA 分配为 managed memory 的内存表现为统一内存。通过系统 API 分配的内存不是 managed memory，也不一定能被 GPU kernel 访问。

一般来说，完整支持下的统一分配具有以下行为：

- managed memory 通常首先分配在第一次访问它的处理器对应的内存空间中；
- 如果另一个处理器使用当前驻留在其他处理器内存中的 managed memory，通常会发生迁移；
- managed memory 以 memory page（软件一致性）或 cache line（硬件一致性）为粒度迁移或访问；
- 允许超额分配：应用可以分配多于 GPU 物理可用容量的 managed memory。

实际分配和迁移行为可能偏离上述描述。程序员可以使用[提示和预取](#2624-内存建议与预取-memory-advise-and-prefetch)影响这些行为。完整统一内存支持的完整说明见[具有完整 CUDA 统一内存支持的设备](../04-cuda-features/unified-memory.html)。

#### 2.6.2.2.1 具有硬件一致性的完整统一内存（Full Unified Memory with Hardware Coherency）

在 Grace Hopper、Grace Blackwell 等使用 NVIDIA CPU、且 CPU 与 GPU 之间通过 NVLink Chip-to-Chip（C2C）互连的硬件上，可以使用地址转换服务（ATS）。ATS 可用时，`cudaDevAttrPageableMemoryAccessUsesHostPageTables` 为 1。

除完整支持所有 host 分配的统一内存外，ATS 还提供：

- 驻留在 GPU 上的 managed allocation（例如 `cudaMallocManaged` 分配）可以由 CPU 访问而不迁移（`cudaDevAttrDirectManagedMemAccessFromHost` 为 1）；
- CPU 与 GPU 的连接支持原生原子操作（`cudaDevAttrHostNativeAtomicSupported` 为 1）；
- 硬件一致性支持相比软件一致性可能改善性能。

ATS 提供 HMM 的全部能力；ATS 可用时会自动禁用 HMM。硬件一致性和软件一致性的更多讨论见[CPU 与 GPU 页表：硬件一致性与软件一致性](../04-cuda-features/unified-memory.html)。

::: warning
硬件一致性不会使 host 可以访问 GPU-only 分配，例如使用 `cudaMalloc` 创建的分配。
:::

#### 2.6.2.2.2 HMM：具有软件一致性的完整统一内存（HMM - Full Unified Memory with Software Coherency）

*异构内存管理（Heterogeneous Memory Management，HMM）* 是 Linux 操作系统（要求合适的 kernel 版本）提供的一项功能，可启用具有软件一致性的[完整统一内存支持](#2622-完整统一内存功能支持-full-unified-memory-feature-support)。HMM 为通过 PCIe 连接的 GPU 带来了 ATS 的一部分能力和便利性。

在 Linux Kernel 6.1.24、6.2.11 或 6.3 及更高版本上，HMM 可能可用。可以使用下面的命令检查寻址模式是否为 `HMM`：

```bash
$ nvidia-smi -q | grep Addressing
Addressing Mode : HMM
```

HMM 可用时，支持[完整统一内存](#2622-完整统一内存功能支持-full-unified-memory-feature-support)，所有系统分配都会隐式成为统一内存。如果系统同时具有 [ATS](#26221-具有硬件一致性的完整统一内存-full-unified-memory-with-hardware-coherency)，则禁用 HMM 并使用 ATS，因为 ATS 提供 HMM 的全部能力以及更多能力。

### 2.6.2.3 有限统一内存支持（Limited Unified Memory Support）

在 Windows（包括 Windows Subsystem for Linux，WSL）以及一些 Tegra 系统上，只提供统一内存功能的有限子集。这些系统支持 managed memory，但 CPU 与 GPU 之间的迁移行为不同：

- managed memory 首先分配在 CPU 的物理内存中；
- managed memory 以大于虚拟内存页的粒度迁移；
- GPU 开始执行时，managed memory 才迁移到 GPU；
- GPU 活跃时，CPU 不得访问 managed memory；
- GPU 同步时，managed memory 迁移回 CPU；
- 不允许 GPU 内存超额分配；
- 只有 CUDA 显式分配为 managed memory 的内存才是统一内存。

这种范式的完整说明见[Windows、WSL 和 Tegra 上的统一内存](../04-cuda-features/unified-memory.html)。

### 2.6.2.4 内存建议与预取（Memory Advise and Prefetch）

程序员可以向管理统一内存的 NVIDIA Driver 提供提示，帮助它最大化应用性能。CUDA API `cudaMemAdvise` 允许程序员指定影响分配放置位置，以及从另一个 device 访问时是否迁移内存的属性。

`cudaMemPrefetchAsync` 允许程序员建议开始把特定分配异步迁移到另一个位置。常见用法是在启动 kernel 之前开始传输 kernel 将使用的数据，从而可以在其他 GPU kernel 执行期间传输数据。

[性能提示](../04-cuda-features/unified-memory.html)一节介绍可以传给 `cudaMemAdvise` 的不同提示，并展示 `cudaMemPrefetchAsync` 的用法。

## 2.6.3 页锁定 Host 内存（Page-Locked Host Memory）

在[入门代码示例](./intro-to-cuda-cpp.html)中，使用 `cudaMallocHost` 在 CPU 上分配内存。该 API 在 host 上分配**页锁定（page-locked）**或 **pinned** 内存。通过 `malloc`、`new` 或 `mmap` 等传统机制分配的 host 内存没有页锁定，这意味着它可能被操作系统换出到磁盘，或在物理上重新定位。

CPU 与 GPU 之间的[异步复制](./asynchronous-execution.html)必须使用页锁定 host 内存。页锁定 host 内存也能改善同步复制的性能。页锁定内存还可以[映射](#2631-映射内存-mapped-memory)到 GPU，使 GPU kernel 能直接访问。

CUDA runtime 提供了分配页锁定 host 内存，或对已有分配进行页锁定的 API：

- `cudaMallocHost`：分配页锁定 host 内存；
- `cudaHostAlloc`：默认行为与 `cudaMallocHost` 相同，但还接受用于指定其他内存参数的标志；
- `cudaFreeHost`：释放由 `cudaMallocHost` 或 `cudaHostAlloc` 分配的内存；
- `cudaHostRegister`：对 CUDA API 之外分配的现有内存范围执行页锁定，例如由 `malloc` 或 `mmap` 分配的内存。

`cudaHostRegister` 可以对第三方库或开发者无法控制的其他代码分配的 host 内存执行页锁定，使它能够用于异步复制或映射。

::: tip 注意
系统中的所有 GPU 都可以使用页锁定 host 内存执行异步复制和映射内存。

在非 I/O 一致性 Tegra 设备上，页锁定 host 内存不会被缓存；并且这些设备不支持 `cudaHostRegister()`。
:::

### 2.6.3.1 映射内存（Mapped Memory）

在具有 [HMM](#26222-hmm具有软件一致性的完整统一内存-hmm-full-unified-memory-with-software-coherency) 或 [ATS](#26221-具有硬件一致性的完整统一内存-full-unified-memory-with-hardware-coherency) 的系统上，GPU 可以使用 host 指针直接访问全部 host memory。没有 ATS 或 HMM 时，可以把 host allocation **映射（mapping）**到 GPU 的内存空间，使 GPU 可以访问。映射内存始终是页锁定内存。

下面的代码示例使用一个直接操作映射 host memory 的数组复制 kernel：

```cpp
__global__ void copyKernel(float* a, float* b)
{
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        a[idx] = b[idx];
}
```

在某些情况下，如果某些数据不需要复制到 GPU、但 kernel 需要访问它，映射内存可能有用。不过，kernel 访问映射内存需要通过 CPU-GPU 互连、PCIe 或 NVLink C2C 传输；与访问 device memory 相比，这些操作延迟更高、带宽更低。因此，对大多数 kernel 内存需求而言，不应把映射内存视为统一内存或[显式内存管理](./intro-to-cuda-cpp.html)的高性能替代方案。

#### 2.6.3.1.1 `cudaMallocHost` 与 `cudaHostAlloc`

使用 `cudaMallocHost` 或 `cudaHostAlloc` 分配的 host 内存会自动映射。这些 API 返回的指针可以直接在 kernel 代码中使用，以访问 host 上的内存；host memory 通过 CPU-GPU 互连访问。

```cpp
void usingMallocHost() {
  float* a = nullptr;
  float* b = nullptr;

  CUDA_CHECK(cudaMallocHost(&a, vLen*sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&b, vLen*sizeof(float)));

  initVector(b, vLen);
  memset(a, 0, vLen*sizeof(float));

  int threads = 256;
  int blocks = vLen/threads;
  copyKernel<<<blocks, threads>>>(a, b);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("Using cudaMallocHost: ");
  checkAnswer(a,b);
}
```

```cpp
void usingCudaHostAlloc() {
  float* a = nullptr;
  float* b = nullptr;

  CUDA_CHECK(cudaHostAlloc(&a, vLen*sizeof(float), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc(&b, vLen*sizeof(float), cudaHostAllocMapped));

  initVector(b, vLen);
  memset(a, 0, vLen*sizeof(float));

  int threads = 256;
  int blocks = vLen/threads;
  copyKernel<<<blocks, threads>>>(a, b);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("Using cudaHostAlloc: ");
  checkAnswer(a, b);
}
```

#### 2.6.3.1.2 `cudaHostRegister`

在没有 ATS 和 HMM 时，系统分配器创建的内存仍可以通过 `cudaHostRegister` 映射，使 GPU kernel 直接访问。不过，与 CUDA API 创建的内存不同，这类内存不能在 kernel 中使用 host 指针访问。必须使用 `cudaHostGetDevicePointer()` 获取 device 内存区域中的指针，并在 kernel 代码中使用该指针。

```cpp
void usingRegister() {
  float* a = nullptr;
  float* b = nullptr;
  float* devA = nullptr;
  float* devB = nullptr;

  a = (float*)malloc(vLen*sizeof(float));
  b = (float*)malloc(vLen*sizeof(float));
  CUDA_CHECK(cudaHostRegister(a, vLen*sizeof(float), 0 ));
  CUDA_CHECK(cudaHostRegister(b, vLen*sizeof(float), 0  ));

  CUDA_CHECK(cudaHostGetDevicePointer((void**)&devA, (void*)a, 0));
  CUDA_CHECK(cudaHostGetDevicePointer((void**)&devB, (void*)b, 0));

  initVector(b, vLen);
  memset(a, 0, vLen*sizeof(float));

  int threads = 256;
  int blocks = vLen/threads;
  copyKernel<<<blocks, threads>>>(devA, devB);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("Using cudaHostRegister: ");
  checkAnswer(a, b);
}
```

#### 2.6.3.1.3 比较统一内存和映射内存（Comparing Unified Memory and Mapped Memory）

映射内存使 GPU 可以访问 CPU 内存，但不保证所有类型的访问（例如原子操作）在所有系统上都受支持。统一内存则保证支持所有访问类型。

映射内存仍然驻留在 CPU 内存中，因此所有 GPU 访问都必须经过 CPU 与 GPU 之间的 PCIe 或 NVLink 连接。经过这些连接的访问延迟明显高于访问 GPU 内存，且可用总带宽更低。因此，用映射内存承载 kernel 的所有内存访问，很可能无法充分利用 GPU 计算资源。

统一内存通常会迁移到正在访问它的处理器的物理内存中。第一次迁移之后，kernel 对同一个 memory page 或 cache line 的重复访问就可以利用完整的 GPU 内存带宽。

::: tip 注意
过去的文档也把映射内存称为 *zero-copy* 内存。

在所有 CUDA 应用使用[统一虚拟地址空间](#261-统一虚拟地址空间-unified-virtual-address-space)之前，需要额外的 API 启用内存映射（使用 `cudaDeviceMapHost` 调用 `cudaSetDeviceFlags`）。现在不再需要这些 API。

对映射 host memory 执行的原子函数，从 host 或其他 GPU 的角度看并不是原子的。

CUDA runtime 要求 device 发起的、对 host memory 的 1、2、4、8 和 16 字节自然对齐 load/store，从 host 和其他设备角度看都保持为单次访问。在一些平台上，硬件可能把内存原子操作拆成独立的 load 和 store；这些组成操作同样必须满足自然对齐访问的保持要求。CUDA runtime 不支持这样的 PCI Express 拓扑：PCI Express bridge 会拆分 8 字节自然对齐操作；NVIDIA 也不知道有会拆分 16 字节自然对齐操作的拓扑。
:::

## 2.6.4 小结（Summary）

- 在具有异构内存管理（HMM）或地址转换服务（ATS）的 Linux 平台上，所有系统分配的内存都是 managed memory；
- 在没有 HMM 或 ATS 的 Linux 平台、Tegra 处理器以及所有 Windows 平台上，managed memory 必须通过 CUDA 分配：
  - `cudaMallocManaged`；
  - 使用 `allocType=cudaMemAllocationTypeManaged` 创建的内存池配合 `cudaMallocFromPoolAsync`；
  - 使用 `__managed__` 说明符的全局变量；
- Windows 和 Tegra 处理器上的统一内存存在限制；
- 在通过 NVLink C2C 连接、且具有 ATS 的系统上，使用 `cudaMallocManaged` 分配的 device memory 可以由 CPU 或其他 GPU 直接访问。

## 本节导航

- [2.1 CUDA C++ 入门](./intro-to-cuda-cpp.html)
- [2.5 异步执行](./asynchronous-execution.html)
- [4.1 统一内存专题](../04-cuda-features/unified-memory.html)
