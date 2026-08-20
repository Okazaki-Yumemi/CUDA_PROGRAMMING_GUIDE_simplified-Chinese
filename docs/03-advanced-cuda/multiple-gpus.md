---
title: 3.4 使用多 GPU 编程系统
description: 多 GPU 上下文、执行管理、点对点传输与内存访问
---

# 3.4 使用多 GPU 编程系统（Programming Systems with Multiple GPUs）

多 GPU 编程可以利用多 GPU 系统提供的更大聚合算力、内存容量和内存带宽，使应用能够处理单个 GPU 无法处理的问题规模，并达到单个 GPU 无法实现的性能水平。

CUDA 通过主机 API、驱动基础设施以及 GPU 硬件技术支持多 GPU 编程：

- 主机线程的 CUDA 上下文管理；
- 系统中所有处理器统一的内存寻址；
- GPU 之间的点对点大块内存传输；
- 细粒度的 GPU 点对点 load/store 内存访问；
- 更高层的抽象和配套系统软件，例如 CUDA 进程间通信、使用 [NCCL](https://developer.nvidia.com/nccl) 进行并行归约，以及使用 NVLink 和/或 GPU-Direct RDMA，并通过 [NVSHMEM](https://developer.nvidia.com/nvshmem) 和 MPI 等 API 进行通信。

在最基本的层面上，多 GPU 编程要求应用程序同时管理多个活动的 CUDA 上下文，将数据分发给各个 GPU，在 GPU 上启动 kernel 来完成工作，并通信或收集结果，以便应用程序继续处理这些结果。具体实现方式取决于如何将应用算法、可用并行性和现有代码结构，以最有效的方式映射到适合的多 GPU 编程方法。常见的多 GPU 编程方法包括：

- 由单个主机线程驱动多个 GPU；
- 使用多个主机线程，每个线程驱动自己的 GPU；
- 使用多个单线程主机进程，每个进程驱动自己的 GPU；
- 使用包含多个线程的多个主机进程，每个进程驱动自己的 GPU；
- 使用多节点、通过 NVLink 连接的集群，在集群节点上的多个操作系统实例中运行的线程和进程共同驱动 GPU。

GPU 之间可以通过设备内存之间的内存传输和点对点访问进行通信，这覆盖了上面列出的各种多设备工作分发方式。通过查询并启用 GPU 点对点内存访问，以及利用 NVLink 实现设备间的高带宽传输和更细粒度的 load/store 操作，可以支持高性能、低延迟的 GPU 通信。

CUDA 统一虚拟寻址允许同一主机进程中的多个 GPU 之间进行通信，只需很少的额外步骤即可查询并启用高性能的点对点内存访问和传输，例如通过 NVLink 进行通信。

由不同主机进程管理的多个 GPU 之间的通信，可以通过进程间通信（IPC）和虚拟内存管理（VMM）API 实现。有关高层 IPC 概念和节点内 CUDA IPC API 的介绍，请参阅[进程间通信](../04-cuda-features/interprocess-communication.html#interprocess-communication)一节。高级虚拟内存管理（VMM）API 同时支持节点内和多节点 IPC，可在 Linux 和 Windows 操作系统上使用，并允许对 IPC 共享内存缓冲区的每次分配进行粒度控制，详见[虚拟内存管理](../04-cuda-features/virtual-memory-management.html#virtual-memory-management)。

CUDA 本身提供了在一组 GPU（也可以包括主机）内实现集合操作所需的 API，但不会直接提供高层的多 GPU 集合 API。多 GPU 集合操作由 NCCL 和 NVSHMEM 等更高层的 CUDA 通信库提供。

## 3.4.1 多设备上下文与执行管理（Multi-Device Context and Execution Management）

应用要使用多个 GPU，首先需要枚举可用的 GPU 设备，根据设备的硬件属性、CPU 亲和性以及与其他设备的连接情况，在可用设备中进行适当选择，并为应用将使用的每个设备创建 CUDA 上下文。

### 3.4.1.1 设备枚举（Device Enumeration）

下面的代码示例展示了如何查询启用 CUDA 的设备数量、枚举每个设备，以及查询设备属性。

```cpp
int deviceCount;
cudaGetDeviceCount(&deviceCount);
int device;
for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
}
```

### 3.4.1.2 设备选择（Device Selection）

主机线程可以随时调用 `cudaSetDevice()`，设置它当前正在操作的设备。设备内存分配和 kernel 启动都会作用于当前设备；流和事件也会与当前设置的设备关联。在主机线程调用 `cudaSetDevice()` 之前，当前设备默认为设备 0。

下面的代码示例说明，设置当前设备会如何影响后续的内存分配和 kernel 执行操作。

```cpp
size_t size = 1024 * sizeof(float);
cudaSetDevice(0);            // Set device 0 as current
float* p0;
cudaMalloc(&p0, size);       // Allocate memory on device 0
MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0

cudaSetDevice(1);            // Set device 1 as current
float* p1;
cudaMalloc(&p1, size);       // Allocate memory on device 1
MyKernel<<<1000, 128>>>(p1); // Launch kernel on device 1
```

### 3.4.1.3 多设备流、事件与内存复制行为（Multi-Device Stream, Event, and Memory Copy Behavior）

如果 kernel 启动提交到一个不属于当前设备的流，就会失败，示例如下。

```cpp
cudaSetDevice(0);               // Set device 0 as current
cudaStream_t s0;
cudaStreamCreate(&s0);          // Create stream s0 on device 0
MyKernel<<<100, 64, 0, s0>>>(); // Launch kernel on device 0 in s0

cudaSetDevice(1);               // Set device 1 as current
cudaStream_t s1;
cudaStreamCreate(&s1);          // Create stream s1 on device 1
MyKernel<<<100, 64, 0, s1>>>(); // Launch kernel on device 1 in s1

// This kernel launch will fail, since stream s0 is not associated to device 1:
MyKernel<<<100, 64, 0, s0>>>(); // Launch kernel on device 1 in s0
```

即使内存复制提交到一个不属于当前设备的流，复制操作也会成功。

如果输入事件和输入流属于不同设备，`cudaEventRecord()` 会失败。

如果两个输入事件属于不同设备，`cudaEventElapsedTime()` 会失败。

即使输入事件属于与当前设备不同的设备，`cudaEventSynchronize()` 和 `cudaEventQuery()` 也会成功。

即使输入流和输入事件属于不同设备，`cudaStreamWaitEvent()` 也会成功。因此，可以使用 `cudaStreamWaitEvent()` 在多个设备之间进行同步。

每个设备都有自己的[默认流](../02-programming-gpus/asynchronous-execution.html#async-execution-blocking-non-blocking-default-stream)，因此，一个设备默认流上提交的命令，可能会相对于另一个设备默认流上提交的命令乱序执行，或者与其并发执行。

## 3.4.2 多设备点对点传输与内存访问（Multi-Device Peer-to-Peer Transfers and Memory Access）

### 3.4.2.1 点对点内存传输（Peer-to-Peer Memory Transfers）

CUDA 可以在设备之间执行内存传输。当点对点内存访问可用时，CUDA 会利用专用的复制引擎和 NVLink 硬件来最大化性能。

`cudaMemcpy` 可以配合 `cudaMemcpyDeviceToDevice` 或 `cudaMemcpyDefault` 复制类型使用。

在其他情况下，必须使用 `cudaMemcpyPeer()`、`cudaMemcpyPeerAsync()`、`cudaMemcpy3DPeer()` 或 `cudaMemcpy3DPeerAsync()` 执行复制，示例如下。

```cpp
cudaSetDevice(0);                   // Set device 0 as current
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size);              // Allocate memory on device 0

cudaSetDevice(1);                   // Set device 1 as current
float* p1;
cudaMalloc(&p1, size);              // Allocate memory on device 1

cudaSetDevice(0);                   // Set device 0 as current
MyKernel<<<1000, 128>>>(p0);        // Launch kernel on device 0

cudaSetDevice(1);                   // Set device 1 as current
cudaMemcpyPeer(p1, 1, p0, 0, size); // Copy p0 to p1
MyKernel<<<1000, 128>>>(p1);        // Launch kernel on device 1
```

在两个不同设备的内存之间，通过隐式 *NULL* 流进行的复制：

- 不会在之前提交给任一设备的所有命令完成之前开始；
- 在复制完成之前，提交给任一设备的后续命令（参见[异步执行](../02-programming-gpus/asynchronous-execution.html#asynchronous-execution)）都不能开始。

与流的常规行为一致，在另一个流中的复制或 kernel 执行期间，设备内存之间的异步复制可以与它们重叠执行。

如果两个设备之间启用了点对点访问，例如按照[点对点内存访问](#3422-点对点内存访问-peer-to-peer-memory-access)中的说明启用，那么这两个设备之间的点对点内存复制就不再需要经过主机中转，因此速度更快。

### 3.4.2.2 点对点内存访问（Peer-to-Peer Memory Access）

根据系统属性，尤其是 PCIe 和/或 NVLink 拓扑结构，设备可以寻址彼此的内存，也就是说，在一个设备上执行的 kernel 可以解引用指向另一个设备内存的指针。如果对指定设备调用 `cudaDeviceCanAccessPeer()` 返回 `true`，则这两个设备之间支持点对点内存访问。

必须通过调用 `cudaDeviceEnablePeerAccess()` 在两个设备之间启用点对点内存访问，示例如下。在不支持 NVSwitch 的系统上，每个设备最多可以支持 8 个系统级的 peer 连接。

两个设备使用统一虚拟地址空间（参见[统一虚拟地址空间](../04-cuda-features/unified-memory.html#memory-unified-virtual-address-space)），因此可以使用同一个指针为两个设备寻址内存，如下面的代码示例所示。

```cpp
cudaSetDevice(0);                   // Set device 0 as current
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size);              // Allocate memory on device 0
MyKernel<<<1000, 128>>>(p0);        // Launch kernel on device 0

cudaSetDevice(1);                   // Set device 1 as current
cudaDeviceEnablePeerAccess(0, 0);   // Enable peer-to-peer access
                                    // with device 0

// Launch kernel on device 1
// This kernel launch can access memory on device 0 at address p0
MyKernel<<<1000, 128>>>(p0);
```

> **注意**
>
> 使用 `cudaDeviceEnablePeerAccess()` 启用 peer 内存访问，会对 peer 设备上之前和之后的所有 GPU 内存分配全局生效。通过 `cudaDeviceEnablePeerAccess()` 启用对某个设备的 peer 访问，会增加该 peer 上设备内存分配操作的运行时成本，因为这些分配必须立即对当前设备以及其他已获得访问权限的 peer 设备可见。该开销会随着 peer 设备数量增加而产生乘法式增长。
>
> 一种更具扩展性的做法，是使用 CUDA 虚拟内存管理 API，仅在需要时、分配内存时显式分配可供 peer 访问的内存区域。通过在内存分配时显式请求 peer 可访问性，不需要让 peer 访问的分配不会产生额外的运行时分配成本；同时，peer 可访问的数据结构会得到正确的作用域控制，从而改善软件调试和可靠性。详见[虚拟内存管理](../04-cuda-features/virtual-memory-management.html#virtual-memory-management)。

### 3.4.2.3 点对点内存一致性（Peer-to-Peer Memory Consistency）

必须使用同步操作，来保证分布在多个设备上的并发执行 grid 中各线程的内存访问顺序和正确性。跨设备同步的线程运行在 `thread_scope_system` [同步作用域](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#thread-scopes)中。类似地，内存操作属于 `thread_scope_system` [内存同步域](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-synchronization-domains)。

当只有一个 GPU 访问某个 peer 设备内存中的对象时，CUDA [原子函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions)可以对该对象执行读-修改-写操作。peer 原子性所需满足的条件和限制，详见 CUDA 内存模型中的[原子性要求](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#atomicity)讨论。

### 3.4.2.4 多设备托管内存（Multi-Device Managed Memory）

支持点对点的多 GPU 系统可以使用托管内存。有关并发多设备托管内存访问的详细要求，以及用于让 GPU 独占访问托管内存的 API，请参阅[多 GPU](../04-cuda-features/unified-memory.html#um-legacy-multi-gpu)一节。

### 3.4.2.5 主机 IOMMU 硬件、PCI 访问控制服务与虚拟机（Host IOMMU Hardware, PCI Access Control Services, and VMs）

在 Linux 上，CUDA 和显示驱动不支持启用 IOMMU 的裸机 PCIe 点对点内存传输。但是，CUDA 和显示驱动支持通过虚拟机直通使用 IOMMU。在裸机系统上运行 Linux 时，必须禁用 IOMMU，以避免设备内存发生静默损坏。相反，对于虚拟机，应启用 IOMMU，并使用 VFIO 驱动程序进行 PCIe 直通。

Windows 不存在上述 IOMMU 限制。

另请参阅：[在 64 位平台上分配 DMA 缓冲区](https://download.nvidia.com/XFree86/Linux-x86_64/510.85.02/README/dma_issues.html)。

此外，在支持 IOMMU 的系统上可以启用 PCI 访问控制服务（ACS）。PCI ACS 功能会将所有 PCI 点对点流量重定向到 CPU 根复合体，这可能因为整体二分带宽降低而造成显著的性能损失。
