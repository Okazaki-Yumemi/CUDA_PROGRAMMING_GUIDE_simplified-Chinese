---
title: 4.15 进程间通信（Interprocess Communication）
description: CUDA 进程间通信 API、传统 IPC 以及虚拟内存管理 API。
---

# 4.15 进程间通信（Interprocess Communication）

CUDA 支持由不同主机进程管理的多个 GPU 之间进行通信。应用可以使用进程间通信（IPC）API 和可通过 IPC 共享的内存缓冲区，创建可移植到其他进程的句柄；随后，其他进程可以使用这些句柄获取指向对等 GPU 设备内存的进程本地设备指针。

由某个主机线程创建的任意设备内存指针或事件句柄，都可以被同一进程内的其他线程直接引用。但是，设备指针或事件句柄在创建它们的进程之外无效，因此属于其他进程的线程不能直接引用它们。要跨进程访问设备内存和 CUDA 事件，应用必须使用 CUDA 进程间通信（IPC）API 或虚拟内存管理（VMM）API，创建可移植到其他进程的句柄，然后通过标准主机操作系统 IPC 机制（例如进程间共享内存或文件）将句柄共享给其他进程。进程之间交换可移植句柄后，必须使用 CUDA IPC 或 VMM API 从句柄获取进程本地设备指针。之后，进程本地设备指针的用法与单进程中的设备指针完全相同。

在单节点、单个操作系统实例内进行 IPC 时使用的这种可移植句柄方法，同样用于多节点、NVLink 连接集群中 GPU 之间的点对点通信。在多节点场景中，通信 GPU 由运行于每个集群节点独立操作系统实例中的进程管理，因此需要在操作系统实例层面之上增加抽象。多节点点对点通信通过在多节点 GPU 对等端之间创建并交换所谓的“fabric”句柄实现；参与进程和操作系统实例随后会根据多节点 rank 获取相应的进程本地设备指针。

有关建立和交换进程可移植句柄，以及节点和操作系统实例可移植句柄（这些句柄用于获取 GPU 通信所需的进程本地设备指针）的具体 API，请参阅下文的单节点 CUDA IPC 和 [4.16 虚拟内存管理（Virtual Memory Management）](./virtual-memory-management.html)。

::: note

CUDA IPC API 与虚拟内存管理（VMM）API 用于 IPC 时，各自具有不同的优点和限制。

CUDA IPC API 当前仅支持 Linux 平台。

CUDA 虚拟内存管理 API 允许在内存分配时对每次分配分别控制对等访问和共享，但要求使用 CUDA Driver API。

:::

## 4.15.1 使用传统进程间通信 API（IPC using the Legacy Interprocess Communication API）

要跨进程共享设备内存指针和事件，应用必须使用 CUDA 进程间通信 API；该 API 的详细说明见参考手册。IPC API 可以使用 `cudaIpcGetMemHandle()` 获取给定设备内存指针对应的 IPC 句柄。CUDA IPC 句柄可以通过标准主机操作系统 IPC 机制（例如进程间共享内存或文件）传递给另一个进程。另一个进程可以使用 `cudaIpcOpenMemHandle()` 从 IPC 句柄恢复一个有效的设备指针，并在该进程中使用它。事件句柄也可以通过类似的入口点共享。

IPC API 的一个使用示例是：由一个主进程生成一批输入数据，再将数据提供给多个辅助进程，而不需要让每个辅助进程重新生成或复制这批数据。

::: note

IPC API 仅支持 Linux。

请注意，IPC API 不支持 `cudaMallocManaged` 分配的内存。

相互使用 CUDA IPC 通信的应用，应使用相同的 CUDA driver 和 runtime 进行编译、链接和运行。

出于性能原因，`cudaMalloc()` 进行的分配可能来自一个更大的底层内存块。在这种情况下，CUDA IPC API 会共享整个底层内存块，这可能导致其他子分配也被共享，并可能造成进程之间的信息泄露。为避免这种行为，建议只共享大小按 2 MiB 对齐的分配。

在计算能力 7.x 及更高版本的 L4T 和嵌入式 Linux Tegra 设备上，仅支持 IPC 事件共享 API；Tegra 平台不支持 IPC 内存共享 API。

:::

## 4.15.2 使用虚拟内存管理 API 进行 IPC（IPC using the Virtual Memory Management API）

CUDA 虚拟内存管理 API 支持创建可通过 IPC 共享的内存分配，并且借助操作系统特定的 IPC 句柄数据结构支持多个操作系统。
