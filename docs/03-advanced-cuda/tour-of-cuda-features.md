---
title: 3.5 CUDA 功能概览（A Tour of CUDA Features）
description: 按问题域介绍 CUDA 的主要功能、适用场景及相关官方章节
---

# 3.5 CUDA 功能概览（A Tour of CUDA Features）

本编程指南的第 1–3 部分介绍了 CUDA 和 GPU 编程，涵盖了概念基础以及简单代码示例。指南第 4 部分中介绍具体 CUDA 功能的章节，默认读者已经掌握本指南第 1–3 部分涉及的概念。

CUDA 提供了许多面向不同问题的功能，但并非所有功能都适用于每一种使用场景。本章将逐一介绍这些功能，说明它们的设计用途，以及它们可能帮助解决的问题。这些功能按照要解决的问题类型粗略分类；例如 CUDA Graphs 这样的功能可能同时属于多个类别。

[第 4 部分：CUDA Features](/04-cuda-features/index.html) 会对这些 CUDA 功能进行更完整的说明。

## 3.5.1 改进 Kernel 性能（Improving Kernel Performance）

本节列出的功能都旨在帮助 Kernel 开发者最大限度地提升 Kernel 的性能。

### 3.5.1.1 异步屏障（Asynchronous Barriers）

[异步屏障](/04-cuda-features/asynchronous-barriers.html#asynchronous-barriers) 在[第 3.2.4.2 节](/03-advanced-cuda/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers)中介绍，它可以对线程之间的同步进行更细致的控制。异步屏障将“到达屏障”和“等待屏障”这两个动作分离开来。这样，应用程序在等待其他线程到达时，仍然可以执行不依赖该屏障的工作。异步屏障可以针对不同的[线程作用域（thread scopes）](/03-advanced-cuda/advanced-kernel-programming.html#advanced-kernels-thread-scopes)进行指定。异步屏障的完整细节见[第 4.9 节](/04-cuda-features/asynchronous-barriers.html#asynchronous-barriers)。

### 3.5.1.2 异步数据复制与张量内存加速器（TMA）（Asynchronous Data Copies and the Tensor Memory Accelerator (TMA)）

在 CUDA Kernel 代码的上下文中，[异步数据复制](/04-cuda-features/asynchronous-data-copies.html#async-copies)是指在执行计算的同时，在共享内存与 GPU DRAM 之间移动数据的能力。它不要与 CPU 和 GPU 之间的异步内存复制混淆。这项功能使用了异步屏障。[第 4.11 节](/04-cuda-features/asynchronous-data-copies.html#async-copies)详细介绍了异步复制的使用方法。

### 3.5.1.3 流水线（Pipelines）

[流水线](/04-cuda-features/pipelines.html#pipelines)是一种安排工作阶段并协调多缓冲生产者–消费者模式的机制，通常用于将计算与[异步数据复制](/04-cuda-features/asynchronous-data-copies.html#async-copies)重叠执行。[第 4.10 节](/04-cuda-features/pipelines.html#pipelines)提供了在 CUDA 中使用流水线的详细说明和示例。

### 3.5.1.4 使用 Cluster Launch Control 进行工作窃取（Work Stealing with Cluster Launch Control）

工作窃取是一种在负载不均衡时维持利用率的技术：已经完成自身工作的工作线程可以从其他工作线程那里“窃取”任务。Cluster Launch Control 是计算能力 10.0（Blackwell）引入的一项功能，它让 Kernel 可以直接控制正在执行的 Block 的调度，从而实时适应不均衡的工作负载。一个线程块可以取消另一个尚未开始执行的线程块或 Cluster 的启动，取得它的索引，并立即开始执行被窃取的工作。这种工作窃取流程可以让 SM 保持忙碌，在数据不规则或运行时变化的情况下减少空闲时间，从而实现更细粒度的负载均衡，而不必只依赖硬件调度器。

[第 4.12 节](/04-cuda-features/work-stealing.html#cluster-launch-control)介绍了如何使用此功能。

## 3.5.2 降低延迟（Improving Latencies）

本节列出的功能有一个共同目标：降低某种类型的延迟，尽管不同功能所针对的延迟类型并不相同。总体而言，它们主要关注 Kernel 启动层面或更高层面的延迟。Kernel 内部的 GPU 内存访问延迟不属于本节讨论的延迟范围。

### 3.5.2.1 Green Context（Green Contexts）

[Green Context](/04-cuda-features/green-contexts.html#green-contexts) 也称为*执行上下文（execution context）*，这是 CUDA 为一种功能所使用的名称：程序可以创建只在 GPU 的部分 SM 上执行工作的 [CUDA context](/03-advanced-cuda/driver-api.html#driver-api-context)。默认情况下，Kernel 启动中的线程块会被分派到 GPU 中任何能够满足该 Kernel 资源需求的 SM 上。能够执行一个线程块的 SM 受到许多因素影响，包括但不限于共享内存用量、寄存器用量、是否使用 Cluster，以及线程块中的线程总数。

执行上下文允许 Kernel 在一个专门创建的上下文中启动；该上下文会进一步限制可用于执行 Kernel 的 SM 数量。重要的是，当程序创建一个使用某组 SM 的 Green Context 后，GPU 上的其他上下文不会再把线程块调度到分配给该 Green Context 的 SM 上。这包括主上下文（primary context），即 CUDA Runtime 默认使用的上下文。这样，应用程序就可以把这些 SM 保留给高优先级或对延迟敏感的工作负载。

[第 4.6 节](/04-cuda-features/green-contexts.html#green-contexts)完整介绍了 Green Context 的使用方法。CUDA Runtime 从 CUDA 13.1 开始提供 Green Context。

### 3.5.2.2 流有序内存分配（Stream-Ordered Memory Allocation）

[流有序内存分配器（stream-ordered memory allocator）](/04-cuda-features/stream-ordered-memory-allocator.html#stream-ordered-memory-allocator)允许程序将 GPU 内存的分配和释放操作按顺序插入 [CUDA stream](/02-programming-gpus/asynchronous-execution.html#cuda-streams) 中。`cudaMalloc` 和 `cudaFree` 会立即执行，而 `cudaMallocAsync` 和 `cudaFreeAsync` 则会把内存分配或释放操作插入 CUDA stream。[第 4.3 节](/04-cuda-features/stream-ordered-memory-allocator.html#stream-ordered-memory-allocator)介绍了这些 API 的全部细节。

### 3.5.2.3 CUDA Graphs（CUDA Graphs）

[CUDA Graphs](/04-cuda-features/cuda-graphs.html#cuda-graphs)允许应用程序指定一系列 CUDA 操作（例如 Kernel 启动或内存复制）以及这些操作之间的依赖关系，以便在 GPU 上高效执行。使用 [CUDA streams](/02-programming-gpus/asynchronous-execution.html#cuda-streams)也可以获得类似行为；事实上，创建 Graph 的机制之一就叫做[流捕获（stream capture）](/04-cuda-features/cuda-graphs.html#cuda-graphs-creating-a-graph-using-stream-capture)，它可以将 stream 上的操作记录到 CUDA Graph 中。也可以使用 [CUDA Graphs API](/04-cuda-features/cuda-graphs.html#cuda-graphs-creating-a-graph-using-graph-apis)创建 Graph。

创建 Graph 后，可以将其实例化并执行多次。这对于描述会重复执行的工作负载很有用。Graph 可以降低调用 CUDA 操作时的 CPU 启动开销，并且可以启用只有在预先指定完整工作负载时才能使用的优化，因此能够带来一定的性能收益。

[第 4.2 节](/04-cuda-features/cuda-graphs.html#cuda-graphs)介绍并演示了 CUDA Graphs 的使用方法。

### 3.5.2.4 程序化依赖启动（Programmatic Dependent Launch）

[程序化依赖启动](/04-cuda-features/programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization)是一项 CUDA 功能，它允许依赖 Kernel（即依赖前一个 Kernel 输出的 Kernel）在其所依赖的主 Kernel 完成之前开始执行。依赖 Kernel 可以先执行准备代码和无关工作，直到需要主 Kernel 产生的数据；然后它可以在此处阻塞。主 Kernel 可以在依赖 Kernel 所需的数据准备好后发出信号，解除依赖 Kernel 的阻塞，使其继续执行。这样，两个 Kernel 之间可以形成一定程度的重叠，有助于保持较高的 GPU 利用率，同时缩短关键数据路径的延迟。[第 4.5 节](/04-cuda-features/programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization)介绍了程序化依赖启动。

### 3.5.2.5 延迟加载（Lazy Loading）

[延迟加载](/04-cuda-features/lazy-loading.html#lazy-loading)允许应用程序控制 JIT 编译器在应用启动时的工作方式。如果应用程序包含许多需要从 PTX 编译为 cubin 的 Kernel，并且在启动时对所有 Kernel 执行 JIT 编译，可能会经历较长的启动时间。默认行为是：只有在模块被需要时才编译模块。通过使用[环境变量](/05-technical-appendices/environment-variables.html#cuda-environment-variables)可以改变这一行为，具体说明见[第 4.7 节](/04-cuda-features/lazy-loading.html#lazy-loading)。

## 3.5.3 功能扩展类特性（Functionality Features）

这里介绍的功能有一个共同特点：它们用于提供额外的能力或功能。

### 3.5.3.1 扩展 GPU 内存（Extended GPU Memory）

[扩展 GPU 内存](/04-cuda-features/extended-gpu-memory.html#extended-gpu-memory)适用于通过 NVLink-C2C 连接的系统，可以让 GPU 高效访问系统中的全部内存。[第 4.17 节](/04-cuda-features/extended-gpu-memory.html#extended-gpu-memory)详细介绍了 EGM。

### 3.5.3.2 动态并行（Dynamic Parallelism）

CUDA 应用程序最常见的做法是从 CPU 上运行的代码中启动 Kernel。不过，也可以从 GPU 上运行的 Kernel 中创建新的 Kernel 调用。这项功能称为 [CUDA dynamic parallelism](/04-cuda-features/cuda-dynamic-parallelism.html#cuda-dynamic-parallelism)。[第 4.18 节](/04-cuda-features/cuda-dynamic-parallelism.html#cuda-dynamic-parallelism)介绍了如何从 GPU 上运行的代码创建新的 GPU Kernel 启动，以及其中的详细机制。

## 3.5.4 CUDA 互操作性（CUDA Interoperability）

### 3.5.4.1 CUDA 与其他 API 的互操作性（CUDA Interoperability with other APIs）

除了 CUDA 之外，还有其他机制可以在 GPU 上运行代码。应用 GPU 最初是为加速计算机图形而构建的，因此图形应用拥有自己的一组 API，例如 Direct3D 和 Vulkan。应用程序可能希望使用某个图形 API 进行 3D 渲染，同时使用 CUDA 执行计算。CUDA 提供了在 CUDA context 与 3D API 使用的 GPU context 之间交换 GPU 上存储数据的机制。例如，应用程序可以使用 CUDA 执行仿真，然后使用 3D API 将结果制作成可视化内容。具体做法是，让某些缓冲区同时可被 CUDA 和图形 API 读取和/或写入。

用于与图形 API 共享缓冲区的相同机制，也用于与通信机制共享缓冲区；后者可以在多节点环境中实现快速、直接的 GPU 到 GPU 通信。

[第 4.19 节](/04-cuda-features/cuda-interoperability.html#cuda-interoperability)介绍了 CUDA 如何与其他 GPU API 互操作，以及如何在 CUDA 与其他 API 之间共享数据，并为多种不同 API 提供了具体示例。

### 3.5.4.2 进程间通信（Interprocess Communication）

对于规模非常大的计算，通常会将多个 GPU 组合使用，以获得更多内存和更多协同工作的计算资源。在单个系统中，或者按照集群计算术语称为单个节点时，可以在一个主机进程中使用多个 GPU。[第 3.4 节](/03-advanced-cuda/multiple-gpus.html#multi-gpu-introduction)介绍了这种方式。

也可以使用跨越单台或多台计算机的多个独立主机进程。当多个进程协同工作时，它们之间的通信称为进程间通信。CUDA 进程间通信（CUDA IPC）提供了在不同进程之间共享 GPU 缓冲区的机制。[第 4.15 节](/04-cuda-features/interprocess-communication.html#interprocess-communication)解释并演示了如何使用 CUDA IPC 在不同主机进程之间进行协调和通信。

## 3.5.5 细粒度控制（Fine-Grained Control）

### 3.5.5.1 虚拟内存管理（Virtual Memory Management）

如[第 2.6.1 节](/02-programming-gpus/unified-and-system-memory.html#memory-unified-virtual-address-space)所述，系统中的所有 GPU 与 CPU 内存共享一个统一的虚拟地址空间。大多数应用程序可以直接使用 CUDA 提供的默认内存管理，无需改变其行为。不过，[CUDA Driver API](/03-advanced-cuda/driver-api.html#driver-api)为有需要的应用提供了对虚拟内存空间布局的高级、细粒度控制。这主要适用于控制缓冲区在同一系统内以及跨多个系统的 GPU 之间共享时的行为。

[第 4.16 节](/04-cuda-features/virtual-memory-management.html#virtual-memory-management)介绍 CUDA Driver API 提供的控制项、这些控制项的工作方式，以及开发者可能从中获得优势的场景。

### 3.5.5.2 Driver 入口点访问（Driver Entry Point Access）

[Driver 入口点访问](/04-cuda-features/driver-entry-point-access.html#driver-entry-point-access)是指从 CUDA 11.3 开始获取 CUDA Driver API 和 CUDA Runtime API 函数指针的能力。它还允许开发者获取特定 Driver 函数变体的函数指针，并访问比 CUDA Toolkit 自带版本更新的 Driver 中提供的函数。[第 4.20 节](/04-cuda-features/driver-entry-point-access.html#driver-entry-point-access)介绍了 Driver 入口点访问。

### 3.5.5.3 错误日志管理（Error Log Management）

[错误日志管理](/04-cuda-features/error-log-management.html#error-log-management)提供了处理和记录 CUDA API 错误的工具。设置单个环境变量 `CUDA_LOG_FILE`，即可将 CUDA 错误直接捕获到 stderr、stdout 或文件中。错误日志管理还允许应用程序注册一个回调函数，在 CUDA 遇到错误时触发。[第 4.8 节](/04-cuda-features/error-log-management.html#error-log-management)提供了错误日志管理的更多细节。
