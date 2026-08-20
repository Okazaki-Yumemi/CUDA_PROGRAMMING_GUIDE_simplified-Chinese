---
title: 第一部分：CUDA 简介
description: Introduction to CUDA 的中文翻译与编程模型导读
---

# 第一部分：CUDA 简介

> 原文部分：**1. Introduction to CUDA**。本页按 PDF 的 `1.1 Introduction`、`1.2 Programming Model`、`1.3 The CUDA platform` 组织。

## 1.1 引言

### 1.1.1 图形处理器

GPU 最初是为实时三维渲染设计的专用处理器，用固定功能硬件加速图形流水线中的并行操作。随着几代产品演进，GPU 逐渐变得可编程；到 2003 年，图形流水线中的部分阶段已经可以运行针对场景组件或图像元素编写的并行代码。

2006 年，NVIDIA 引入 **Compute Unified Device Architecture（CUDA）**，使通用计算工作负载可以独立于图形 API 使用 GPU 的吞吐能力。此后，CUDA/GPU computing 被用于流体动力学、能量传输、数据库、分析、图像分类、扩散模型和大语言模型等工作。

这里的关键转变不是“GPU 也能运行 C++”，而是：GPU 把硬件预算和功耗预算更多地投入到数据处理单元，从而同时推进大量相似的工作项。

### 1.1.2 使用 GPU 的收益

在相近的价格和功耗范围内，GPU 通常能提供比 CPU 更高的指令吞吐量与内存带宽。CPU 与 GPU 的设计目标不同：

| 维度 | CPU | GPU |
| --- | --- | --- |
| 优先目标 | 尽快完成少量复杂、串行或分支多的线程 | 同时推进成千上万个相似线程，追求总吞吐 |
| 晶体管侧重 | 缓存、分支预测、控制流和复杂单线程执行 | 数据处理单元、带宽和线程调度 |
| 擅长工作 | 低延迟、复杂控制流、操作系统和串行协调 | 规则的数据并行、矩阵/向量运算、图像与张量处理 |
| 典型限制 | 并行度和内存带宽有限 | 单线程延迟较高，分支和不规则访问需要谨慎设计 |

这不是说 GPU 必然更快。只有当工作可以拆分为大量相互独立或规则协作的任务，并且数据搬运、同步与启动开销可控时，GPU 的吞吐优势才容易兑现。PDF 第 20 页的[图 1：GPU 为数据处理投入更多晶体管](../figures.html#图-1-gpu-为数据处理投入更多晶体管)把这个设计取舍画了出来。

### 1.1.3 快速开始的路径

直接写 kernel 不是使用 GPU 的唯一方式。指南建议优先考虑已经针对 GPU 架构优化的库和框架，例如：

- `cuBLAS`：线性代数基本运算；
- `cuFFT`：快速傅里叶变换；
- `cuDNN`：深度学习相关的高性能原语；
- `CUTLASS`：可组合的矩阵乘法和线性代数实现；
- AI 框架：通常通过调用上述 GPU 库获得加速；
- DSL：例如 NVIDIA Warp 或 OpenAI Triton，把更高层的表达编译到 CUDA 平台。

使用库通常比从零实现成熟算法更节省时间，也更容易获得跨 GPU 架构的性能和可移植性。只有当现有库无法表达需求、需要定制数据布局，或需要把多个阶段融合成专门 kernel 时，才值得深入编写 CUDA kernel。

## 1.2 编程模型

CUDA 编程模型与具体语言分开描述。下面的术语同样适用于 CUDA C++、CUDA Python 以及能够生成 CUDA kernel 的其他工具。

### 1.2.1 异构系统

CUDA 假定系统由 CPU 与 GPU 共同组成：

- CPU 和直接连接到它的内存称为 **host** 与 **host memory**；
- GPU 和直接连接到它的内存称为 **device** 与 **device memory**；
- 在某些 SoC 中，CPU/GPU 和内存可能处于同一个封装；较大的系统也可能包含多个 CPU 或 GPU。

应用总是从 CPU 开始运行。host 代码可以通过 CUDA API 在 host memory 与 device memory 之间复制数据、启动 GPU 代码，并等待复制或 GPU 计算完成。CPU 与 GPU 可以同时执行代码；高性能程序通常会尽量让二者都保持有用工作。

在 GPU 上执行的代码叫 **device code**，被调用到 GPU 上运行的函数历史上称为 **kernel**，启动它的动作叫 **launching a kernel**。一次 kernel launch 可以理解为：让许多 GPU 线程并行执行同一段 kernel 代码。

### 1.2.2 GPU 硬件模型

为了编程，GPU 可以抽象为由多个 **Streaming Multiprocessor（SM）** 组成，并由 **Graphics Processing Cluster（GPC）** 组织。一个 SM 包含：

- 本地寄存器文件；
- 统一数据缓存；
- 执行算术、加载、存储和特殊功能的功能单元；
- 为 shared memory 与 L1 cache 提供物理资源的统一数据缓存。

统一数据缓存如何在 L1 cache 与 shared memory 之间分配，可以在运行时配置；不同 GPU 架构的内存大小和功能单元数量也可能不同。这个硬件模型是帮助理解资源和性能的抽象，并不承诺物理布局在不同代 GPU 上完全相同。只要程序遵守 CUDA 编程模型，底层调度和布局可以演进而不影响正确性。

PDF 第 22 页的[图 2：GPU 由许多 SM 组成](../figures.html#图-2-gpu-由-sm-组成)展示了 GPC、SM、功能单元、GPU memory 与 CPU 的关系。

### 1.2.2.1 线程块与网格

kernel 经常以数百万个线程启动。线程先组织成 **thread block（线程块）**，线程块再组织成 **grid（网格）**。同一个 grid 中的线程块具有相同的尺寸和维度。grid 和 block 都可以是一维、二维或三维，这使得线程坐标可以自然地映射到数组元素、图像像素或矩阵 tile。

启动 kernel 时，execution configuration 指定 grid 和 thread block 的维度，还可以包含 cluster size、stream、SM 配置等参数。kernel 内每个线程都可以通过内建变量得知自己在 block 和 grid 中的位置。

所有属于同一线程块的线程都在一个 SM 上执行，因此它们可以高效地使用 shared memory 互相交换数据，并通过 block 级同步原语协作。一个 grid 可能有数百万个 block，而 GPU 只有几十或几百个 SM；block 会在可用 SM 之间被调度，顺序不保证。

因此，通常不能让一个 block 依赖另一个 block 的中间结果，也不能假定不同 block 之间有隐式同步。CUDA 要求 grid 能够按任意顺序执行，既可以并行，也可以串行。这样同一个 kernel 才能在只有一个 SM 或拥有成千上万个 SM 的 GPU 上运行。

PDF 第 22 页的[图 3：线程块网格](../figures.html#图-3-线程块网格)和第 24 页的[图 4：SM 上的线程块调度](../figures.html#图-4-sm-上的线程块调度)分别解释了层次结构与调度不确定性。

### 1.2.2.1.1 线程块集群

从 compute capability 9.0 开始，CUDA 增加了可选的 **thread block cluster** 层级。cluster 是一组线程块，也可以是一维、二维或三维布局。指定 cluster 不会改变 grid 的维度，也不会改变 block 在 grid 中的索引；它只是把相邻 block 分组，并提供 cluster 级的通信和同步机会。

cluster 中的线程块会在同一个 GPC 内执行。通过 Cooperative Groups 提供的接口，cluster 内不同 block 的线程可以协作；它们还可以访问 cluster 中其他 block 的 shared memory，这种能力称为 **distributed shared memory**。cluster 最大尺寸取决于硬件，使用前必须查询目标设备支持的配置。

相关图版见[图 5：线程块集群](../figures.html#图-5-线程块集群)和[图 6：GPC 内的集群调度](../figures.html#图-6-gpc-内的集群调度)。

### 1.2.2.2 Warp 与 SIMT

线程块内部的线程以 32 个为一组组织成 **warp**。warp 中每个线程拥有一个从 0 到 31 的 warp lane。warp 以 **Single Instruction, Multiple Threads（SIMT）** 的方式推进：程序模型上，同一 warp 的线程执行同一条指令，但每个线程仍可以拥有自己的寄存器、数据和控制流状态。

当同一 warp 的线程遇到分支而选择不同路径时，硬件会让一条路径的线程执行，同时 mask 掉不在这条路径上的 lane；之后再处理另一条路径。这种现象叫 **warp divergence（warp 分歧）**。分歧不会自动让程序错误，但会减少一个时刻真正有用的 lane 数量。尽量让同一 warp 的线程走相同控制流，通常更有利于利用率。

SIMT 与 SIMD 不完全相同：SIMD 通常沿单一控制流操作固定宽度的数据，而 SIMT 允许线程拥有自己的控制流路径，也没有固定的“数据位宽”概念。写普通 CUDA 代码不必手动模拟 warp，但理解 warp 对解释全局内存合并访问、shared memory bank pattern 和高级 warp 专用优化很有帮助。

线程块可以使用任意线程数，但把总线程数设为 32 的倍数通常更合理，否则最后一个 warp 会包含长期不使用的 lane。不要把“warp 是 32”理解为硬件永远以某种固定物理方式执行；应依赖 CUDA 编程模型和公开的同步/内存语义，而不是依赖未承诺的硬件细节。

见[图 7：warp 分歧与掩码线程](../figures.html#图-7-warp-分歧)。

### 1.2.2.3 CUDA 中的 Tile 编程

tile 是把一组相邻数据元素视为一个整体的编程抽象。线程块中的线程可以共同加载、变换和计算 tile；例如矩阵乘法可以把大矩阵拆成能放入 shared memory 的小块，再让线程协同完成块内乘加。

tile 编程与 SIMT 并不冲突：每个线程依然执行 kernel，但 tile 的加载、布局和算子以一组线程的集合视角表达。使用 tile 时要明确三件事：tile 在逻辑空间中的形状、数据从哪个内存空间搬运到哪个空间，以及哪些线程负责边界元素。

## 1.2.3 GPU 内存

### 1.2.3.1 异构系统中的 DRAM

host memory 和 device memory 可能位于不同的 DRAM 中。GPU 计算通常要面对数据如何到达 GPU、什么时候回传 host、以及复制能否与计算重叠等问题。统一内存和可访问的系统内存可以简化部分数据移动，但不意味着所有访问都拥有相同的延迟和带宽。

### 1.2.3.2 GPU 片上内存

每个 SM 有自己的寄存器文件和 shared memory；GPU 还可能有 L1、L2 等缓存。片上内存通常比设备 DRAM 延迟低，但容量也小，并且会受到寄存器/共享内存使用量、bank 访问模式和线程块驻留的共同影响。

写 kernel 时应先把数据生命周期和复用关系说清楚，再决定使用寄存器、shared memory、constant memory、texture/cache 或 global memory。只看“哪个空间最快”很容易误导，因为访问模式、并发度和同步成本同样重要。

### 1.2.3.3 统一内存

统一内存提供统一的地址空间和更高层的数据迁移/一致性抽象。它能显著降低管理 host/device 指针和复制时机的负担，但页面迁移、并发访问、预取和系统一致性仍会影响性能。使用统一内存时，必须了解目标操作系统、GPU、互连和 compute capability 的支持模式。

## 1.3 CUDA 平台

### 1.3.1 Compute Capability 与 SM 版本

CUDA 用 compute capability 标识一组硬件能力。它不只代表“快或慢”，还决定某些指令、内存空间、cluster、异步拷贝或原子操作是否可用。程序可以在运行时查询设备属性，再选择 kernel 配置和功能路径。

### 1.3.2 CUDA Toolkit 与 NVIDIA Driver

CUDA Toolkit 包含编译器、头文件、库和工具；NVIDIA Driver 负责让应用与 GPU 设备交互。CUDA Runtime API 提供相对高层的使用方式，CUDA Driver API 提供更底层的上下文、模块和函数管理。两套 API 可以互操作，但资源所有权和错误处理边界要保持清晰。

### 1.3.3 Parallel Thread Execution（PTX）

PTX 是 CUDA 平台中的虚拟指令集和中间表示。高层 CUDA 代码可以编译到 PTX，之后由驱动在目标 GPU 上进行即时编译（JIT），也可以在构建时生成面向具体架构的 cubin。PTX 的存在帮助程序在一定范围内跨代 GPU 保持可移植性，但不能保证新架构上的性能一定相同。

### 1.3.4 Cubin 与 Fatbin

cubin 是针对具体 GPU 架构的二进制代码；fatbin 是可以容纳多个 GPU 代码变体的容器。一个可执行文件或库通常同时包含 CPU 代码和 GPU fatbin。fatbin 中可以同时存放多个 cubin 与 PTX，运行时选择最合适的代码，必要时对 PTX 做 JIT。

兼容性有两层：

1. **Binary compatibility**：已经编译好的机器码是否能直接在目标设备执行；
2. **PTX compatibility**：目标驱动能否把 PTX 编译成目标设备的机器码。

构建发布包时，应根据支持的 GPU 架构规划 `-gencode` 等选项，并权衡包体积、覆盖范围和首次 JIT 时间。完整图示见[图 10：可执行文件中的 fatbin 与 PTX](../figures.html#图-10-fatbin-与-ptx)。

## 本部分小结

CUDA 的核心心智模型可以压缩为一条链：

`host 启动 kernel → grid 组织 blocks → block 驻留在 SM → warp 以 SIMT 推进 → 线程通过不同内存空间读写数据 → 用同步和错误检查保证正确性`

接下来进入[第二部分：用 CUDA 编程 GPU](./02-programming-gpus.html)，把这些抽象落到 CUDA C++、CUDA Python、内存访问和异步执行上。
