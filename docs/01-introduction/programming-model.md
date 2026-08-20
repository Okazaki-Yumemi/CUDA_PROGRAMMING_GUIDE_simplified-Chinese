---
title: 1.2 Programming Model
description: CUDA 异构系统、GPU 硬件模型、线程层次、SIMT、tile 与内存的中文翻译
---

# 1.2 Programming Model

> 原文标题：**1.2. Programming Model**。本节与具体语言分开介绍 CUDA；后续 C++、Python 和其他工具都会使用这些术语。

## 异构系统

CUDA 假定系统同时包含 CPU 与 GPU。CPU 和直接连接到它的内存称为 **host** 与 **host memory**；GPU 和直接连接到它的内存称为 **device** 与 **device memory**。在 SoC 中，二者可能位于同一个封装；更大的系统则可能拥有多个 CPU 或 GPU。

应用总是从 CPU 开始。host code 可以通过 CUDA API 在 host memory 与 device memory 之间复制数据、启动 GPU 代码，并等待复制或 GPU 代码完成。CPU 与 GPU 可以同时执行；高性能程序通常会尽量提高二者的利用率。

GPU 上运行的代码称为 **device code**，被调用到 GPU 上执行的函数称为 **kernel**，启动它称为 **launching the kernel**。一次 kernel launch 可以理解为让大量线程并行执行 kernel。

## GPU 硬件模型

从 CUDA 编程角度看，GPU 可以抽象为许多 **Streaming Multiprocessors（SM）**，SM 又由 **Graphics Processing Clusters（GPC）** 组织。每个 SM 包含本地寄存器文件、统一数据缓存和执行计算的功能单元；统一数据缓存为 shared memory 与 L1 cache 提供物理资源。

硬件布局和执行方式可能随着架构变化。程序应依赖 CUDA 编程模型保证的语义，而不是依赖未公开承诺的物理细节。

![PDF 第 22 页的图 2：GPU、GPC、SM 与 CPU 的关系](/figures/page-022.png)

*PDF p.22，图 2。GPU 由多个 GPC 组成，GPC 组织多个 SM；CPU/GPU 通过 PCIe 或 NVLink 等互连连接。*

## 线程块与网格

kernel 往往以大量线程启动。线程组织成 **thread block（线程块）**，线程块再组织成 **grid（网格）**。grid 和 block 都可以是一维、二维或三维，这便于把线程坐标映射到数组、图像或矩阵。

启动配置指定 grid 和 block 的尺寸，也可以指定 cluster、stream 和 SM 配置。线程通过 `threadIdx`、`blockIdx`、`blockDim`、`gridDim` 得到自己的坐标和整体尺寸。

同一个 block 的所有线程在一个 SM 上执行，并可使用 shared memory 与 block 级同步协作。不同 block 的调度顺序没有保证，不能默认 block 之间有同步或先后关系。要让 grid 可扩展到不同规模的 GPU，通常要求不同 block 之间没有未处理的数据依赖。

![PDF 第 22 页的图 3：线程块网格](/figures/page-022.png)

*PDF p.22，图 3。箭头表示线程，数量不代表真实规模。网格中的 block 具有相同形状，但会被独立调度。*

![PDF 第 24 页的图 4：SM 上的线程块调度](/figures/page-024.png)

*PDF p.24，图 4。同一 SM 可同时驻留多个 block，但来自 grid 的 block 被分配到 SM 的顺序不保证。*

## Thread Block Clusters

compute capability 9.0 及以上的 GPU 可使用可选的 **thread block cluster** 层级。cluster 把相邻 block 分组，并提供 cluster 级通信与同步；同一 cluster 的 block 在一个 GPC 内执行，还可以通过 distributed shared memory 访问 cluster 中其他 block 的 shared memory。

cluster 不改变 grid 的尺寸和 block 在 grid 中的索引。最大 cluster size 取决于硬件，程序应查询设备能力并准备回退路径。

![PDF 第 25 页的图 5：cluster 中的线程块](/figures/page-025.png)

![PDF 第 26 页的图 6：GPC 内的 cluster 调度](/figures/page-026.png)

## Warps 与 SIMT

线程块内的线程以 32 个为一组组织成 **warp**。warp 按 **Single-Instruction, Multiple-Threads（SIMT）** 模型执行：同一 warp 的线程执行同一段 kernel 代码，但不同线程可以选择不同分支。

如果一个 warp 的线程走不同的控制流路径，硬件会在执行一条路径时 mask 掉不参与的 lane，这叫 **warp divergence**。分歧并非语法错误，但会降低同时执行有效工作的 lane 数。线程块总线程数设为 32 的倍数通常更合适，避免最后一个 warp 长期包含无效 lane。

![PDF 第 27 页的图 7：warp 分歧](/figures/page-027.png)

*PDF p.27，图 7。只有偶数线程进入 `if` 体，其他 lane 在这条路径上被 mask。理解这一点有助于解释分支和利用率，但不要依赖未承诺的物理执行细节。*

## Tile Programming

CUDA 还支持 tile programming。SIMT 模型让程序员编写逐线程代码；tile 模型让程序员以整个 block 的视角描述多维 tile，编译器再把 tile 操作映射到 block 内线程。

tile 与 block 不是同一概念：block 是执行单位，tile 是数据单位；一个 block 可以生成并操作多个不同形状或类型的 tile。tile 通常在编译期确定形状，数据在 array 与 tile 之间通过 load/store 移动，边界元素可以按规则填充或丢弃。

tile programming 与 SIMT 共存，不是后者的替代品。需要细粒度线程控制时仍应使用 SIMT；想让编译器负责线程映射、数据块形状和部分协作时，tile 抽象可以减少样板代码。

## GPU Memory

### DRAM Memory in Heterogeneous Systems

GPU 直接连接的 DRAM 从 device code 角度称为 global memory；CPU 连接的 DRAM 称为 system/host memory。多 GPU 系统中，每张 GPU 通常拥有自己的内存。CUDA 提供分配、复制和控制数据 locality 的 API，统一内存则允许运行时或硬件自动处理部分放置/迁移。

### On-Chip Memory in GPUs

每个 SM 有寄存器文件和 shared memory，线程可以快速访问。寄存器保存线程私有变量；shared memory 可被 block 或 cluster 内线程用于数据交换。寄存器、shared memory 和 L1 cache 的容量有限，并且由 block 内线程共同消耗；寄存器或 shared memory 需求过大可能使 block 无法启动或减少同时驻留的 block 数。

GPU 还有 L1、L2 和 constant cache。L1 属于 SM 的统一数据缓存，L2 由 GPU 中的多个 SM 共享；constant cache 用于缓存 kernel 生命周期内不变的 constant 数据和部分 kernel 参数。

### Unified Memory

显式分配的 CPU/GPU 内存默认只能由对应处理器直接访问，程序通过 CUDA API 复制数据。统一内存允许 CPU 与 GPU 访问同一分配，运行时或硬件在需要时迁移/安排数据。它简化管理但不等于零成本；最佳性能仍然要求尽量减少迁移，让数据由直接连接它的处理器访问。

## 下一节

继续阅读[1.3 The CUDA platform](./cuda-platform.html)，了解 compute capability、Toolkit、Driver、PTX、cubin 与 fatbin。
