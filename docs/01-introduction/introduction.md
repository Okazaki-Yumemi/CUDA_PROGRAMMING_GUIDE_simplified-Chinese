---
title: 1.1 引言
description: GPU、CPU 与 CUDA 的中文翻译
---

# 1.1 引言（Introduction）

> 本页按 NVIDIA CUDA Programming Guide Release 13.3 的在线页面逐段翻译。原文页面：[1.1. Introduction](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html)；官方页面标注的更新时间为 2026-05-27。

## 1.1.1 图形处理器（The Graphics Processing Unit）

图形处理器（Graphics Processing Unit，GPU）最初是用于三维图形的专用处理器。它以固定功能硬件加速实时三维渲染中的并行操作。经过多代演进，GPU 变得越来越可编程；到 2003 年，图形流水线中的一些阶段已经完全可编程，可以为三维场景或图像的每个组成部分并行运行自定义代码。

2006 年，NVIDIA 引入计算统一设备架构（Compute Unified Device Architecture，CUDA），使任意计算工作负载都能够独立于图形 API 使用 GPU 的吞吐能力。

从那以后，CUDA 和 GPU 计算被用于加速几乎所有类型的计算工作负载：既包括流体动力学、能量传输等科学模拟，也包括数据库和分析等商业应用。此外，GPU 的能力与可编程性已经成为许多新算法和新技术发展的基础，应用范围从图像分类一直延伸到扩散模型和大语言模型等生成式人工智能。

## 1.1.2 使用 GPU 的收益（The Benefits of Using GPUs）

在相近的价格和功耗范围内，GPU 能提供远高于 CPU 的指令吞吐量和内存带宽。许多应用在 GPU 上运行时会比在 CPU 上快得多；可以参考 NVIDIA 的 [GPU Applications](https://www.nvidia.com/en-us/solutions/gpu-applications/) 页面。FPGA 等其他计算设备也具有很高的能效，但它们的编程灵活性远低于 GPU。

CPU 和 GPU 的设计目标不同。CPU 的目标是尽可能快地执行一串串行操作（称为一个线程），并且可以让几十个这样的线程并行运行；GPU 的目标则是让成千上万个线程并行执行，为此牺牲较低的单线程性能，以换取更高的总体吞吐量。

GPU 专门面向高度并行的计算，会把更多晶体管用于数据处理单元；CPU 则会把更多晶体管用于数据缓存和控制流。下图展示了 CPU 与 GPU 在芯片资源分配上的一种示例性差异。

![图 1：GPU 为数据处理投入更多晶体管](/images/chapter-01/figure-01-gpu-devotes-more-transistors.png)

<div class="figure-caption"><strong>图 1：GPU 为数据处理投入更多晶体管。</strong> 原 PDF 第 20 页。图示表达的是 CPU 与 GPU 的设计取舍：CPU 更强调缓存和控制流，GPU 更强调并行数据处理。</div>

这并不意味着 GPU 对每一种任务都更快。只有当工作能够拆分成大量并行任务，并且数据搬运、同步和启动开销可控时，GPU 的吞吐优势才容易体现。图 1 是概念模型，不是某一颗具体芯片的物理版图；实际性能仍应以目标程序和目标 GPU 上的测量结果为准。

## 1.1.3 快速开始（Getting Started Quickly）

利用 GPU 提供的计算能力有许多途径。本指南涵盖使用 C++ 等高级语言为 CUDA GPU 平台编程，但并不要求所有 GPU 加速应用都直接编写 GPU 代码。

不同领域已经有越来越多的算法和例程通过专用库提供出来。如果某个库已经实现了所需功能，尤其是 NVIDIA 提供的库，那么使用它通常比从头重新实现算法更高效，也更有可能获得更好的性能。`cuBLAS`、`cuFFT`、`cuDNN` 和 `CUTLASS` 只是其中几个例子，它们可以帮助开发者避免重新实现已经成熟的算法。

这些库还有一个额外好处：它们会针对不同 GPU 架构进行优化，从而在开发效率、性能和可移植性之间取得理想平衡。

此外，尤其是在人工智能领域，各类框架也提供了 GPU 加速的基础构件。许多这类框架正是通过调用前述 GPU 加速库来获得加速效果的。

领域专用语言（domain-specific language，DSL）也提供了比本指南所介绍的高级语言更高层次的 GPU 编程方式。例如，NVIDIA 的 Warp 或 OpenAI 的 Triton 都可以编译为直接运行在 CUDA 平台上的代码。

NVIDIA 的 [Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub) 提供了学习 GPU 和 CUDA 计算所需的资源、示例和教程。

## 本节导航

- [1.2 编程模型](./programming-model.html)：介绍 host/device、grid/block、warp/SIMT、tile 和 GPU 内存。
- [1.3 CUDA 平台](./cuda-platform.html)：介绍 compute capability、Toolkit、Driver、PTX、cubin 和 fatbin。
- [PDF 图版：图 1](../figures.html#图-1-gpu-为数据处理投入更多晶体管)：查看原 PDF 页码和图示解读。

<div class="translation-note">
  <strong>术语说明：</strong>GPU、CPU、CUDA、kernel、host、device、PTX、cubin、fatbin 等工程术语在中文段落中保留英文，以便与 API、编译器输出和 NVIDIA 官方文档交叉检索。
</div>
