---
title: 1.1 Introduction
description: GPU、CPU 与 CUDA 的中文翻译
---

# 1.1 Introduction

> 原文标题：**1.1. Introduction**。本页对应 NVIDIA 官方的 `01-introduction/introduction.html`，内容按项目内 PDF Release 13.3 翻译与解释。

## 图形处理器

GPU 起初是为三维图形设计的专用处理器，用固定功能硬件加速实时渲染中的并行操作。经过多代演进，GPU 逐渐变得可编程；到 2003 年，图形流水线中的部分阶段已经能够为场景组件或图像元素运行自定义并行代码。

2006 年，NVIDIA 引入 **Compute Unified Device Architecture（CUDA）**，让各种计算工作负载可以独立于图形 API 使用 GPU 的吞吐能力。此后，CUDA 和 GPU computing 被用于流体动力学、能量传输、数据库、分析、图像分类、扩散模型和大语言模型等领域。

这里的“通用计算”并不是把 CPU 的每个工作原样搬到 GPU，而是把工作拆成大量相似任务，使硬件能同时推进多个线程。

## 使用 GPU 的收益

在相近的价格和功耗范围内，GPU 通常拥有更高的指令吞吐量和内存带宽。CPU 更擅长尽快完成少量复杂线程；GPU 则牺牲一部分单线程性能，把更多晶体管投入数据处理单元，以并行执行成千上万个线程。

GPU 并非对所有任务都更快。分支多、任务粒度小、数据依赖强或数据搬运占主导的程序，可能无法兑现 GPU 的理论吞吐。真正的性能取决于并行度、内存访问、同步、启动开销和 CPU/GPU 是否能同时工作。

![PDF 第 20 页的图 1：GPU 为数据处理投入更多晶体管](/figures/page-020.png)

*PDF p.20，图 1。CPU 更多资源用于缓存和控制流，GPU 更多资源用于数据处理。图示是设计取舍的概念模型，不是某一颗具体芯片的版图。*

## 快速开始

使用 GPU 不一定要从手写 kernel 开始。NVIDIA 和社区提供的库通常已经针对不同 GPU 架构做过优化：

- `cuBLAS`、`cuFFT`、`cuDNN`、`CUTLASS` 等库覆盖线性代数、FFT、深度学习和矩阵运算；
- AI 框架通过调用 GPU 库提供更高层的张量操作；
- NVIDIA Warp、OpenAI Triton 等领域 DSL 可以把更高层的代码编译到 CUDA 平台；
- 需要定制数据布局、融合多个阶段或实现库未覆盖的操作时，再考虑写自己的 kernel。

库的价值不仅是少写代码，也包括针对架构的性能、成熟的边界处理和跨设备可移植性。学习 CUDA 时可以把“调用库”和“写 kernel”视为两种互补的工具。

## 从哪里继续

- [1.2 Programming Model：编程模型](./programming-model.html)：理解 host/device、grid/block、warp/SIMT、tile 和内存空间；
- [1.3 The CUDA platform：CUDA 平台](./cuda-platform.html)：理解 compute capability、Toolkit/Driver、PTX 和 fatbin；
- [PDF 图版：图 1](../figures.html#图-1-gpu-为数据处理投入更多晶体管)：查看图版解释和 PDF 页码。

<div class="translation-note">
  本页是学习向中文翻译，不替代 NVIDIA 的性能建议或产品规格。CPU/GPU 的对比是抽象模型，具体结果应以目标程序的 profiler 数据为准。
</div>
