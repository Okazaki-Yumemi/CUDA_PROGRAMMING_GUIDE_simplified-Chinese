---
title: 1. Introduction to CUDA
description: CUDA Programming Guide 第一部分的中文分节入口
---

# 1. Introduction to CUDA

这部分先建立 CUDA 的共同语言，再进入具体编程语言。阅读顺序与 NVIDIA 官方网页一致：

1. [1.1 Introduction：引言](./introduction.html)
2. [1.2 Programming Model：编程模型](./programming-model.html)
3. [1.3 The CUDA platform：CUDA 平台](./cuda-platform.html)

<div class="source-note">
  官方页面采用“一个分节一个 URL”的组织方式。本项目也保留这组路径，便于从目录、浏览器历史和搜索结果直接定位到中文页面；PDF 页码仍以项目内 Release 13.3 原文件为准。
</div>

## 这一部分要回答的三个问题

- GPU 和 CPU 的设计目标有什么不同？
- kernel、grid、thread block、warp、SM 之间如何对应？
- CUDA Toolkit、Driver、PTX、cubin、fatbin 如何组成一个可运行的平台？

如果你已经熟悉 CUDA，可以直接跳到[编程模型](./programming-model.html)或[CUDA 平台](./cuda-platform.html)。
