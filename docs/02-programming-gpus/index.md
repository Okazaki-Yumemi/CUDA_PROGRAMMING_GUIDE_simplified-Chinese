---
title: 2. Programming GPUs in CUDA
description: 用 CUDA C++、Python、SIMT、tile 和异步执行编程 GPU
---

# 2. Programming GPUs in CUDA

本部分按 NVIDIA 官方网页的七个分节拆开。建议先读 [2.1 Intro to CUDA C++](./intro-to-cuda-cpp.html)，再根据任务进入 Python、SIMT、tile、异步、内存或编译器页面。

<div class="chapter-grid">
  <div class="chapter-card"><h3><a href="./intro-to-cuda-cpp.html">2.1 Intro to CUDA C++</a></h3><p>nvcc、kernel、三尖括号启动、内建索引、内存和错误检查。</p></div>
  <div class="chapter-card"><h3><a href="./intro-to-cuda-python.html">2.2 Intro to CUDA Python</a></h3><p>Python 生态、JIT kernel、数组、同步和数据驻留。</p></div>
  <div class="chapter-card"><h3><a href="./writing-simt-kernels.html">2.3 Writing SIMT Kernels</a></h3><p>线程层次、内存空间、合并访问、bank conflict、原子和 occupancy。</p></div>
  <div class="chapter-card"><h3><a href="./writing-tile-kernels.html">2.4 Writing Tile Kernels</a></h3><p>数组、tile space、load/store、tile 操作以及与 SIMT 的关系。</p></div>
  <div class="chapter-card"><h3><a href="./asynchronous-execution.html">2.5 Asynchronous Execution</a></h3><p>stream、event、CUDA Graph、异步复制和流水线。</p></div>
  <div class="chapter-card"><h3><a href="./unified-and-system-memory.html">2.6 Unified and System Memory</a></h3><p>统一虚拟地址、迁移、预取、系统差异和显式复制。</p></div>
  <div class="chapter-card"><h3><a href="./nvcc.html">2.7 NVCC</a></h3><p>host/device 编译、架构代码生成、fatbin、调试和构建检查。</p></div>
</div>

完整的长篇中文导读仍保留在[第二部分总览](../chapters/02-programming-gpus.html)，本页结构与官方网页对齐，便于按分节阅读。
