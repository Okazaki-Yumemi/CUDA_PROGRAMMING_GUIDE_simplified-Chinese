---
home: true
title: CUDA 编程指南中文图解
heroText: CUDA 编程指南中文图解
tagline: 对照 NVIDIA CUDA Programming Guide Release 13.3 的中文阅读站
actions:
  - text: 从阅读地图开始
    link: /guide-index.html
    type: primary
  - text: 查看 PDF 图版
    link: /figures.html
    type: secondary
features:
  - title: 以 PDF 13.3 为主线
    details: 章节名称、顺序和图版页码以项目内的 cuda-programming-guide.pdf 为准，并用 NVIDIA 官方在线目录做版本校对。
  - title: 中文解释优先
    details: 保留 CUDA、warp、SM、PTX、API 等技术名词，围绕“它是什么、为什么这样设计、代码中怎么用”组织译文和导读。
  - title: 图版不只贴图
    details: 每张选取的原 PDF 图版都标注 PDF 页码，并附上图中层级、数据流和实际编程含义的中文说明。
---

<div class="source-note">
  <strong>版本声明：</strong>项目内 PDF 的封面标注为 Release 13.3，日期为 2026-05-27。本网站不是 NVIDIA 官方翻译，适合学习和检索；遇到版本敏感的 API、硬件限制或合规问题，请回到官方文档核对。
</div>

## 六个入口

<div class="chapter-grid">
  <div class="chapter-card"><span class="badge">Part 1</span><h3><a href="./01-introduction/introduction.html">CUDA 简介</a></h3><p>异构系统、GPU 硬件模型、线程层次、SIMT、内存与 CUDA 平台。</p></div>
  <div class="chapter-card"><span class="badge">Part 2</span><h3><a href="./02-programming-gpus/intro-to-cuda-cpp.html">用 CUDA 编程 GPU</a></h3><p>CUDA C++、CUDA Python、SIMT kernel、tile kernel、异步执行和 nvcc。</p></div>
  <div class="chapter-card"><span class="badge">Part 3</span><h3><a href="./03-advanced-cuda/advanced-apis-and-features.html">高级 CUDA</a></h3><p>高级 API、多 GPU、驱动 API，以及从功能角度浏览 CUDA。</p></div>
  <div class="chapter-card"><span class="badge">Part 4</span><h3><a href="./04-cuda-features/unified-memory.html">CUDA 功能</a></h3><p>统一内存、CUDA Graphs、协作组、异步拷贝、虚拟内存等主题索引。</p></div>
  <div class="chapter-card"><span class="badge">Part 5</span><h3><a href="./05-technical-appendices/compute-capabilities.html">技术附录</a></h3><p>5.1–5.8 的 Compute Capability、语言扩展、浮点计算、设备端 API 与内存模型完整译文。</p></div>
  <div class="chapter-card"><span class="badge">Part 6</span><h3><a href="./06-notices/notice.html">声明与商标</a></h3><p>6.1–6.3 的 Notice、OpenCL、Trademarks 中文译文与法律版本边界。</p></div>
</div>

## 阅读建议

1. 第一次接触 CUDA：先看[1.1 Introduction](./01-introduction/introduction.html)，再看[2.1 CUDA C++](./02-programming-gpus/intro-to-cuda-cpp.html)。
2. 想理解性能：重点看线程块调度、warp divergence、全局内存合并访问、共享内存 bank conflict 和占用率。
3. 想查新功能：直接看[第四部分：CUDA 功能](./04-cuda-features/unified-memory.html)，每个条目都保留官方英文名，便于回到 NVIDIA 文档搜索。
4. 想确认图形含义：进入[PDF 图版](./figures.html)，图下说明会把“图上画了什么”和“写 CUDA 时意味着什么”分开写。

## 翻译约定

- API、宏、内建变量、编译选项、类型名和代码标识符保留英文，例如 `threadIdx.x`、`cudaMalloc`、`__global__`。
- “thread block”译为“线程块”，“grid”译为“网格”，“warp”保留英文并在首次出现时说明为 32 个线程的执行组。
- “device”在 CUDA 语境译为“设备（GPU）”，“host”译为“主机（CPU）”；不要把 device memory 与统一内存混为一谈。
- 公式、代码和限制条件优先保留原意；中文段落是学习向翻译/导读，不替代 NVIDIA 的版本化 API 参考。
