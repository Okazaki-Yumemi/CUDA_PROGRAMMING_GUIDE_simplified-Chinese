---
title: 阅读地图
description: CUDA Programming Guide Release 13.3 的中文目录与阅读路径
---

# 阅读地图

<div class="source-note">
  本页按 NVIDIA 官方在线目录和项目内 PDF 的顶层结构整理。官方在线目录最后更新日期为 2026-05-27；本地 PDF 封面同样标注 Release 13.3。点击分节进入中文译稿与概念说明。
</div>

## 原 guide 的六个部分

| 原文部分 | 中文入口 | 官方范围 | 本站内容 |
| --- | --- | --- | --- |
| 1. Introduction to CUDA | [第一部分：CUDA 简介](./chapters/01-introduction.html) | Introduction、Programming Model、The CUDA platform | 核心概念翻译、线程层次和关键图版 |
| 2. Programming GPUs in CUDA | [第二部分：用 CUDA 编程 GPU](./chapters/02-programming-gpus.html) | CUDA C++、CUDA Python、SIMT、Tile、异步、内存、nvcc | 代码导读、编程模型的落地方式和性能注意事项 |
| 3. Advanced CUDA | [第三部分：高级 CUDA](./chapters/03-advanced-cuda.html) | 高级 API、kernel、Driver API、多 GPU、CUDA 功能导览 | 高级主题的中文索引和使用场景 |
| 4. CUDA Features | [第四部分：CUDA 功能](./chapters/04-cuda-features.html) | 20 个当前功能主题 | 每个主题的中文标题、作用和官方检索词 |
| 5. Technical Appendices | [第五部分：技术附录](./chapters/05-technical-appendices.html) | Compute Capability、语言、浮点、设备端 API、模型 | 参考手册式的中文索引 |
| 6. Notices | [第六部分：声明](./chapters/06-notices.html) | Notice、OpenCL、Trademarks | 中文说明与来源保留 |

## 推荐路线

### A. 第一次学习 CUDA

`异构系统 → grid/block/thread → warp/SIMT → global/shared/register → kernel launch → 同步与错误检查`

先读第一部分中的[异构系统](./chapters/01-introduction.html#1-2-1-异构系统)、[线程块与网格](./chapters/01-introduction.html#1-2-2-1-线程块与网格)、[warp 与 SIMT](./chapters/01-introduction.html#1-2-2-2-warp-与-simt)，再看第二部分的 [CUDA C++ 最小向量加法](./chapters/02-programming-gpus.html#2-1-4-一个可运行的最小心智模型)。

### B. 关注性能

`线程块调度 → warp divergence → 合并访问 → 共享内存 bank conflict → occupancy → 异步流水线`

推荐同时查看[图 14 的全局内存转置](./figures.html#图-14-全局内存中的矩阵转置)和[图 15 的共享内存 bank conflict](./figures.html#图-15-共享内存中的步长访问)。

### C. 查版本敏感功能

直接进入[CUDA 功能](./chapters/04-cuda-features.html)。每个条目都保留官方英文名称；页面中的“适用性提示”只做阅读导航，真正的 compute capability、驱动和 API 限制请以 NVIDIA 官方页面为准。

## 中文稿状态

<div class="status-grid">
  <div class="status-card"><h3>结构</h3><p>按官方 6 个顶层部分和在线目录的主要分节建立导航。</p></div>
  <div class="status-card"><h3>核心正文</h3><p>第一部分与第二部分的核心概念、示例和性能心智模型已写入中文稿。</p></div>
  <div class="status-card"><h3>图版</h3><p>选取原 PDF 中的代表性图版，保留 PDF 页码并附中文解释。</p></div>
  <div class="status-card"><h3>版本核对</h3><p>保留 Release 13.3 和官方在线目录入口，方便发现后续版本差异。</p></div>
</div>

<div class="translation-note">
  <strong>阅读边界：</strong>这是一个持续扩展的中文学习站，不把摘要冒充逐句翻译。代码标识符、API 名称、公式和版本限制尽量保持原文；涉及生产环境或新硬件时，请将本文与官方文档并读。
</div>
