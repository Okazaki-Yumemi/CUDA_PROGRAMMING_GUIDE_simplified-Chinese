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
| 1. Introduction to CUDA | [第一部分：CUDA 简介](./01-introduction/introduction.html) | Introduction、Programming Model、The CUDA platform | 1.1–1.3 已按官方网页逐段翻译，含图 1–10 与脚注 |
| 2. Programming GPUs in CUDA | [第二部分：用 CUDA 编程 GPU](./02-programming-gpus/intro-to-cuda-cpp.html) | CUDA C++、CUDA Python、SIMT、Tile、异步、内存、nvcc | 2.1–2.7 已按官方网页逐节翻译，含代码、表格、图 11–21 与边界说明 |
| 3. Advanced CUDA | [第三部分：高级 CUDA](./03-advanced-cuda/advanced-apis-and-features.html) | 高级 API、kernel、Driver API、多 GPU、CUDA 功能导览 | 3.1–3.5 已按官方网页逐节翻译，含代码、表格、图 22–23 与 PDL 图示 |
| 4. CUDA Features | [第四部分：CUDA 功能](./04-cuda-features/unified-memory.html) | 20 个当前功能主题 | 4.1–4.20 已按官方网页逐节翻译，含代码、表格、图 24–57 与功能说明 |
| 5. Technical Appendices | [第五部分：技术附录](./05-technical-appendices/compute-capabilities.html) | Compute Capability、语言、浮点、设备端 API、模型 | 参考手册式的中文索引 |
| 6. Notices | [第六部分：声明](./06-notices/notice.html) | Notice、OpenCL、Trademarks | 中文说明与来源保留 |

## 推荐路线

### A. 第一次学习 CUDA

`异构系统 → grid/block/thread → warp/SIMT → global/shared/register → kernel launch → 同步与错误检查`

先读第一部分中的[编程模型](./01-introduction/programming-model.html)，再看第二部分的 [CUDA C++ 最小心智模型](./02-programming-gpus/intro-to-cuda-cpp.html)。

### B. 关注性能

`线程块调度 → warp divergence → 合并访问 → 共享内存 bank conflict → occupancy → 异步流水线`

推荐同时查看[图 14 的全局内存转置](./figures.html#图-14-全局内存中的矩阵转置)和[图 15 的共享内存 bank conflict](./figures.html#图-15-共享内存中的步长访问)。

### C. 查版本敏感功能

直接进入[CUDA 功能](./04-cuda-features/unified-memory.html)。每个条目都保留官方英文名称；页面中的“适用性提示”只做阅读导航，真正的 compute capability、驱动和 API 限制请以 NVIDIA 官方页面为准。

## 中文稿状态

<div class="status-grid">
  <div class="status-card"><h3>结构</h3><p>按官方 6 个顶层部分和在线目录的主要分节建立导航。</p></div>
  <div class="status-card"><h3>逐段翻译</h3><p>第一部分 1.1–1.3、第二部分 2.1–2.7、第三部分 3.1–3.5 与第四部分 4.1–4.20 已和官方网页逐段核对；第五、六部分按章节继续推进。</p></div>
  <div class="status-card"><h3>图版</h3><p>原 PDF 图 1–57 已按页保留并附中文解释，章节正文同时使用官方图示资源。</p></div>
  <div class="status-card"><h3>版本核对</h3><p>保留 Release 13.3 和官方在线目录入口，方便发现后续版本差异。</p></div>
</div>

<div class="translation-note">
  <strong>阅读边界：</strong>第一至第四部分已完成逐段翻译，第五、六部分仍按章节推进；未完成页面不会把摘要标成全文翻译。代码标识符、API 名称、公式和版本限制尽量保持原文；涉及生产环境或新硬件时，请将本文与官方文档并读。
</div>
