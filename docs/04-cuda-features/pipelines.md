---
title: 4.10 Pipelines
description: CUDA 异步流水线
---

# 4.10 Pipelines

Pipeline 把异步复制、提交、等待和计算分成可重复的阶段。双缓冲是常见形式：一个 buffer 计算时另一个搬运；切换由 barrier、event 或 pipeline API 建立可见性和顺序。
