---
title: 4.18 CUDA Dynamic Parallelism
description: CUDA 动态并行与 child grid
---

# 4.18 CUDA Dynamic Parallelism

CUDA Dynamic Parallelism 允许 device thread 配置并启动 child grid，适合工作递归或数据依赖只有在 device 端才确定的算法。它增加 device-side launch、嵌套同步和资源管理开销，必须明确 parent/child grid 的完成关系。
