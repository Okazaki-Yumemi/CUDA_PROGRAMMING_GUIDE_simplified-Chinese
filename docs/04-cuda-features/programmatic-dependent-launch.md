---
title: 4.5 Programmatic Dependent Launch and Synchronization
description: 程序化依赖启动与同步
---

# 4.5 Programmatic Dependent Launch and Synchronization

程序化依赖启动允许依赖的 secondary kernel 在 primary kernel 完全结束前按依赖关系启动，减少不必要的等待。使用前要明确两个 kernel 的数据依赖、内存可见性以及目标 GPU/Toolkit 是否支持该机制。
