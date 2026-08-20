---
title: 4.7 Lazy Loading
description: CUDA 模块和 kernel 的惰性加载
---

# 4.7 Lazy Loading

Lazy Loading 把模块或 kernel 的一部分加载推迟到第一次使用，降低启动时开销。代价是首次调用延迟和更晚暴露的加载错误；延迟敏感服务可以通过预热或显式加载控制行为。
