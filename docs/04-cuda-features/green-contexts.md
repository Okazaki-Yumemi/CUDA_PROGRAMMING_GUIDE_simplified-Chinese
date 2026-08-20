---
title: 4.6 Green Contexts
description: Green Context 资源分区和尾延迟
---

# 4.6 Green Contexts

Green Context 是关联到一组 GPU 资源的轻量上下文，当前重点是 SM 和工作队列资源。静态分区可能让延迟敏感任务更早开始，但会牺牲其他任务的共享资源，应通过 Nsight Systems timeline 评估。
