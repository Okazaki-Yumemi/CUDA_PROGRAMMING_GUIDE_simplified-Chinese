---
title: 4.14 Memory Synchronization Domains
description: 内存同步域
---

# 4.14 Memory Synchronization Domains

内存同步域让同步/栅栏流量限制在更合适的域中，减少无关工作之间的干扰。先确定数据依赖需要的作用域，再选择同步域；不能用更小的域替代真实的可见性需求。
