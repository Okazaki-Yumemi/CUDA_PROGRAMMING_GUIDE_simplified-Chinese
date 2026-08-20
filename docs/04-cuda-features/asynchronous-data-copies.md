---
title: 4.11 Asynchronous Data Copies
description: 异步数据复制和 TMA
---

# 4.11 Asynchronous Data Copies

异步数据复制可让 host/device、global/shared 或其他支持的内存路径与计算重叠。是否真正异步取决于对齐、粒度、复制引擎、目标架构和同步语义，不能只根据函数名推断。

TMA 等硬件路径还要求正确的描述符、tile 形状、共享内存布局和 barrier 配合。
