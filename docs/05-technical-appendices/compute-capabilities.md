---
title: 5.1 Compute Capabilities
description: CUDA compute capability 和资源限制
---

# 5.1 Compute Capabilities

本附录按 compute capability/SM 版本列出硬件特性、限制、内存和执行资源。block size、寄存器、shared memory、cluster、原子操作和指令支持等问题，都应根据目标设备在这里核对。

compute capability 是能力集合，不是简单性能分数；程序应查询设备属性并为不支持的功能提供回退。
