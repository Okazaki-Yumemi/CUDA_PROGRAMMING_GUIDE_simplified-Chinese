---
title: 4.16 Virtual Memory Management
description: CUDA 虚拟内存管理
---

# 4.16 Virtual Memory Management

VMM 将虚拟地址保留、物理内存分配、映射和访问权限设置拆成独立步骤。它适合稀疏地址空间、共享映射和自定义内存管理器，但复杂度高于普通 allocation。

流程图见[PDF 图 55：VMM 使用概览](../figures.html#图-55-vmm-使用概览)。
