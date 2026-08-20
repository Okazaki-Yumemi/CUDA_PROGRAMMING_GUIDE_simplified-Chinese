---
title: 2.4 Writing Tile Kernels
description: Tile kernel、数组、tile space 和数据移动
---

# 2.4 Writing Tile Kernels

tile programming 让程序员以整个 thread block 的视角描述多维数据块，编译器再把 tile 操作映射到线程。它与 SIMT 使用相同的 SM、block、grid 和 device memory，但抽象层次更高。

## Kernel and Function Declarations

tile kernel 仍然在 grid 的 block 上执行；程序通常指定 grid 维度，线程数由编译器根据 tile 操作决定。要把“执行单位 block”和“数据单位 tile”分开理解：一个 block 可以创建多个不同形状的 tile。

## Arrays and Tiles

array 是存储在 device memory 中、可通过 store 修改的多维容器；tile 是仅存在于 tile code、属于单个 block 的多维值集合。tile 通常是不可变的，每个操作会产生新 tile，编译器可以选择用寄存器、shared memory 或其他 SM 资源存储。

tile 每一维通常需要编译期可知并满足特定形状约束，不能把它当作普通运行时指针传给 kernel。

## Tile Space and Data Movement

load/store 把 array 与 tile 连接起来。tile space 是把 array 概念上切成等大小、不重叠 tile 的坐标空间；边界 tile 可用零填充等策略处理越界元素，store 越出边界的写入可被丢弃。gather/scatter 则允许更任意的位置访问。

## Operations on Tiles

tile 操作包括逐元素算术、矩阵乘、沿维归约、reshape、transpose 和类型转换。不同形状的 tile 组合时可能触发广播或形状扩展，必须理解其语义和资源代价。

## Relationship to SIMT Programming

tile kernel 与 SIMT kernel 可以在同一应用中共存，并读写同一 device memory。SIMT 适合需要细粒度线程控制的算法；tile 适合让编译器管理线程映射和块级数据操作的算法。选择是每个 kernel 的决定，不是整个项目只能选一种模型。
