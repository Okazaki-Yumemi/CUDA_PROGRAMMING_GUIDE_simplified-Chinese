---
title: 第五部分：技术附录
description: CUDA Programming Guide 技术附录的中文索引
---

# 第五部分：技术附录

> 原文部分：**5. Technical Appendices**。附录适合在写代码、查限制或核对架构差异时按关键词检索，不建议第一次学习时从头通读。

## 5.1 Compute Capabilities：计算能力

列出不同 compute capability/SM 版本的硬件特性、限制、内存和执行资源。block size、寄存器、shared memory、cluster、原子操作和指令支持等问题，都应回到这里按目标设备核对。

## 5.2 CUDA Environment Variables：CUDA 环境变量

记录影响设备选择、调试、日志、加载、内存和运行时行为的环境变量。环境变量适合诊断和部署配置，不应在应用正确性上依赖开发机偶然设置的值。

## 5.3 C++ Language Support：C++ 语言支持

说明 CUDA C++ 对标准版本、编译器特性、模板、异常、RTTI、Lambda 和其他语言能力的支持边界。host 端和 device 端的支持可能不同，编译器也可能要求特定的限制。

## 5.4 C/C++ Language Extensions：C/C++ 语言扩展

解释 `__global__`、`__device__`、`__host__`、`__shared__`、`__constant__` 等声明符，以及 kernel launch、内建变量、向量类型和其他 CUDA 语法扩展。

## 5.5 Floating-Point Computation：浮点计算

覆盖浮点格式、normal/subnormal、特殊值、舍入、FMA、精度、IEEE-754 和 CUDA 数学函数。浮点结果的 bit-level 差异可能来自运算顺序、融合、fast math、架构和 host/device 编译路径。

## 5.6 Device-Callable APIs and Intrinsics：设备可调用 API 与内建函数

列出可在 device 代码中调用的 API、intrinsic、memory barrier、pipeline、Cooperative Groups、CUDA Device Runtime 等接口。查 API 时要确认调用方是 host 还是 device，以及所需的 compute capability。

## 5.7 CUDA C++ Memory Model：CUDA C++ 内存模型

说明线程作用域、同步原语、原子性、数据竞争和消息传递的语义。判断一段并发代码是否正确时，要同时问：写入是否原子？何时可见？哪个线程作用域需要同步？不同线程是否可能读到未发布的数据？

## 5.8 CUDA C++ Execution model：CUDA C++ 执行模型

描述 host thread、device thread、CUDA API、依赖和执行环境之间的关系。它适合解释为何 kernel launch 是异步的、为何不同 stream 可以重叠，以及为什么同步点和错误检查要放在正确的位置。

## 附录使用顺序

1. 遇到编译错误：先看 5.3 和 5.4；
2. 遇到精度差异：看 5.5；
3. 遇到越界、竞态或同步问题：看 5.7 和 5.8；
4. 遇到架构/资源限制：看 5.1；
5. 遇到函数是否能在 kernel 中调用：看 5.6。

附录的英文名称会保留在本页和代码注释中，方便在[NVIDIA 官方目录](https://docs.nvidia.com/cuda/cuda-programming-guide/contents.html)中跳转到对应原文。
