---
title: 4.14 内存同步域
description: 使用 CUDA 内存同步域隔离同步流量并降低栅栏干扰
---

# 4.14 内存同步域（Memory Synchronization Domains）

## 4.14.1 内存栅栏干扰（Memory Fence Interference）

一些 CUDA 应用可能会因为内存栅栏或刷新操作等待了超出 CUDA 内存一致性模型要求的事务而出现性能下降。

下面的示例包含三个作用域不同的对象：

```cpp
__managed__ int x = 0;
__device__  cuda::atomic<int, cuda::thread_scope_device> a(0);
__managed__ cuda::atomic<int, cuda::thread_scope_system> b(0);
```

| 线程 1（SM） | 线程 2（SM） | 线程 3（CPU） |
| --- | --- | --- |
| ```cpp
x = 1;
a = 1;
``` | ```cpp
while (a != 1) ;
assert(x == 1);
b = 1;
``` | ```cpp
while (b != 1) ;
assert(x == 1);
``` |

考虑上面的示例。CUDA 内存一致性模型保证断言条件为真，因此线程 1 对 `x` 的写入必须在线程 2 对 `b` 写入之前对线程 3 可见。

`a` 的 release/acquire 所提供的内存排序只足以让 `x` 对线程 2 可见，因为它是设备作用域操作。因此，`b` 的系统作用域排序不仅需要保证线程 2 自身发出的写入对线程 3 可见，还需要保证线程 2 可见的其他线程写入也对线程 3 可见。这称为累积性（cumulativity）。GPU 在执行时无法知道哪些写入在源代码层面已经被保证可见、哪些写入只是由于时序巧合而可见，因此必须对正在传输的内存操作采取保守而宽泛的处理范围。

这有时会导致干扰：由于 GPU 等待了源代码层面并不要求等待的内存操作，栅栏或刷新可能比实际必要的时间更长。

请注意，栅栏既可能像上例那样由代码中的 intrinsic 或原子操作显式产生，也可能在任务边界上为实现 *synchronizes-with* 关系而隐式产生。

一个常见例子是：某个 kernel 在 GPU 本地内存中执行计算，同时另一个并行 kernel（例如来自 NCCL）与对端执行通信。完成时，本地 kernel 会隐式刷新其写入，以满足下游工作所需的 *synchronizes-with* 关系。这可能会不必要地等待通信 kernel 较慢的 NVLink 或 PCIe 写入，全部等待或等待其中一部分。

## 4.14.2 使用域隔离流量（Isolating Traffic with Domains）

从计算能力 9.0（Hopper 架构）GPU 和 CUDA 12.0 开始，内存同步域功能提供了一种缓解这类干扰的方法。GPU 得到代码显式协助后，可以缩小一次栅栏需要处理的总体范围。每次 kernel 启动都会获得一个域 ID；写入和栅栏都会带有该 ID，而栅栏只会对与其域匹配的写入排序。在并行计算与通信的例子中，可以将通信 kernel 放入不同的域。

使用域时，代码必须遵守以下规则：**同一 GPU 上不同域之间的排序或同步需要使用系统作用域栅栏**。在同一域内，设备作用域栅栏仍然足够。这是累积性所必需的，因为一个 kernel 的写入不会被另一个域中发出的栅栏涵盖。本质上，跨域流量需要提前刷新到系统作用域，从而满足累积性。

请注意，这会修改 `thread_scope_device` 的定义。不过，由于 kernel 默认使用域 0（如下文所述），因此仍然保持向后兼容。

## 4.14.3 在 CUDA 中使用域（Using Domains in CUDA）

域通过新的启动属性 `cudaLaunchAttributeMemSyncDomain` 和 `cudaLaunchAttributeMemSyncDomainMap` 访问。前者在逻辑域 `cudaLaunchMemSyncDomainDefault` 与 `cudaLaunchMemSyncDomainRemote` 之间进行选择，后者提供从逻辑域到物理域的映射。远程域用于执行远程内存访问的 kernel，目的是将其内存流量与本地 kernel 隔离。但是，选择某个域不会影响 kernel 可以合法执行哪些内存访问。

可以通过设备属性 `cudaDevAttrMemSyncDomainCount` 查询域的数量。计算能力 9.0（Hopper）的设备有 4 个域。为了便于编写可移植代码，所有设备都可以使用域功能；对于计算能力低于 9.0 的设备，CUDA 会报告域数量为 1。

逻辑域可以简化应用组合。栈底层的单次 kernel 启动（例如 NCCL 发起的启动）可以选择具有语义含义的逻辑域，而不必关注外围应用的架构；更高层可以通过映射来控制逻辑域。如果没有设置逻辑域，默认值是默认域；默认映射是将默认域映射到 0、远程域映射到 1（在域数大于 1 的 GPU 上）。在 CUDA 12.0 及更高版本中，特定库可以将启动标记为远程域，例如 NCCL 2.16 会这样做。这为常见应用提供了开箱即用的有益模式，无需修改其他组件、框架或应用代码。另一种模式是划分并行 stream，例如使用 NVSHMEM，或应用没有清晰的 kernel 类型划分时，可以让 stream A 将两个逻辑域都映射到物理域 0，让 stream B 将两个逻辑域都映射到物理域 1，依此类推。

```cpp
// 使用远程逻辑域启动 kernel
cudaLaunchAttribute domainAttr;
domainAttr.id = cudaLaunchAttributeMemSyncDomain;
domainAttr.val = cudaLaunchMemSyncDomainRemote;
cudaLaunchConfig_t config;
// 填充其他 config 字段
config.attrs = &domainAttr;
config.numAttrs = 1;
cudaLaunchKernelEx(&config, myKernel, kernelArg1, kernelArg2...);
```

```cpp
// 为 stream 设置映射
// （这是计算能力 9.0（Hopper）及更高版本上 stream 的默认映射，
// 这里显式写出仅用于说明。）
cudaLaunchAttributeValue mapAttr;
mapAttr.memSyncDomainMap.default_ = 0;
mapAttr.memSyncDomainMap.remote = 1;
cudaStreamSetAttribute(stream, cudaLaunchAttributeMemSyncDomainMap, &mapAttr);
```

```cpp
// 将不同 stream 映射到不同的物理域，忽略逻辑域设置
cudaLaunchAttributeValue mapAttr;
mapAttr.memSyncDomainMap.default_ = 0;
mapAttr.memSyncDomainMap.remote = 0;
cudaStreamSetAttribute(streamA, cudaLaunchAttributeMemSyncDomainMap, &mapAttr);
mapAttr.memSyncDomainMap.default_ = 1;
mapAttr.memSyncDomainMap.remote = 1;
cudaStreamSetAttribute(streamB, cudaLaunchAttributeMemSyncDomainMap, &mapAttr);
```

与其他启动属性一样，这些属性可以统一应用于 CUDA stream、使用 `cudaLaunchKernelEx` 的单次启动，以及 CUDA Graph 中的 kernel 节点。典型用法是在 stream 层设置映射，并在启动层设置逻辑域（或者在一段 stream 使用的前后设置它），如上所述。

在 stream capture 期间，这两个属性都会复制到 graph 节点。Graph 会从节点自身取得这两个属性，本质上是间接指定物理域。设置在 graph 所启动到的 stream 上、与域相关的属性不会参与该 graph 的执行。
