---
title: 1.3 CUDA 平台
description: CUDA 平台、Toolkit、Driver、PTX、cubin 与 fatbin 的中文翻译
---

# 1.3 CUDA 平台（The CUDA platform）

> 本页按 NVIDIA CUDA Programming Guide Release 13.3 的在线页面逐段翻译。原文页面：[1.3. The CUDA platform](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)；官方页面标注的更新时间为 2026-05-27。

NVIDIA CUDA 平台由许多软件和硬件组成，也包含为异构系统计算而开发的许多重要技术。本章介绍应用开发者需要理解的 CUDA 平台基础概念和组件。本章与[编程模型](./programming-model.html)一样，不针对某一种编程语言，而适用于所有使用 CUDA 平台的程序。

## 1.3.1 计算能力与流式多处理器版本（Compute Capability and Streaming Multiprocessor Versions）

每个 NVIDIA GPU 都有一个计算能力（Compute Capability，CC）编号。它表示该 GPU 支持哪些功能，也规定了该 GPU 的一些硬件参数。这些规格记录在[第 5.1 节：计算能力](../05-technical-appendices/compute-capabilities.html)中；NVIDIA GPU 及其计算能力的完整列表维护在 [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) 页面上。

计算能力采用 `X.Y` 格式表示，其中 `X` 是主版本号，`Y` 是次版本号。例如，CC 12.0 的主版本号为 12，次版本号为 0。计算能力直接对应 SM 的版本号：CC 12.0 GPU 中的 SM 版本为 `sm_120`，这个版本标识会用于标记二进制代码。

[第 5.1.1 节](../05-technical-appendices/compute-capabilities.html)介绍如何查询和确定系统中 GPU 的计算能力。

## 1.3.2 CUDA Toolkit 与 NVIDIA Driver

可以把 NVIDIA Driver 理解为 GPU 的操作系统。NVIDIA Driver 是必须安装在 host 系统操作系统中的软件组件，所有 GPU 用途都需要它，包括显示和图形功能。NVIDIA Driver 是 CUDA 平台的基础；除 CUDA 外，它还提供使用 GPU 的其他方式，例如 Vulkan 和 Direct3D。NVIDIA Driver 的版本号形式可以是 `r580`。

CUDA Toolkit 是一组用于编写、构建和分析 GPU 计算软件的库、头文件和工具。CUDA Toolkit 是独立于 NVIDIA Driver 的另一个软件产品。

CUDA runtime 是 CUDA Toolkit 所提供库中的一种特殊情况。CUDA runtime 提供 API 和一些语言扩展，用来处理分配内存、在 GPU 与其他 GPU 或 CPU 之间复制数据、启动 kernel 等常见任务。CUDA runtime 的 API 组件称为 **CUDA Runtime API**。

[CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) 文档详细说明了不同 GPU、NVIDIA Driver 和 CUDA Toolkit 版本之间的兼容性。

### 1.3.2.1 CUDA Runtime API 与 CUDA Driver API

CUDA Runtime API 建立在更底层的 CUDA Driver API 之上；CUDA Driver API 是 NVIDIA Driver 暴露出来的 API。本指南重点介绍 CUDA Runtime API 暴露的 API。只使用 Driver API 也可以实现相同的功能，但有些功能只能通过 Driver API 使用。应用可以选择使用其中一种 API，也可以同时使用两种 API 并让它们互操作。[CUDA Driver API](../03-advanced-cuda/driver-api.html) 一节介绍 Runtime API 与 Driver API 之间的互操作。

CUDA Runtime API 函数的完整 API 参考见 [CUDA Runtime API Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)。

CUDA Driver API 的完整 API 参考见 [CUDA Driver API Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)。

## 1.3.3 并行线程执行（Parallel Thread Execution，PTX）

CUDA 平台中一个基础但有时不可见的层是并行线程执行（Parallel Thread Execution，PTX）虚拟指令集架构（instruction set architecture，ISA）。PTX 是 NVIDIA GPU 的一种高级汇编语言，为真实 GPU 硬件的物理 ISA 提供抽象层。和其他平台一样，应用也可以直接用这种汇编语言编写，但这样可能给软件开发增加不必要的复杂度和难度。

领域专用语言和高级语言编译器可以把代码生成到 PTX 这一中间表示（intermediate representation，IR），然后使用 NVIDIA 的离线编译或即时（just-in-time，JIT）编译工具生成可执行的 GPU 二进制代码。因此，CUDA 平台并不只支持 NVIDIA 工具链所支持的语言；其他语言也可以通过这种方式编程 CUDA 平台，例如 [NVCC：NVIDIA CUDA 编译器](../02-programming-gpus/nvcc.html)。

随着 GPU 能力不断变化和增长，PTX 虚拟 ISA 规范也会进行版本化。PTX 版本与 SM 版本一样，对应某个计算能力。例如，支持计算能力 8.0 全部功能的 PTX 称为 `compute_80`。

PTX 的完整文档见 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)。

## 1.3.4 Cubin 与 Fatbin（Cubins and Fatbins）

CUDA 应用和库通常使用 C++ 等高级语言编写。高级语言会先编译为 PTX，然后 PTX 再被编译成针对真实 GPU 的二进制代码，称为 CUDA binary，简称 **cubin**。cubin 针对特定的 SM 版本使用特定的二进制格式，例如 `sm_120`。

使用 GPU 计算的可执行文件和库二进制文件同时包含 CPU 代码和 GPU 代码。GPU 代码存储在称为 **fatbin** 的容器中。fatbin 可以包含针对多个不同目标的 cubin 和 PTX。例如，一个应用可以构建出针对多种 GPU 架构（也就是不同 SM 版本）的二进制代码。应用运行时，GPU 代码会被加载到某一张具体的 GPU 上，然后从 fatbin 中选择最适合该 GPU 的二进制版本。

fatbin 还可以包含一个或多个 GPU 代码的 PTX 版本，其用途见 [PTX Compatibility](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)。下图展示了一个应用或库二进制文件的示例：其中包含多个 cubin 版本的 GPU 代码和一个 PTX 版本的 GPU 代码。

![图 10：可执行文件或库中的 fatbin 容器](/images/chapter-01/figure-10-fatbin.png)

<div class="figure-caption"><strong>图 10：fatbin、cubin 与 PTX。</strong> 原 PDF 第 33 页。可执行文件或库同时包含 CPU 二进制代码和 GPU 的 fatbin 容器；fatbin 可以包含多个 cubin，也可以包含 PTX 虚拟 ISA 代码，后者能够为未来的目标执行 JIT 编译。</div>

### 1.3.4.1 二进制兼容性（Binary Compatibility）

在特定情况下，NVIDIA GPU 保证二进制兼容性。具体来说，在同一个计算能力主版本内，次版本号大于或等于 cubin 目标版本的 GPU 可以加载并执行该 cubin。例如，如果应用包含针对计算能力 8.6 编译的 cubin，那么它可以在计算能力 8.6 或 8.9 的 GPU 上加载和执行。

但是，这个 cubin 不能在计算能力 8.0 的 GPU 上加载，因为 GPU 的 CC 次版本号 0 低于代码的次版本号 6。

不同计算能力主版本之间不存在二进制兼容性。也就是说，针对计算能力 8.6 编译的 cubin 不能在计算能力 9.0 的 GPU 上加载。

讨论二进制代码时，代码常被称为具有某个版本，例如前面例子中的 `sm_86`。这和说二进制代码是针对计算能力 8.6 构建的是同一件事。之所以经常使用这种简写，是因为开发者正是以这种形式向 NVIDIA CUDA 编译器 [nvcc](../02-programming-gpus/nvcc.html) 指定二进制构建目标的。

::: warning 兼容性承诺的边界
二进制兼容性只对 NVIDIA 工具（例如 `nvcc`）生成的二进制有效。NVIDIA 不支持手工编辑或生成 GPU 二进制代码；如果以任何方式修改二进制，兼容性承诺即失效。
:::

### 1.3.4.2 PTX 兼容性（PTX Compatibility）

GPU 代码可以以二进制形式或 PTX 形式存储在可执行文件中，这一点在[上面的 Cubin 与 Fatbin](#134-cubin-与-fatbin-cubins-and-fatbins)中已经介绍。当应用存储 GPU 代码的 PTX 版本时，该 PTX 可以在应用运行时为计算能力等于或高于 PTX 代码对应能力的目标执行 JIT 编译。

例如，如果应用包含 `compute_80` 的 PTX，那么该 PTX 可以在运行时 JIT 编译为后续的 SM 版本，例如 `sm_120`。

这使应用或库无需重新构建，就能与未来的 GPU 保持前向兼容。

### 1.3.4.3 即时编译（Just-in-Time Compilation）

应用在运行时加载的 PTX 代码由设备驱动编译成二进制代码，这个过程称为即时编译（just-in-time compilation，JIT）。JIT 会增加应用的加载时间，但允许应用受益于每个新设备驱动中不断改进的编译器。它还使应用可以运行在编译时尚不存在的设备上。

当设备驱动为应用执行 PTX 的 JIT 编译时，会自动缓存生成的二进制代码，从而避免后续调用重复编译。这个缓存称为 **compute cache**；设备驱动升级时会自动使其失效，以便应用使用新驱动内置的最新 JIT 编译器改进。

从 CUDA 的早期版本开始，PTX 在运行时如何以及何时执行 JIT 编译已经逐步放宽，从而可以更灵活地决定是否以及何时为部分或全部 kernel 执行 JIT。[Lazy Loading](../04-cuda-features/lazy-loading.html) 一节介绍可用选项以及控制 JIT 行为的方法；另有一些环境变量可以控制即时编译行为，详见[环境变量](../05-technical-appendices/environment-variables.html)。

除了使用 `nvcc` 编译 CUDA C++ device code 外，还可以使用 NVRTC 在运行时把 CUDA C++ device code 编译为 PTX。更多信息见 [NVRTC User Guide](https://docs.nvidia.com/cuda/nvrtc/index.html)。

## 本节导航

- [1.1 引言](./introduction.html)
- [1.2 编程模型](./programming-model.html)
- [第 5.1 节：计算能力](../05-technical-appendices/compute-capabilities.html)
- [PDF 图版：图 10](../figures.html#图-10-fatbin-与-ptx)
