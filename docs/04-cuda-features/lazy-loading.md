---
title: 4.7 惰性加载
description: CUDA 模块惰性加载的原理、要求、用法与潜在影响
---

# 4.7 惰性加载（Lazy Loading）

## 4.7.1 介绍（Introduction）

惰性加载通过等到 CUDA 模块真正需要时才加载它们，来缩短程序初始化时间。对于只使用所包含 kernel 中少数几个的程序（库的常见使用方式），惰性加载尤其有效。遵循 CUDA 编程模型时，惰性加载的设计目标是对用户透明。详见[潜在风险](#475-潜在风险)。从 CUDA 12.3 开始，所有平台默认启用惰性加载，但可以通过环境变量 `CUDA_MODULE_LOADING` 控制。

## 4.7.2 变更历史（Change History）

| CUDA 版本 | 变更 |
| --- | --- |
| 12.3 | 惰性加载性能提升；现在 Windows 默认启用。 |
| 12.2 | Linux 默认启用惰性加载。 |
| 11.7 | 首次引入惰性加载，默认禁用。 |

表 17：按 CUDA 版本列出的惰性加载变更（Select Lazy Loading Changes by CUDA Version）

## 4.7.3 惰性加载的要求（Requirements for Lazy Loading）

惰性加载是 CUDA runtime 与 driver 共同提供的功能。只有同时满足 runtime 和 driver 的版本要求时，才能使用惰性加载。

### 4.7.3.1 CUDA Runtime 版本要求（CUDA Runtime Version Requirement）

CUDA runtime 从 11.7 版本开始支持惰性加载。由于 CUDA runtime 通常静态链接到程序和库中，只有使用 CUDA 11.7 及以上 toolkit 编译或构建的程序和库才能受益于惰性加载。使用较旧 CUDA runtime 版本编译的库会立即加载全部模块。

### 4.7.3.2 CUDA Driver 版本要求（CUDA Driver Version Requirement）

惰性加载要求 driver 版本为 515 或更高。即使使用 CUDA 11.7 或更高版本的 toolkit，版本低于 515 的 driver 也不支持惰性加载。

### 4.7.3.3 编译器要求（Compiler Requirements）

惰性加载不要求编译器提供任何支持。使用 11.7 以前的编译器生成的 SASS 和 PTX 也可以在启用惰性加载时加载，并完整受益于该功能。不过，如上所述，仍然需要 11.7 及以上版本的 CUDA runtime。

### 4.7.3.4 Kernel 要求（Kernel Requirements）

惰性加载不影响包含 managed variable 的模块；这类模块仍会被立即加载。

## 4.7.4 用法（Usage）

### 4.7.4.1 启用与禁用（Enabling & Disabling）

将环境变量 `CUDA_MODULE_LOADING` 设置为 `LAZY` 即可启用惰性加载。将其设置为 `EAGER` 即可禁用惰性加载。从 CUDA 12.3 开始，所有平台默认启用惰性加载。

### 4.7.4.2 运行时检查是否启用惰性加载（Checking if Lazy Loading is Enabled at Runtime）

CUDA driver API 中的 `cuModuleGetLoadingMode` API 可用于确定是否启用了惰性加载。调用此函数前必须先初始化 CUDA。下面给出示例用法。

```cpp
#include "<cuda.h>"
#include "<assert.h>"
#include "<iostream>"

int main() {
        CUmoduleLoadingMode mode;

        assert(CUDA_SUCCESS == cuInit(0));
        assert(CUDA_SUCCESS == cuModuleGetLoadingMode(&mode));

        std::cout << "CUDA Module Loading Mode is " << ((mode == CU_MODULE_LAZY_LOADING) ? "lazy" : "eager") << std::endl;

        return 0;
}
```

### 4.7.4.3 在运行时强制模块立即加载（Forcing a Module to Load Eagerly at Runtime）

Kernel 和变量会自动加载，不需要显式加载。即使不执行 kernel，也可以通过以下方式显式加载 kernel：

- `cuModuleGetFunction()` 会使模块加载到 device memory。
- `cudaFuncGetAttributes()` 会使 kernel 加载到 device memory。

> **注意**
>
> `cuModuleLoad()` 不保证模块会立即加载。

## 4.7.5 潜在风险（Potential Hazards）

惰性加载的设计目标是：应用无需为使用它进行修改。不过，当应用没有完全遵循 CUDA 编程模型时，仍有一些注意事项，具体如下。

### 4.7.5.1 对并发 Kernel 执行的影响（Impact on Concurrent Kernel Execution）

一些程序错误地假设并发 kernel 执行一定会发生。如果需要跨 kernel 同步，而 kernel 执行却被串行化，就可能发生死锁。要尽量降低惰性加载对并发 kernel 执行的影响，可以采取以下措施：

- 在启动之前预加载所有希望并发执行的 kernel；或者
- 运行应用时将 `CUDA_MODULE_LOADING` 设置为 `EAGER`，强制立即加载数据，而不强制每个函数都立即加载。

### 4.7.5.2 大型内存分配（Large Memory Allocations）

惰性加载会把 CUDA 模块的内存分配从程序初始化阶段推迟到更接近执行的时刻。如果应用在启动时分配了全部 VRAM，CUDA 可能在运行时无法为模块分配内存。可行的解决方案包括：

- 使用 `cudaMallocAsync()`，而不是在启动时分配全部 VRAM 的分配器；
- 增加一定缓冲空间，以弥补 kernel 延迟加载所需的内存；
- 在尝试初始化分配器之前，预加载程序将使用的所有 kernel。

### 4.7.5.3 对性能测量的影响（Impact on Performance Measurements）

惰性加载可能把 CUDA 模块初始化移入被测执行窗口，从而使性能测量产生偏差。为避免这种情况：

- 在测量前至少执行一次预热迭代；
- 在启动被测 kernel 之前预加载它。
