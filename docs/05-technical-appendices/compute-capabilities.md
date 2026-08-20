---
title: 5.1 计算能力（Compute Capabilities）
description: CUDA 计算能力、功能支持与技术规格
---

<a id="compute-capabilities"></a>

# 5.1 计算能力（Compute Capabilities）

计算设备的一般规格和功能取决于其计算能力（见[计算能力与流式多处理器版本](../01-introduction/cuda-platform.html#cuda-platform-compute-capability-sm-version)）。

[表 29](#compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability)、[表 30](#compute-capabilities-table-device-and-streaming-multiprocessor-sm-information-per-compute-capability)和[表 31](#compute-capabilities-table-memory-information-per-compute-capability)展示了当前支持的各个计算能力对应的功能与技术规格。

所有 NVIDIA GPU 架构都使用小端表示。

<a id="obtain-the-gpu-compute-capability"></a>

## 5.1.1 获取 GPU 计算能力（Obtain the GPU Compute Capability）

[CUDA GPU 计算能力](https://developer.nvidia.com/cuda-gpus)页面提供了 NVIDIA GPU 型号与其计算能力之间的完整对应关系。

另外，也可以使用随[NVIDIA 驱动程序](https://www.nvidia.com/en-us/drivers/)提供的[nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html)工具获取 GPU 的计算能力。例如，下面的命令会输出系统中可用的 GPU 名称和计算能力：

```bash
nvidia-smi --query-gpu=name,compute_cap
```

在运行时，可以通过 CUDA Runtime API 的 [`cudaDeviceGetAttribute()`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151)、CUDA Driver API 的 [`cuDeviceGetAttribute()`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266)或 NVML API 的 [`nvmlDeviceGetCudaComputeCapability()`](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g1f803a2fb4b7dfc0a8183b46b46ab03a)获取计算能力：

```cpp
#include <cuda_runtime_api.h>

int computeCapabilityMajor, computeCapabilityMinor;
cudaDeviceGetAttribute(&computeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, device_id);
cudaDeviceGetAttribute(&computeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, device_id);
```

```cpp
#include <cuda.h>

int computeCapabilityMajor, computeCapabilityMinor;
cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_id);
cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_id);
```

```cpp
#include <nvml.h> // required linking with -lnvidia-ml

int computeCapabilityMajor, computeCapabilityMinor;
nvmlDeviceGetCudaComputeCapability(nvmlDevice, &computeCapabilityMajor, &computeCapabilityMinor);
```

<a id="feature-availability"></a>

## 5.1.2 功能可用性（Feature Availability）

某个计算架构引入的大多数计算功能，预期会在之后的所有架构上可用。在[表 29](#compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability)中，对于引入该功能之后的计算能力，会以“是（Yes）”表示该功能可用。

<a id="architecture-specific-features"></a>
<a id="compute-capabilities-architecture-specific-features"></a>

### 5.1.2.1 架构特定功能（Architecture-Specific Features）

从计算能力 9.0 的设备开始，某个架构引入的专用计算功能可能不保证在之后的所有计算能力上可用。这些功能称为*架构特定（architecture-specific）*功能，目标是加速专用操作，例如 Tensor Core 操作；这些操作并非面向所有类别的计算能力，或者在未来代际中可能发生重大变化。必须使用架构特定编译器目标（见[功能集编译器目标](#feature-set-compiler-targets)）编译代码，才能启用架构特定功能。使用架构特定编译器目标编译的代码，只能在编译时所针对的确切计算能力上运行。

<a id="family-specific-features"></a>
<a id="compute-capabilities-family-specific-features"></a>

### 5.1.2.2 系列特定功能（Family-Specific Features）

从计算能力 10.0 的设备开始，一些架构特定功能会由多个计算能力的设备共同拥有。包含这些功能的设备属于同一系列，这些功能也可以称为*系列特定（family-specific）*功能。系列特定功能保证在同一系列的所有设备上可用。启用系列特定功能需要使用系列特定编译器目标，详见[第 5.1.2.3 节](#feature-set-compiler-targets)。针对系列特定目标编译的代码，只能在属于该系列的 GPU 上运行。

<a id="feature-set-compiler-targets"></a>
<a id="compute-capabilities-feature-set-compiler-targets"></a>

### 5.1.2.3 功能集编译器目标（Feature Set Compiler Targets）

编译器可以针对三组计算功能：

**基线功能集（Baseline Feature Set）**：以可在后续计算架构上使用为目标而引入的、占主导地位的一组计算功能。这些功能及其可用性汇总在[表 29](#compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability)中。

**架构特定功能集（Architecture-Specific Feature Set）**：一组规模较小且高度专用的功能，称为架构特定功能。这些功能用于加速专用操作，但不保证在后续计算架构上可用，或者可能发生重大变化。这些功能汇总在相应的“计算能力 #.#”小节中。架构特定功能集是系列特定功能集的超集。架构特定编译器目标随计算能力 9.0 设备引入，在编译目标中使用 **a** 后缀选择，例如将 `compute_100a` 或 `compute_120a` 指定为计算目标。

**系列特定功能集（Family-Specific Feature Set）**：一些架构特定功能由多个计算能力的 GPU 共同拥有。这些功能汇总在相应的“计算能力 #.#”小节中。除少数例外，具有相同主计算能力的后代设备属于同一系列。[表 28](#compute-capabilities-family-specific-compatibility)给出了系列特定目标与设备计算能力之间的兼容性，包括例外情况。系列特定功能集是基线功能集的超集。系列特定编译器目标随计算能力 10.0 设备引入，在编译目标中使用 **f** 后缀选择，例如将 `compute_100f` 或 `compute_120f` 指定为计算目标。

从计算能力 9.0 开始，所有设备都具有一组架构特定功能。要在特定 GPU 上使用这些功能的完整集合，必须使用带 **a** 后缀的架构特定编译器目标。此外，从计算能力 10.0 开始，不同次要计算能力的多个设备中会出现一些共同的功能集合。这些指令集合称为系列特定功能，共享这些功能的设备称为同一系列的成员。系列特定功能是架构特定功能的子集，由该 GPU 系列的所有成员共同拥有。带 **f** 后缀的系列特定编译器目标允许编译器生成使用这部分架构特定公共功能的代码。

例如：

- `compute_100` 编译目标不允许使用架构特定功能。该目标与计算能力 10.0 及更高版本的所有设备兼容。

- *系列特定*编译目标 `compute_100f` 允许使用该 GPU 系列共同拥有的架构特定功能子集。该目标只与属于该 GPU 系列的设备兼容。在本例中，它与计算能力 10.0 和计算能力 10.3 的设备兼容。系列特定 `compute_100f` 目标可用的功能是基线 `compute_100` 目标可用功能的超集。

- *架构特定*编译目标 `compute_100a` 允许在计算能力 10.0 设备上使用完整的架构特定功能集合。该目标只与计算能力 10.0 的设备兼容，不与其他计算能力兼容。`compute_100a` 目标可用的功能是 `compute_100f` 目标可用功能的超集。

<table id="compute-capabilities-family-specific-compatibility">
<caption>表 28　系列特定兼容性（Family-Specific Compatibility）</caption>
<thead>
<tr><th>编译目标（Compilation Target）</th><th colspan="2">兼容的计算能力（Compatible with Compute Capability）</th></tr>
</thead>
<tbody>
<tr><td><code>compute_100f</code></td><td>10.0</td><td>10.3</td></tr>
<tr><td><code>compute_103f</code></td><td colspan="2">10.3<sup><a href="#family-specific-footnote">[1]</a></sup></td></tr>
<tr><td><code>compute_110f</code></td><td colspan="2">11.0<sup><a href="#family-specific-footnote">[1]</a></sup></td></tr>
<tr><td><code>compute_120f</code></td><td>12.0</td><td>12.1</td></tr>
<tr><td><code>compute_121f</code></td><td colspan="2">12.1<sup><a href="#family-specific-footnote">[1]</a></sup></td></tr>
</tbody>
</table>

<p id="family-specific-footnote"><sup>[1]</sup> 某些系列在创建时只包含一个成员。未来可能扩展这些系列，使其包含更多设备。</p>

<a id="features-and-technical-specifications"></a>

## 5.1.3 功能与技术规格（Features and Technical Specifications）

<table id="compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability">
<caption>表 29　各计算能力的功能支持（Feature Support per Compute Capability）</caption>
<thead>
<tr><th><strong>功能支持（Feature Support）</strong></th><th colspan="6"><strong>计算能力（Compute Capability）</strong></th></tr>
</thead>
<tbody>
<tr><td>（未列出的功能在所有计算能力上都受支持）</td><td>7.x</td><td>8.x</td><td>9.0</td><td>10.x</td><td>11.0</td><td>12.x</td></tr>
<tr><td>在 shared memory 和 global memory 中对 128 位整数值执行原子操作（<a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions">Atomic Functions</a>）</td><td colspan="2">否（No）</td><td colspan="4">是（Yes）</td></tr>
<tr><td>在 global memory 中对 <code>float2</code> 和 <code>float4</code> 浮点向量执行原子加法（<a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd">atomicAdd()</a>）</td><td colspan="2">否（No）</td><td colspan="4">是（Yes）</td></tr>
<tr><td>Warp 归约函数（<a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-reduce-functions">Warp Reduce Functions</a>）</td><td>否（No）</td><td colspan="5">是（Yes）</td></tr>
<tr><td>Bfloat16 精度浮点运算</td><td>否（No）</td><td colspan="5">是（Yes）</td></tr>
<tr><td>128 位精度浮点运算</td><td colspan="3">否（No）</td><td colspan="3">是（Yes）</td></tr>
<tr><td>硬件加速的 <code>memcpy_async</code>（<a href="../04-cuda-features/pipelines.html#pipelines">Pipelines</a>）</td><td>否（No）</td><td colspan="5">是（Yes）</td></tr>
<tr><td>硬件加速的 Split Arrive/Wait Barrier（分离式到达/等待屏障）（<a href="../04-cuda-features/asynchronous-barriers.html#asynchronous-barriers">Asynchronous Barriers</a>）</td><td>否（No）</td><td colspan="5">是（Yes）</td></tr>
<tr><td>L2 Cache 驻留管理（<a href="../04-cuda-features/l2-cache-control.html#advanced-kernels-l2-control">L2 Cache Control</a>）</td><td>否（No）</td><td colspan="5">是（Yes）</td></tr>
<tr><td>用于加速动态规划的 DPX 指令（<a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dpx-instructions">Dynamic Programming eXtension (DPX) Instructions</a>）</td><td colspan="2">多条指令（Multiple Instr.）</td><td colspan="2">原生（Native）</td><td colspan="2">多条指令（Multiple Instr.）</td></tr>
<tr><td>分布式共享内存（Distributed Shared Memory）</td><td colspan="2">否（No）</td><td colspan="4">是（Yes）</td></tr>
<tr><td>线程块集群（<a href="../01-introduction/programming-model.html#programming-model-thread-block-clusters">Thread Block Clusters</a>）</td><td colspan="2">否（No）</td><td colspan="4">是（Yes）</td></tr>
<tr><td>Tensor Memory Accelerator（TMA）单元（<a href="../04-cuda-features/asynchronous-data-copies.html">使用 Tensor Memory Accelerator（TMA）</a>）</td><td colspan="2">否（No）</td><td colspan="4">是（Yes）</td></tr>
</tbody>
</table>

注意，下面各表使用的 KB 和 K 单位分别对应 1024 字节（即 KiB）和 1024。

<table id="compute-capabilities-table-device-and-streaming-multiprocessor-sm-information-per-compute-capability">
<caption>表 30　各计算能力的设备和流式多处理器（SM）信息（Device and Streaming Multiprocessor (SM) Information per Compute Capability）</caption>
<thead>
<tr><th></th><th colspan="10"><strong>计算能力（Compute Capability）</strong></th></tr>
</thead>
<tbody>
<tr><td></td><td>7.5</td><td>8.0</td><td>8.6</td><td>8.7</td><td>8.9</td><td>9.0</td><td>10.0</td><td>10.3</td><td>11.0</td><td>12.x</td></tr>
<tr><td>FP32 与 FP64 吞吐量之比<sup><a href="#non-tensor-throughput-footnote">[2]</a></sup></td><td>32:1</td><td>2:1</td><td colspan="3">64:1</td><td colspan="2">2:1</td><td colspan="3">64:1</td></tr>
<tr><td>每个设备的最大常驻 grid 数量（并发 kernel 执行）</td><td colspan="10">128</td></tr>
<tr><td>grid 的最大维度</td><td colspan="10">3</td></tr>
<tr><td>grid 的最大 x 维度</td><td colspan="10">2<sup>31</sup>-1</td></tr>
<tr><td>grid 的最大 y 或 z 维度</td><td colspan="10">65535</td></tr>
<tr><td>thread block 的最大维度</td><td colspan="10">3</td></tr>
<tr><td>thread block 的最大 x 或 y 维度</td><td colspan="10">1024</td></tr>
<tr><td>thread block 的最大 z 维度</td><td colspan="10">64</td></tr>
<tr><td>每个 block 的最大线程数</td><td colspan="10">1024</td></tr>
<tr><td>Warp 大小</td><td colspan="10">32</td></tr>
<tr><td>每个 SM 的最大常驻 block 数</td><td>16</td><td>32</td><td colspan="2">16</td><td>24</td><td colspan="3">32</td><td colspan="2">24</td></tr>
<tr><td>每个 SM 的最大常驻 warp 数</td><td>32</td><td>64</td><td colspan="3">48</td><td colspan="3">64</td><td colspan="2">48</td></tr>
<tr><td>每个 SM 的最大常驻线程数</td><td>1024</td><td>2048</td><td colspan="3">1536</td><td colspan="3">2048</td><td colspan="2">1536</td></tr>
<tr><td>Green Contexts：useFlags 为 0 时的最小 SM 分区大小</td><td>2</td><td colspan="4">4</td><td colspan="5">8</td></tr>
<tr><td>Green Contexts：useFlags 为 0 时每个分区的 SM 协同调度对齐量</td><td colspan="5">2</td><td colspan="5">8</td></tr>
</tbody>
</table>

<p id="non-tensor-throughput-footnote"><sup>[2]</sup> 非 Tensor Core 吞吐量。有关吞吐量的更多信息，请参阅 <a href="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions-throughput-native-arithmetic-instructions">CUDA Best Practices Guide</a>。</p>

<table id="compute-capabilities-table-memory-information-per-compute-capability">
<caption>表 31　各计算能力的内存信息（Memory Information per Compute Capability）</caption>
<thead>
<tr><th></th><th colspan="9"><strong>计算能力（Compute Capability）</strong></th></tr>
</thead>
<tbody>
<tr><td></td><td>7.5</td><td>8.0</td><td>8.6</td><td>8.7</td><td>8.9</td><td>9.0</td><td>10.x</td><td>11.0</td><td>12.x</td></tr>
<tr><td>每个 SM 的 32 位寄存器数量</td><td colspan="9">64 K</td></tr>
<tr><td>每个 thread block 的最大 32 位寄存器数量</td><td colspan="9">64 K</td></tr>
<tr><td>每个线程的最大 32 位寄存器数量</td><td colspan="9">255</td></tr>
<tr><td>每个 SM 的最大 shared memory 数量</td><td>64 KB</td><td>164 KB</td><td>100 KB</td><td>164 KB</td><td>100 KB</td><td colspan="3">228 KB</td><td>100 KB</td></tr>
<tr><td>每个 thread block 的最大 shared memory 数量<sup><a href="#shared-memory-footnote">[3]</a></sup></td><td>64 KB</td><td>163 KB</td><td>99 KB</td><td>163 KB</td><td>99 KB</td><td colspan="3">227 KB</td><td>99 KB</td></tr>
<tr><td>shared memory bank 数量</td><td colspan="9">32</td></tr>
<tr><td>每个线程的最大 local memory 数量</td><td colspan="9">512 KB</td></tr>
<tr><td>constant memory 大小</td><td colspan="9">64 KB</td></tr>
<tr><td>每个 SM 的 constant memory 缓存工作集</td><td colspan="9">8 KB</td></tr>
<tr><td>每个 SM 的 texture memory 缓存工作集</td><td>32 或 64 KB</td><td>28 KB ~ 192 KB</td><td>28 KB ~ 128 KB</td><td>28 KB ~ 192 KB</td><td>28 KB ~ 128 KB</td><td colspan="3">28 KB ~ 256 KB</td><td>28 KB ~ 128 KB</td></tr>
</tbody>
</table>

<p id="shared-memory-footnote"><sup>[3]</sup> 依赖每个 block 超过 48 KB shared memory 分配的 kernel，必须使用 dynamic shared memory 并显式选择启用，见[配置 L1/Shared Memory 平衡](../03-advanced-cuda/advanced-kernel-programming.html#advanced-kernel-l1-shared-config)。</p>

<table id="compute-capabilities-table-shared-memory-capacity-per-compute-capability">
<caption>表 32　各计算能力的 shared memory 容量（Shared Memory Capacity per Compute Capability）</caption>
<thead>
<tr><th>计算能力（Compute Capability）</th><th>统一数据缓存大小（KB）</th><th>SMEM 容量大小（KB）</th></tr>
</thead>
<tbody>
<tr><td>7.5</td><td>96</td><td>32, 64</td></tr>
<tr><td>8.0</td><td>192</td><td>0, 8, 16, 32, 64, 100, 132, 164</td></tr>
<tr><td>8.6</td><td>128</td><td>0, 8, 16, 32, 64, 100</td></tr>
<tr><td>8.7</td><td>192</td><td>0, 8, 16, 32, 64, 100, 132, 164</td></tr>
<tr><td>8.9</td><td>128</td><td>0, 8, 16, 32, 64, 100</td></tr>
<tr><td>9.0</td><td>256</td><td>0, 8, 16, 32, 64, 100, 132, 164, 196, 228</td></tr>
<tr><td>10.x</td><td>256</td><td>0, 8, 16, 32, 64, 100, 132, 164, 196, 228</td></tr>
<tr><td>11.0</td><td>256</td><td>0, 8, 16, 32, 64, 100, 132, 164, 196, 228</td></tr>
<tr><td>12.x</td><td>128</td><td>0, 8, 16, 32, 64, 100</td></tr>
</tbody>
</table>

[表 33](#compute-capabilities-table-tensor-core-data-types-per-compute-capability)展示了 Tensor Core 加速支持的输入数据类型。Tensor Core 功能集可以通过 inline PTX 在 CUDA 编译工具链中使用。强烈建议应用通过 cuDNN、cuBLAS 和 cuFFT 等 CUDA-X 库使用该功能集，或者使用[CUTLASS](https://docs.nvidia.com/cutlass/index.html)。CUTLASS 是一组 CUDA C++ 模板抽象和 Python 领域特定语言（DSL），旨在支持 CUDA 各个层级上的高性能矩阵-矩阵乘法（GEMM）及相关计算。

<table id="compute-capabilities-table-tensor-core-data-types-per-compute-capability">
<caption>表 33　各计算能力支持 Tensor Core 加速的输入数据类型（Input Data Types Supported by Tensor Core Acceleration per Compute Capability）</caption>
<thead>
<tr><th>计算能力（Compute Capability）</th><th colspan="9">Tensor Core 输入数据类型（Tensor Core Input Data Types）</th></tr>
</thead>
<tbody>
<tr><td></td><td>FP64</td><td>TF32</td><td>BF16</td><td>FP16</td><td>FP8</td><td>FP6</td><td>FP4</td><td>INT8</td><td>INT4</td></tr>
<tr><td>7.5</td><td colspan="3"></td><td>是（Yes）</td><td colspan="3"></td><td>是（Yes）</td><td>是（Yes）</td></tr>
<tr><td>8.0</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td colspan="3"></td><td>是（Yes）</td><td>是（Yes）</td></tr>
<tr><td>8.6</td><td></td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td colspan="3"></td><td>是（Yes）</td><td>是（Yes）</td></tr>
<tr><td>8.7</td><td></td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td colspan="3"></td><td>是（Yes）</td><td>是（Yes）</td></tr>
<tr><td>8.9</td><td></td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td colspan="2"></td><td>是（Yes）</td><td>是（Yes）</td></tr>
<tr><td>9.0</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td colspan="2"></td><td>是（Yes）</td><td></td></tr>
<tr><td>10.0</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td></td></tr>
<tr><td>10.3</td><td></td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td></td></tr>
<tr><td>11.0</td><td></td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td></td></tr>
<tr><td>12.x</td><td></td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td>是（Yes）</td><td></td></tr>
</tbody>
</table>
