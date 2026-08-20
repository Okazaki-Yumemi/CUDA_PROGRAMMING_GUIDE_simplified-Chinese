---
title: '2.7 NVCC：NVIDIA CUDA 编译器（NVCC: The NVIDIA CUDA Compiler）'
description: 完整翻译 NVIDIA CUDA Programming Guide 第 2.7 节 NVCC
---

# 2.7 NVCC：NVIDIA CUDA 编译器（NVCC: The NVIDIA CUDA Compiler）

[`nvcc`](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) 是 NVIDIA 提供的工具链，用于编译 CUDA C/C++ 以及 [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 代码。该工具链属于 [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)，包含编译器、链接器以及 PTX 和 [Cubin](../01-introduction/cuda-platform.html#cuda-平台中的-cubin与-fatbin) 汇编器等多个工具。顶层的 `nvcc` 工具负责协调编译过程，并为每个编译阶段调用适当的工具。

`nvcc` 驱动 CUDA 代码的离线编译，这一点不同于由 CUDA 运行时编译器 [`nvrtc`](https://docs.nvidia.com/cuda/nvrtc/index.html) 驱动的在线编译或即时（Just-in-Time，JIT）编译。

本节介绍构建应用程序时最常用的 `nvcc` 用法和所需细节。关于 `nvcc` 的完整说明请参阅 [nvcc 文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)。

## 2.7.1 CUDA 源文件与头文件（CUDA Source Files and Headers）

使用 `nvcc` 编译的源文件可以同时包含在 CPU 上执行的 host code 和在 GPU 上执行的 device code。对于只包含 host code 的文件，`nvcc` 接受常见的 C/C++ 源文件扩展名：`.c`、`.cpp`、`.cc`、`.cxx`；对于包含 device code，或混合包含 host code 与 device code 的文件，使用 `.cu`。包含 device code 的头文件通常使用 `.cuh` 扩展名，以便与只包含 host code 的 `.h`、`.hpp`、`.hh`、`.hxx` 等头文件区分开来。

| 文件扩展名 | 描述 | 内容 |
| --- | --- | --- |
| `.c` | C 源文件 | 仅 host code |
| `.cpp`、`.cc`、`.cxx` | C++ 源文件 | 仅 host code |
| `.h`、`.hpp`、`.hh`、`.hxx` | C/C++ 头文件 | device code、host code，或 host/device 混合代码 |
| `.cu` | CUDA 源文件 | device code、host code，或 host/device 混合代码 |
| `.cuh` | CUDA 头文件 | device code、host code，或 host/device 混合代码 |

## 2.7.2 NVCC 编译工作流（NVCC Compilation Workflow）

在初始阶段，`nvcc` 将 device code 与 host code 分离，然后分别将它们交给 GPU 编译器和 host 编译器编译。

要编译 host code，CUDA 编译器 `nvcc` 需要系统中存在兼容的 host 编译器。CUDA Toolkit 分别为 [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy) 和 [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements) 平台定义了 host 编译器支持策略。

只包含 host code 的文件既可以使用 `nvcc` 构建，也可以直接使用 host 编译器构建。最终生成的目标文件可以在链接时与包含 GPU code 的 `nvcc` 目标文件合并。

GPU 编译器将 C/C++ device code 编译为 PTX 汇编代码。对于命令行中指定的每个虚拟机器指令集架构（例如 `compute_90`），GPU 编译器都会运行一次。

随后，单独的 PTX 代码会传递给 `ptxas` 工具；`ptxas` 为目标硬件 ISA 生成 [Cubin](../01-introduction/cuda-platform.html#cuda-平台中的-cubin与-fatbin)。硬件 ISA 由其 [SM 版本](../01-introduction/cuda-platform.html#cuda-平台中的计算能力sm版本) 标识。

可以把多个 PTX 和 Cubin 目标嵌入应用程序或库中的单个二进制 [Fatbin](../01-introduction/cuda-platform.html#cuda-平台中的-cubin与-fatbin) 容器，从而让一个二进制文件支持多个虚拟 ISA 和目标硬件 ISA。

上面描述的工具调用与协调由 `nvcc` 自动完成。使用 `-v` 选项可以显示完整的编译工作流和工具调用；使用 `-keep` 选项可以保存编译过程中生成的[中间文件](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#keeping-intermediate-phase-files)，这些文件默认保存在当前目录，也可以通过 `--keep-dir` 指定保存目录。

下面的示例展示了 CUDA 源文件 `example.cu` 的编译工作流：

```cuda
// ----- example.cu -----
#include <stdio.h>
__global__ void kernel() {
    printf("Hello from kernel\n");
}

void kernel_launcher() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

int main() {
    kernel_launcher();
    return 0;
}
```

`nvcc` 的基本编译工作流如下图所示：

![nvcc 基本编译工作流](/images/chapter-02/nvcc-flow.png)

图示说明：源文件首先被拆分为 host code 与 device code；host code 交给 host 编译器，device code 则先编译为 PTX，再由 `ptxas` 为目标 GPU 生成 Cubin，最后由链接阶段组合为应用程序可以加载的结果。

包含多个 PTX 和 Cubin 架构时的 `nvcc` 编译工作流如下图所示：

![nvcc 多架构编译工作流](/images/chapter-02/nvcc-flow-multi-archs.png)

图示说明：同一份 CUDA 源代码可以针对多个虚拟架构和真实硬件架构生成 PTX/Cubin，并将多个目标放入一个 Fatbin。运行时可以选择与当前 GPU 匹配的 Cubin，或者在驱动中将合适的 PTX 即时编译为目标代码。

关于 `nvcc` 编译工作流的更详细说明，请参阅[编译器文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#the-cuda-compilation-trajectory)。

## 2.7.3 NVCC 基本用法（NVCC Basic Usage）

编译 CUDA 源文件的基本命令是：

```bash
nvcc <source_file>.cu -o <output_file>
```

`nvcc` 接受常见的编译器选项，用于指定头文件目录 `-I <path>` 和库目录 `-L <path>`、链接其他库 `-l<library>`，以及定义宏 `-D<macro>=<value>`。

```bash
nvcc example.cu -I path_to_include/ -L path_to_library/ -lcublas -o <output_file>
```

### 2.7.3.1 NVCC 生成 PTX 和 Cubin（NVCC PTX and Cubin Generation）

默认情况下，`nvcc` 会为 CUDA Toolkit 支持的最早 GPU 架构（最低的 `compute_XY` 和 `sm_XY` 版本）生成 PTX 和 Cubin，以尽可能扩大兼容性。

- `-arch` [选项](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-architecture-arch)可以用于为特定 GPU 架构生成 PTX 和 Cubin。
- `-gencode` [选项](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#generate-code-specification-gencode)可以用于为多个 GPU 架构生成 PTX 和 Cubin。

向 `nvcc` 传入 `--list-gpu-code` 和 `--list-gpu-arch` 选项，可以分别获得所有受支持的真实 GPU 架构和虚拟 GPU 架构列表；也可以参阅 `nvcc` 文档中的[虚拟架构列表](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list)和[ GPU 架构列表](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list)。

```bash
nvcc --list-gpu-code # 列出所有受支持的真实 GPU 架构
nvcc --list-gpu-arch # 列出所有受支持的虚拟 GPU 架构
```

```bash
nvcc example.cu -arch=compute_<XY> # 例如：-arch=compute_80，适用于 NVIDIA Ampere GPU 及后续 GPU
                                   # 仅生成 PTX，具有 GPU 前向兼容性

nvcc example.cu -arch=sm_<XY>      # 例如：-arch=sm_80，适用于 NVIDIA Ampere GPU 及后续 GPU
                                   # 生成 PTX 和 Cubin，具有 GPU 前向兼容性

nvcc example.cu -arch=native       # 自动检测当前 GPU 并为其生成 Cubin
                                   # 不生成 PTX，不具有 GPU 前向兼容性

nvcc example.cu -arch=all          # 为所有受支持的 GPU 架构生成 Cubin
                                   # 同时包含最新 PTX，以提供 GPU 前向兼容性

nvcc example.cu -arch=all-major    # 为所有受支持的主要 GPU 架构生成 Cubin，例如 sm_80、sm_90，
                                   # 同时包含最新 PTX，以提供 GPU 前向兼容性
```

更高级的用法允许单独指定 PTX 和 Cubin 目标：

```bash
# 为虚拟架构 compute_80 生成 PTX，并将其编译为真实架构 sm_86 的 Cubin，同时保留 compute_80 PTX
nvcc example.cu -arch=compute_80 -gpu-code=sm_86,compute_80 # （PTX 和 Cubin）

# 为虚拟架构 compute_80 生成 PTX，并将其编译为真实架构 sm_86、sm_89 的 Cubin
nvcc example.cu -arch=compute_80 -gpu-code=sm_86,sm_89     # （不包含 PTX）
nvcc example.cu -gencode=arch=compute_80,code=sm_86,sm_89 # 与上一条相同

# （1）为虚拟架构 compute_80 生成 PTX，并将其编译为真实架构 sm_86、sm_89 的 Cubin
# （2）为虚拟架构 compute_90 生成 PTX，并将其编译为真实架构 sm_90 的 Cubin
nvcc example.cu -gencode=arch=compute_80,code=sm_86,sm_89 -gencode=arch=compute_90,code=sm_90
```

关于控制 GPU code 生成的 `nvcc` 命令行选项的完整参考，请参阅 [nvcc 文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation)。

### 2.7.3.2 Host Code 编译说明（Host Code Compilation Notes）

不包含 device code 或 device symbol 的编译单元（即一个源文件及其头文件）可以直接使用 host 编译器编译。如果任何编译单元使用 CUDA Runtime API 函数，应用程序就必须链接 CUDA runtime library。CUDA runtime 同时提供静态库 `libcudart_static` 和共享库 `libcudart`。默认情况下，`nvcc` 链接静态 CUDA runtime library；要使用共享库版本，请在编译或链接命令中向 `nvcc` 传入 `--cudart=shared`。

`nvcc` 允许通过 `-ccbin <compiler>` 参数指定用于编译 host function 的 host 编译器。也可以定义环境变量 `NVCC_CCBIN`，以指定 `nvcc` 使用的 host 编译器。`nvcc` 的 `-Xcompiler` 参数会将参数传递给 host 编译器。例如，在下面的示例中，`-O3` 参数由 `nvcc` 传递给 host 编译器。

```bash
nvcc example.cu -ccbin=clang++

export NVCC_CCBIN='gcc'
nvcc example.cu -Xcompiler=-O3
```

### 2.7.3.3 GPU Code 的分离编译（Separate Compilation of GPU Code）

`nvcc` 默认使用*整程序编译（whole-program compilation）*，这要求使用某个 GPU code 或 symbol 的编译单元中包含该 GPU code 和 symbol 的全部定义。CUDA device function 可以调用其他编译单元中定义的 device function，或者访问其他编译单元中定义的 device variable；但必须在 `nvcc` 命令行中指定 `-rdc=true` 或其别名 `-dc`，以启用来自不同编译单元的 device code 链接。能够从不同编译单元链接 device code 和 symbol 的能力称为*分离编译（separate compilation）*。

分离编译允许更灵活地组织代码，可以缩短编译时间，并可能生成更小的二进制文件。与整程序编译相比，分离编译可能增加构建时的复杂性。由于 device code 链接会影响性能，因此默认不启用。启用[链接时优化（Link-Time Optimization，LTO）](#27.4.4-链接时优化link-time-optimization-lto)可以帮助降低分离编译的性能开销。

分离编译需要满足以下条件：

- 在一个编译单元中定义的非 `const` device variable，在其他编译单元中引用时必须使用 `extern` 关键字。
- 所有 `const` device variable 都必须使用 `extern` 关键字定义和引用。
- 所有 CUDA 源文件 `.cu` 都必须使用 `-dc` 或 `-rdc=true` 选项编译。

host function 和 device function 默认具有 external linkage，不需要 `extern` 关键字。注意，[从 CUDA 13 开始](https://developer.nvidia.com/blog/cuda-c-compiler-updates-impacting-elf-visibility-and-linkage/)，`__global__` function 以及 `__managed__`、`__device__`、`__constant__` variable 默认具有 internal linkage。

下面的示例中，`definition.cu` 定义了一个 variable 和一个 function，`example.cu` 则引用它们。两个文件分别编译，之后链接到最终二进制文件中。

```cuda
// ----- definition.cu -----
extern __device__ int device_variable = 5;
__device__        int device_function() { return 10; }
```

```cuda
// ----- example.cu -----
extern __device__ int  device_variable;
__device__        int device_function();

__global__ void kernel(int* ptr) {
    device_variable = 0;
    *ptr            = device_function();
}
```

```bash
nvcc -dc definition.cu -o definition.o
nvcc -dc example.cu    -o example.o
nvcc definition.o example.o -o program
```

## 2.7.4 常用编译器选项（Common Compiler Options）

本节介绍最相关的 `nvcc` 编译器选项，覆盖语言特性、优化、调试、性能分析和构建相关方面。所有选项的完整说明请参阅 [nvcc 文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#command-option-description)。

### 2.7.4.1 语言特性（Language Features）

`nvcc` 支持从 C++03 到 [C++23 语言特性](../05-technical-appendices/cpp-language-support.html#cpp23-language-features)的 C++ 核心语言特性。可以使用 `-std` 选项指定要使用的语言标准：

- `--std={c++03|c++11|c++14|c++17|c++20|c++23}`

此外，`nvcc` 支持以下语言扩展：

- `-restrict`：断言所有 kernel pointer parameter 都是 [`restrict`](../05-technical-appendices/cpp-language-extensions.html#restrict) pointer。
- `-extended-lambda`：允许在 lambda 声明中使用 `__host__`、`__device__` 注解。
- `-expt-relaxed-constexpr`：（实验性选项）允许 host code 调用 `__device__ constexpr` function，并允许 device code 调用 `__host__ constexpr` function。

关于这些特性的更多细节，请参阅[扩展 lambda](../05-technical-appendices/cpp-language-support.html#extended-lambdas)和 [`constexpr` function](../05-technical-appendices/cpp-language-support.html#constexpr-functions)章节。

### 2.7.4.2 调试选项（Debugging Options）

`nvcc` 支持以下用于生成调试信息的选项：

- `-g`：为 host code 生成调试信息。`gdb`、`lldb` 及类似工具依赖这些信息调试 host code。
- `-G`：为 device code 生成调试信息。[`cuda-gdb`](https://docs.nvidia.com/cuda/cuda-gdb/index.html) 依赖这些信息调试 device code。该选项还会定义 `__CUDACC_DEBUG__` 宏。
- `-lineinfo`：为 device code 生成行号信息。该选项不会影响执行性能，并且与 [Compute Sanitizer](https://developer.nvidia.com/compute-sanitizer) 一起使用时，可以追踪 kernel 执行。

默认情况下，`nvcc` 对 GPU code 使用最高优化级别 `-O3`。调试选项 `-G` 会阻止一部分编译器优化，因此调试代码的性能预计低于非调试代码。还可以定义 `-DNDEBUG` 宏来禁用 runtime assertion，因为 assertion 也可能降低执行速度。

### 2.7.4.3 优化选项（Optimization Options）

`nvcc` 提供许多用于优化性能的选项。本节简要介绍开发者可能有用的一些选项，并提供进一步信息的链接；完整说明请参阅 [nvcc 文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)。

- `-Xptxas` 将参数传递给 PTX assembler 工具 `ptxas`。`nvcc` 文档提供了 [有用的 `ptxas` 参数列表](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options)。例如，`-Xptxas=-maxrregcount=N` 指定每个线程可以使用的最大寄存器数量。
- `-extra-device-vectorization`：启用更激进的 device code 向量化。
- `--apply-controls=/path/to/file`：将高级控制文件（Advanced Controls File，ACF）传给 `nvcc` 和 `ptxas`。该文件会改变默认编译行为，使编译更有针对性地服务于特定工作负载。使用高级控制文件可能导致编译失败或运行时执行错误，请自行承担风险。关于如何生成高级控制文件，请参阅 [CompileIQ GitHub 页面](https://github.com/NVIDIA/CompileIQ)。
- 提供浮点行为细粒度控制的其他选项，见[浮点计算](../05-technical-appendices/mathematical-functions.html#floating-point-computation)章节和 [`nvcc` 文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math)。

下面这些选项会输出编译器信息，可用于更高级的代码优化：

- `-res-usage`：编译结束后输出资源使用报告，其中包含每个 kernel function 分配的寄存器、shared memory、constant memory 和 local memory 数量。
- `-opt-info=inline`：输出有关函数内联的信息。
- `-Xptxas=-warn-lmem-usage`：使用 local memory 时发出警告。
- `-Xptxas=-warn-spills`：寄存器被溢出到 local memory 时发出警告。

### 2.7.4.4 链接时优化（Link-Time Optimization，LTO）

[分离编译](#2733-gpu-code-的分离编译separate-compilation-of-gpu-code)由于跨文件优化机会有限，可能比整程序编译产生更低的性能。链接时优化（Link-Time Optimization，LTO）在链接阶段跨分别编译的文件执行优化，但代价是增加编译时间。LTO 可以在保持分离编译灵活性的同时，恢复整程序编译的大部分性能。

`nvcc` 需要 `-dlto` [选项](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#dlink-time-opt-dlto)，或需要使用 `lto_<SM version>` 链接时优化目标，才能启用 LTO：

```bash
nvcc -dc -dlto -arch=sm_100 definition.cu -o definition.o
nvcc -dc -dlto -arch=sm_100 example.cu    -o example.o
nvcc -dlto definition.o example.o -o program
```

```bash
nvcc -dc -arch=lto_100 definition.cu -o definition.o
nvcc -dc -arch=lto_100 example.cu    -o example.o
nvcc -dlto definition.o example.o -o program
```

### 2.7.4.5 性能分析选项（Profiling Options）

可以直接使用 [Nsight Compute](https://developer.nvidia.com/nsight-compute) 和 [Nsight Systems](https://developer.nvidia.com/nsight-systems) 工具分析 CUDA 应用，而不需要在编译过程中增加额外选项。不过，`nvcc` 生成的额外信息可以通过将源文件与生成的代码关联起来，帮助性能分析：

- `-lineinfo`：为 device code 生成行号信息，使性能分析工具能够显示源代码。性能分析工具要求原始源代码仍位于编译代码时的同一路径。
- `-src-in-ptx`：将原始源代码保留在 PTX 中，从而避免 `-lineinfo` 上述路径限制。该选项要求同时使用 `-lineinfo`。

### 2.7.4.6 Fatbin 压缩（Fatbin Compression）

默认情况下，`nvcc` 会压缩存储在应用程序或库二进制文件中的 [fatbin](../01-introduction/cuda-platform.html#cuda-平台中的-cubin与-fatbin)。可以使用下面的选项控制 fatbin 压缩：

- `-no-compress`：禁用 fatbin 压缩。
- `--compress-mode={default|size|speed|balance|none}`：设置压缩模式。`speed` 关注快速解压，`size` 关注减小 fatbin 大小，`balance` 在速度和大小之间折中，默认模式为 `speed`，`none` 禁用压缩。

### 2.7.4.7 编译器性能控制（Compiler Performance Controls）

`nvcc` 提供了用于分析和加速编译器自身过程的选项：

- `-t <N>`：针对多个 GPU 架构编译单个编译单元时，用于并行化编译的 CPU 线程数。
- `-split-compile <N>`：用于并行化优化阶段的 CPU 线程数。
- `-split-compile-extended <N>`：更激进的分离编译形式，需要链接时优化。
- `-Ofc <N>`：device code 编译速度级别。
- `-time <filename>`：生成 CSV 表格，记录每个编译阶段耗费的时间。
- `-fdevice-time-trace`：生成 device code 编译的时间轨迹。
