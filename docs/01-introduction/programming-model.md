---
title: 1.2 编程模型
description: CUDA 异构系统、GPU 硬件模型、线程层次、SIMT、tile 与内存的中文翻译
---

# 1.2 编程模型（Programming Model）

> 本页按 NVIDIA CUDA Programming Guide Release 13.3 的在线页面逐段翻译。这里介绍的术语和概念与具体语言无关，后续章节会用 C++ 展示它们。原文页面：[1.2. Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)；官方页面标注的更新时间为 2026-05-27。

## 1.2.1 异构系统（Heterogeneous Systems）

CUDA 编程模型假定系统是一个异构计算系统，也就是同时包含 GPU 和 CPU 的系统。CPU 以及与 CPU 直接连接的内存分别称为 **host** 和 **host memory**；GPU 以及与 GPU 直接连接的内存分别称为 **device** 和 **device memory**。在某些片上系统（system-on-chip，SoC）中，它们可能位于同一个封装内；在更大的系统中，可能存在多个 CPU 或 GPU。

CUDA 应用会把部分代码放到 GPU 上执行，但应用总是从 CPU 开始执行。运行在 CPU 上的 host code 可以调用 CUDA API，在 host memory 与 device memory 之间复制数据，启动 GPU 上的代码，并等待数据复制或 GPU 代码完成。CPU 和 GPU 可以同时执行代码；要获得最佳性能，通常应尽量提高 CPU 与 GPU 的利用率。

应用在 GPU 上执行的代码称为 **device code**。被调用到 GPU 上执行的函数出于历史原因称为 **kernel**；启动该函数的动作称为 **launching the kernel**。一次 kernel launch 可以理解为让大量线程在 GPU 上并行执行 kernel 代码。

GPU 线程的工作方式与 CPU 线程相似，但在正确性和性能方面存在一些重要差异，后续章节会进一步介绍这些差异。

## 1.2.2 GPU 硬件模型（GPU Hardware Model）

和任何编程模型一样，CUDA 也依赖于对底层硬件的概念模型。就 CUDA 编程而言，可以把 GPU 看作许多流式多处理器（Streaming Multiprocessor，SM）的集合；这些 SM 又被组织成称为图形处理集群（Graphics Processing Cluster，GPC）的组。每个 SM 都包含本地寄存器文件、统一数据缓存和执行计算的若干功能单元。统一数据缓存为 shared memory 和 L1 cache 提供物理资源。

统一数据缓存如何分配给 L1 和 shared memory，可以在运行时配置。不同 GPU 架构中的各种内存容量以及 SM 内功能单元的数量可能不同。

::: tip 注意
实际 GPU 的硬件布局，或者硬件实际执行编程模型的方式，可能有所不同。这些差异不应影响遵循 CUDA 编程模型编写的软件的正确性。
:::

![图 2：CUDA 编程模型中的 GPU、CPU 与内存连接](/images/chapter-01/figure-02-gpu-cpu-system-diagram.png)

<div class="figure-caption"><strong>图 2：GPU、GPC、SM 与 CPU 的关系。</strong> 原 PDF 第 22 页。GPU 由多个 GPC 组成，GPC 组织多个 SM；CPU 和 GPU 通过 PCIe 或 NVLink 等互连相连。图中展示的是编程模型视角，而非固定的物理版图。</div>

### 1.2.2.1 线程块与网格（Thread Blocks and Grids）

应用启动 kernel 时，会使用许多线程，数量往往可以达到数百万。这些线程被组织成 block；一个线程组称为 **thread block（线程块）**。thread block 再被组织成 **grid（网格）**。同一个 grid 中的所有 thread block 具有相同的大小和维度。下图展示了一个由 thread block 组成的 grid。

![图 3：线程块组成的网格](/images/chapter-01/figure-03-grid-of-thread-blocks.png)

<div class="figure-caption"><strong>图 3：线程块组成的网格。</strong> 原 PDF 第 22 页。每个箭头表示一个线程，箭头数量不代表真实的线程数量。</div>

thread block 和 grid 都可以是一维、二维或三维。这些维度可以简化把单个线程映射到工作单元或数据元素的过程。

启动 kernel 时，需要使用一个特定的执行配置（execution configuration），其中指定 grid 和 thread block 的维度。执行配置还可以包含 cluster size、stream 和 SM 配置等可选参数，这些参数会在后续章节介绍。

每个执行 kernel 的线程都可以通过内置变量确定自己在所属 block 中的位置，以及所属 block 在 grid 中的位置。线程也可以通过这些内置变量确定启动 kernel 时 thread block 和 grid 的维度。因此，每个线程都能在执行该 kernel 的全部线程中确定唯一身份；程序经常使用这个身份来决定线程负责的数据或操作。

一个 thread block 中的所有线程都在同一个 SM 上执行。这使得同一 thread block 内的线程可以高效地通信和同步。thread block 中的所有线程都可以访问片上 shared memory，用来交换 block 内线程之间的信息。

一个 grid 可能包含数百万个 thread block，而执行这个 grid 的 GPU 可能只有几十个或几百个 SM。一个 thread block 中的所有线程都在单个 SM 上执行，并且在大多数情况下[^1]会在该 SM 上运行到完成。不同 thread block 之间没有调度顺序保证，因此一个 thread block 不能依赖另一个 thread block 的结果，因为另一个 block 可能要等当前 block 完成后才能被调度。

![图 4：SM 上的线程块调度](/images/chapter-01/figure-04-thread-block-scheduling.png)

<div class="figure-caption"><strong>图 4：SM 上的线程块调度。</strong> 原 PDF 第 24 页。在示例中，每个 SM 同时调度三个 thread block；grid 中的 thread block 被分配到 SM 的顺序没有保证。</div>

CUDA 编程模型允许任意大小的 grid 在任意规模的 GPU 上运行，无论 GPU 只有一个 SM 还是拥有数千个 SM。为实现这一点，除少数例外情况外，CUDA 编程模型要求不同 thread block 中的线程之间不能存在未处理的数据依赖。也就是说，一个线程不应依赖同一个 grid 中另一个 thread block 的线程的结果，也不应与其同步。

grid 中不同的 thread block 会在可用的 SM 之间调度，执行顺序可以任意。简言之，CUDA 编程模型要求 thread block 能够按任意顺序执行：既可以并行，也可以串行。

#### 1.2.2.1.1 线程块集群（Thread Block Clusters）

除了 thread block 之外，计算能力（compute capability）为 9.0 及更高版本的 GPU 还支持一种可选的分组层级，称为 **cluster**。和 thread block、grid 一样，cluster 也可以按一维、二维或三维排列。下图展示了同时按 cluster 组织的 thread block grid。指定 cluster 不会改变 grid 的维度，也不会改变 thread block 在 grid 中的索引。

![图 5：网格中的线程块集群](/images/chapter-01/figure-05-grid-of-clusters.png)

<div class="figure-caption"><strong>图 5：线程块集群。</strong> 原 PDF 第 25 页。指定 cluster 后，thread block 除了处于 grid 中的位置外，还拥有相对于所属 cluster 的位置。</div>

指定 cluster 会把相邻的 thread block 分组，并提供 cluster 级别的同步与通信机会。具体来说，一个 cluster 中的所有 thread block 都在同一个 GPC 中执行。

![图 6：GPC 内的线程块集群调度](/images/chapter-01/figure-06-thread-block-scheduling-with-clusters.png)

<div class="figure-caption"><strong>图 6：GPC 内的线程块集群调度。</strong> 原 PDF 第 26 页。同一个 cluster 中的 thread block 会以 cluster 的形状排列在 grid 中，并同时调度到单个 GPC 的多个 SM 上。</div>

由于这些 thread block 会在单个 GPC 内同时调度，同一 cluster 中不同 block 的线程可以使用 [Cooperative Groups](../04-cuda-features/cooperative-groups.html) 提供的软件接口相互通信和同步。cluster 中的线程可以访问 cluster 内所有 block 的 shared memory，这种内存称为 **distributed shared memory**。cluster 的最大规模取决于硬件，并且不同设备可能不同。

### 1.2.2.2 Warp 与 SIMT（Warps and SIMT）

在一个 thread block 内，线程会被组织成每组 32 个线程的 **warp**。warp 采用单指令多线程（Single-Instruction Multiple-Threads，SIMT）范式执行 kernel 代码。在 SIMT 中，warp 内所有线程都执行同一段 kernel 代码，但每个线程可以在代码中沿着不同的分支执行。也就是说，程序中的所有线程执行相同的代码，但它们不必遵循相同的执行路径。

当线程由 warp 执行时，每个线程都会分配到一个 warp lane。warp lane 的编号为 0 到 31；thread block 中的线程会按一种可预测的方式分配到 warp，具体细节见 [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced-cuda/advanced-kernel-programming.html)。

warp 内所有线程同时执行同一条指令。如果 warp 中一部分线程在执行时沿着控制流分支前进而另一部分线程没有走该分支，那么不走该分支的线程会被 mask 掉，先执行走该分支的线程。例如，如果某个条件只对 warp 中一半线程为真，那么另一半线程会被 mask，活跃线程执行这些指令。下图展示了这种情况。

![图 7：不活跃 warp lane 被 mask](/images/chapter-01/figure-07-active-warp-lanes.png)

<div class="figure-caption"><strong>图 7：warp 分支中的 lane mask。</strong> 原 PDF 第 27 页。示例中只有线程索引为偶数的线程执行 `if` 体，其他线程在该路径执行期间被 mask。</div>

当 warp 中不同线程沿着不同代码路径执行时，这种现象有时称为 **warp divergence（warp 分歧）**。因此，当 warp 内线程遵循相同的控制流路径时，GPU 的利用率通常最高。

在 SIMT 模型中，warp 内所有线程以锁步方式推进 kernel。硬件的实际执行方式可能不同；关于这种区别在哪些场景重要，请参阅 [Independent Thread Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced-cuda/advanced-kernel-programming.html)。不建议利用 warp 执行如何映射到真实硬件这一层面的知识。CUDA 编程模型和 SIMT 模型只保证 warp 中所有线程共同推进代码。

只要程序遵循编程模型，硬件可以用对程序透明的方式优化被 mask 的 lane。如果程序违反这一模型，就可能产生未定义行为，而且这种行为可能在不同 GPU 硬件上不同。

编写 CUDA 代码时不一定需要显式考虑 warp，但理解 warp 执行模型有助于理解 [global memory coalescing](https://docs.nvidia.com/cuda/cuda-programming-guide/02-programming-gpus/writing-simt-kernels.html) 和 [shared memory bank access patterns](https://docs.nvidia.com/cuda/cuda-programming-guide/02-programming-gpus/writing-simt-kernels.html) 等概念。一些高级编程技术会利用 thread block 内的 warp 分组来减少线程分歧并提高利用率；其他优化也会利用线程按 warp 分组执行这一事实。

warp 执行带来的一个含义是：thread block 的总线程数最好设为 32 的倍数。使用其他数量在语法和语义上都是合法的，但当线程总数不是 32 的倍数时，thread block 的最后一个 warp 会有一些 lane 在整个执行过程中都未被使用。这很可能导致该 warp 的功能单元利用率不理想。

> SIMT 经常与单指令多数据（Single Instruction Multiple Data，SIMD）并行进行比较，但两者有重要差异。在 SIMD 中，执行遵循一条控制流路径；在 SIMT 中，每个线程都可以遵循自己的控制流路径。因此，SIMT 没有像 SIMD 那样固定的数据宽度。关于 SIMT 的更详细讨论见 [SIMT Execution Model](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced-cuda/advanced-kernel-programming.html)。

### 1.2.2.3 CUDA 中的 Tile 编程（Tile Programming in CUDA）

除了前面介绍的 SIMT 模型之外，CUDA 还支持 **tile programming** 模型。在 tile 编程中，程序员以整个 thread block 为层级编写代码，描述对称为 **tile** 的多维数据集合执行的操作；编译器会把这些操作映射到 block 中的单个线程。

tile kernel 会在由 block 组成的 grid 上启动，方式与前面的 [线程块与网格](#1221-线程块与网格-thread-blocks-and-grids) 相同。每个 block 执行 tile kernel，并且可以查询自己在 grid 中的位置，从而确定自己负责数据的哪一部分。程序员只指定 grid 的维度；每个 block 的线程数量由编译器根据 kernel 中的 tile 操作决定。

![图 8：SIMT 与 tile 编程模型中的程序员视角](/images/chapter-01/figure-08-tile-simt.png)

<div class="figure-caption"><strong>图 8：SIMT 与 tile 编程模型中的程序员视角。</strong> 原 PDF 第 27 页。在 SIMT 中，程序员编写逐线程代码并控制每个线程如何访问数据；在 tile 编程中，程序员编写逐 block 操作 tile 的代码，编译器再把操作映射到 block 内线程。</div>

在 tile kernel 内，block 执行单一控制流。程序员指定对 tile 执行的操作，编译器把工作分配给 block 内的线程。条件和循环等标准控制流结构仍然受到支持，但由于整个 block 遵循单一控制流，所以不存在 warp divergence 的概念。计算索引或循环边界等标量操作由 block 中的单个线程执行。

逐元素相加两个 tile 等 tile 操作，则由 block 内的所有线程共同并行执行。

不要把 block（执行单位）和 tile（数据单位）混淆。单个 block 可以创建并操作许多形状和数据类型不同的 tile。

#### 1.2.2.3.1 数组与 tile（Arrays and tiles）

tile kernel 处理两种数据：array 和 tile。array（或 global array）是一个多维元素容器，存储在 device memory 中。array 是可变的：kernel 内的 store 操作可以修改它的内容。array 具有形状和数据类型。

tile 是一个只存在于 tile 代码中的多维值集合，并且局部属于单个 block。tile 是不可变的：每个对 tile 的操作都会产生一个新 tile，而不是修改已有 tile。与 array 不同，tile 不一定在内存中有对应表示；编译器决定 tile 数据如何存储，可以使用寄存器、shared memory 或 SM 的其他资源。

tile 的每个维度都必须是 2 的幂，并且必须在编译期已知，也就是在 kernel 执行前就能确定，不能等到执行期间再计算。tile 不能作为 kernel 参数传递；它们完全在 tile 代码中创建和消费。

#### 1.2.2.3.2 Tile 空间与数据移动（Tile space and data movement）

数据通过 load 和 store 操作在 array 与 tile 之间移动。这些操作使用 **tile space** 的概念。tile space 是把一个 array 概念上划分成大小相等且互不重叠的 tile 后得到的空间。

例如，有一个形状为 `(M, N)` 的二维 array。如果 load 操作指定的 tile 形状为 `(t_m, t_n)`，那么这个 array 在概念上会被划分成 `ceil(M / t_m)` 行和 `ceil(N / t_n)` 列的 tile。

tile space 的索引（例如 `(i, j)`）标识要加载哪个 tile。load 会返回一个形状为 `(t_m, t_n)` 的 tile，其中包含 array 对应位置的元素。当 tile 超出 array 的边界时——例如 array 的维度不是 tile 维度的整数倍，在边缘就会出现这种情况——load 会指定如何处理越界元素，例如用零填充。

![图 9：Tile 空间与数据移动](/images/chapter-01/figure-09-tile-data-movement.png)

<div class="figure-caption"><strong>图 9：tile 空间与数据移动。</strong> 原 PDF 第 28 页。形状为 `(M, N)` 的二维 array 被概念上划分为形状为 `(t_m, t_n)` 的 tile 网格。对 tile space 索引 `(i, j)` 执行 load 会返回对应 tile；在 array 边界之外的元素可以用零填充。store 会在给定索引处把 tile 写回 array。</div>

store 操作执行相反的过程：给定一个 tile 和 tile space 中的索引，它会把 tile 的元素写入 array 的对应区域。写入超出 array 边界的元素会被静默丢弃。tile 程序还支持 gather 和 scatter 操作，它们可以从 array 的任意位置加载，或向 array 的任意位置存储。

#### 1.2.2.3.3 Tile 上的操作（Operations on tiles）

tile 程序提供了一组作用于 tile 的内置操作，包括逐元素算术、矩阵乘法、沿一个或多个轴执行的归约（例如求和与求最大值）、形状操作（例如 reshape 和 transpose），以及类型转换。当一个操作组合两个形状不同的 tile 时，较小的 tile 会先自动扩展到与较大的 tile 匹配，然后再执行操作。

#### 1.2.2.3.4 与 SIMT 编程的关系（Relationship to SIMT programming）

tile 编程与 SIMT 编程在 CUDA 中共存。一个应用可以同时包含 SIMT kernel 和 tile kernel，两类 kernel 都可以操作 device memory 中的同一份数据。采用哪种编程模型是针对每个 kernel 单独决定的。tile 编程不会取代 SIMT 编程；SIMT 为单个线程提供细粒度控制，这对某些算法和优化技术仍然是必要的。

tile 编程提供了更高层的抽象，可以简化 kernel 开发。由于线程级决策交给编译器处理，同一个 tile kernel 可以运行在不同 GPU 架构上，而不需要修改源代码。这两种模型都建立在前面介绍的同一套底层硬件之上：SM、thread block 和 grid。两种模型也使用相同的 device memory 空间，下面将介绍这些空间。

## 1.2.3 GPU 内存（GPU Memory）

在现代计算系统中，高效利用内存与最大化计算功能单元的利用率同样重要。异构系统包含多个内存空间，GPU 除了 cache 外，还包含多种可编程的片上内存。下面的章节会更详细地介绍这些内存空间。

### 1.2.3.1 异构系统中的 DRAM 内存（DRAM Memory in Heterogeneous Systems）

GPU 和 CPU 都有直接连接的 DRAM 芯片。在包含多张 GPU 的系统中，每张 GPU 都有自己的内存。从 device code 的角度看，连接到 GPU 的 DRAM 称为 **global memory**，因为 GPU 中的所有 SM 都可以访问它。这个术语并不意味着它一定可以在系统的所有位置访问。连接到 CPU 的 DRAM 称为 **system memory** 或 **host memory**。

和 CPU 一样，GPU 使用虚拟内存寻址。在目前支持的所有系统上，CPU 与 GPU 使用一个统一的虚拟内存空间。这意味着系统中每张 GPU 的虚拟地址范围都唯一且不同于 CPU 和其他 GPU。给定一个虚拟内存地址，可以确定该地址位于 GPU 内存还是系统内存；在多 GPU 系统中，还可以确定包含该地址的是哪张 GPU 的内存。

CUDA 提供了分配 GPU 内存、CPU 内存以及在 CPU 与 GPU 之间、GPU 内部或多 GPU 系统中的 GPU 之间复制分配内容的 API。需要时，程序可以显式控制数据的 locality。下面介绍的 [Unified Memory](../04-cuda-features/unified-memory.html) 允许 CUDA runtime 或系统硬件自动处理内存放置。

### 1.2.3.2 GPU 中的片上内存（On-Chip Memory in GPUs）

除 global memory 外，每张 GPU 还包含一些片上内存。每个 SM 都有自己的寄存器文件和 shared memory。这些内存属于 SM，SM 内执行的线程可以极快地访问它们。

寄存器文件存储线程局部变量，这些变量通常由编译器分配。shared memory 可以被 thread block 或 cluster 中的所有线程访问，可用于在 thread block 或 cluster 的线程之间交换数据。

SM 中的寄存器文件和统一数据缓存容量有限。SM 的寄存器文件、统一数据缓存的大小，以及统一数据缓存如何配置 L1 与 shared memory 的比例，可以在 [Memory Information per Compute Capability](https://docs.nvidia.com/cuda/cuda-programming-guide/05-technical-appendices/compute-capabilities.html) 中查到。寄存器文件、shared memory 空间和 L1 cache 由 thread block 中的所有线程共享。

要把一个 thread block 调度到 SM 上，该 block 所需的每线程寄存器数量乘以 block 中的线程数量，必须小于或等于 SM 中可用的寄存器数量。如果一个 thread block 所需的寄存器数超过寄存器文件容量，kernel 就无法启动；必须减少 thread block 中的线程数量，才能使该 block 可以启动。

shared memory 按 thread block 的层级分配。也就是说，与按线程分配的寄存器不同，shared memory 的分配由整个 thread block 共同使用。

#### 1.2.3.2.1 Cache（Caches）

除了可编程内存外，GPU 还拥有 L1 和 L2 cache。每个 SM 都有一个 L1 cache，它是统一数据缓存的一部分；更大的 L2 cache 则由 GPU 内的所有 SM 共享。这一点可以在图 2 所示的 GPU 方框图中看到。

每个 SM 还拥有单独的 **constant cache**，用于缓存 global memory 中那些在一个 kernel 生命周期内被声明为常量的值。编译器也可能把 kernel 参数放入 constant memory。这样可以让 kernel 参数在 SM 中独立于 L1 data cache 被缓存，从而改善 kernel 性能。

### 1.2.3.3 统一内存（Unified Memory）

当应用显式在 GPU 或 CPU 上分配内存时，该内存通常只能被在对应设备上运行的代码访问。也就是说，CPU 内存只能由 CPU 代码访问，GPU 内存只能由 GPU 上运行的 kernel 访问[^2]。CUDA API 用于在 CPU 和 GPU 之间复制内存，以便在正确的时间把数据复制到正确的内存中。

CUDA 的 unified memory 功能允许应用创建可由 CPU 或 GPU 访问的内存分配。CUDA runtime 或底层硬件会在需要时启用访问，或把数据迁移到正确的位置。即使使用 unified memory，要获得最佳性能仍应尽量减少内存迁移，并尽可能让直接连接到数据所在内存的处理器访问这些数据。

系统的硬件特性决定了不同内存空间之间如何访问和交换数据。[Unified Memory](../04-cuda-features/unified-memory.html) 一节介绍不同类别的 unified memory 系统；[第二部分的 Unified and System Memory](../02-programming-gpus/unified-and-system-memory.html) 则详细介绍所有场景中的使用方式和行为。

[^1]: 在使用 CUDA Dynamic Parallelism 等功能时，某些情况下 thread block 可能会被挂起到内存中。这意味着 SM 的状态会被保存到由系统管理的 GPU 内存区域，SM 被释放出来执行其他 thread block。这类似于 CPU 上的上下文切换，但并不常见。
[^2]: 一个例外是 **mapped memory**：它是带有特殊属性、允许 GPU 直接访问的 CPU 内存。可是这种访问需要经过 PCIe 或 NVLink 连接，GPU 无法用并行性隐藏更高的延迟和更低的带宽。因此，mapped access 不是 unified memory 的高性能替代方案，也不能替代把数据放在合适的内存空间中。

## 本节导航

- [1.1 引言](./introduction.html)
- [1.3 CUDA 平台](./cuda-platform.html)
- [PDF 图版：第 1 章图 2–10](../figures.html)
