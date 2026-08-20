---
title: 4.1 统一内存（Unified Memory）
description: CUDA 统一内存编程范式、访问一致性、迁移机制与性能调优
---

# 4.1 统一内存（Unified Memory）

本节详细说明 CUDA 提供的不同统一内存范式的行为和用法。[前面的统一内存章节](../02-programming-gpus/unified-and-system-memory.html)介绍了如何判断适用的统一内存范式，并对每种范式作了简要说明。

如前所述，统一内存编程有四种范式：

- 显式托管内存分配的完整支持；
- 采用软件一致性的所有分配的完整支持；
- 采用硬件一致性的所有分配的完整支持；
- 受限的统一内存支持。

前面三种具有完整统一内存支持的范式，其行为和编程模型非常相近，本节将在[具有完整 CUDA 统一内存支持的设备上的统一内存](#4-1-1-具有完整-cuda-统一内存支持的设备上的统一内存)中介绍，并突出它们之间的差异。

最后一种统一内存支持受限的范式，将在[Windows、WSL 和 Tegra 上的统一内存](#4-1-3-windows-wsl-和-tegra-上的统一内存)中详细讨论。

## 4.1.1 具有完整 CUDA 统一内存支持的设备上的统一内存（Unified Memory on Devices with Full CUDA Unified Memory Support）

这类系统包括具有硬件一致性内存系统的系统，例如 NVIDIA Grace Hopper，以及启用了异构内存管理（Heterogeneous Memory Management，HMM）的现代 Linux 系统。HMM 是一种基于软件的内存管理系统，但提供与硬件一致性内存系统相同的编程模型。

Linux HMM 要求 Linux 内核版本为 6.1.24+、6.2.11+ 或 6.3+，设备计算能力为 7.5 或更高，CUDA 驱动版本为 535+，并安装[开放内核模块](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/kernel-modules.html#open-gpu-kernel-modules-installation)。

> **注意**
>
> 我们将 CPU 与 GPU 共用一套页表的系统称为*硬件一致性*系统。CPU 与 GPU 使用独立页表的系统称为*软件一致性*系统。

诸如 NVIDIA Grace Hopper 这样的硬件一致性系统，为 CPU 和 GPU 提供逻辑上合并的页表，参见[CPU 与 GPU 页表：硬件一致性与软件一致性](#4-1-1-2-1-2-cpu-和-gpu-页表硬件一致性与软件一致性)。以下小节只适用于硬件一致性系统：

- [访问计数器迁移](#4-1-1-2-7-访问计数器迁移)

### 4.1.1.1 统一内存：深入示例（Unified Memory: In-Depth Examples）

具有完整 CUDA 统一内存支持的系统（见[统一内存范式概览](../02-programming-gpus/unified-and-system-memory.html#4-1-1-统一内存范式)）允许设备访问与其交互的 host 进程所拥有的任意内存。

本节展示若干高级用例。所有示例都使用一个 kernel，将输入字符数组的前 8 个字符简单打印到标准输出流：

```cpp
__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}
```

下面的各个选项展示了调用这个 kernel、并向它传入由系统分配的内存的不同方式。

#### 使用 `malloc`

```cpp
void test_malloc() {
  const char test_string[] = "Hello World";
  char* heap_data = (char*)malloc(sizeof(test_string));
  strncpy(heap_data, test_string, sizeof(test_string));
  kernel<<<1, 1>>>("malloc", heap_data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  free(heap_data);
}
```

#### 使用托管内存（Managed）

```cpp
void test_managed() {
  const char test_string[] = "Hello World";
  char* data;
  cudaMallocManaged(&data, sizeof(test_string));
  strncpy(data, test_string, sizeof(test_string));
  kernel<<<1, 1>>>("managed", data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  cudaFree(data);
}
```

#### 使用栈变量（Stack variable）

```cpp
void test_stack() {
  const char test_string[] = "Hello World";
  kernel<<<1, 1>>>("stack", test_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

#### 使用文件作用域静态变量（File-scope static variable）

```cpp
void test_static() {
  static const char test_string[] = "Hello World";
  kernel<<<1, 1>>>("static", test_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

#### 使用全局作用域变量（Global-scope variable）

```cpp
const char global_string[] = "Hello World";

void test_global() {
  kernel<<<1, 1>>>("global", global_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

#### 使用全局作用域 `extern` 变量（Global-scope extern variable）

```cpp
// declared in separate file, see below
extern char* ext_data;

void test_extern() {
  kernel<<<1, 1>>>("extern", ext_data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

```cpp
/** This may be a non-CUDA file */
char* ext_data;
static const char global_string[] = "Hello World";

void __attribute__ ((constructor)) setup(void) {
  ext_data = (char*)malloc(sizeof(global_string));
  strncpy(ext_data, global_string, sizeof(global_string));
}

void __attribute__ ((destructor)) tear_down(void) {
  free(ext_data);
}
```

对于 `extern` 变量，需要注意：它可以由第三方库声明、拥有和管理，而该库完全不与 CUDA 交互。

还要注意，栈变量、文件作用域变量和全局作用域变量只能由 GPU 通过指针访问。在这个特定示例中，这很方便，因为字符数组本身已经声明为指针：`const char*`。但是，考虑下面的全局作用域整数示例：

```cpp
// this variable is declared at global scope
int global_variable;

__global__ void kernel_uncompilable() {
  // this causes a compilation error: global (__host__) variables must not
  // be accessed from __device__ / __global__ code
  printf("%d\n", global_variable);
}

// On systems with pageableMemoryAccess set to 1, we can access the address
// of a global variable. The below kernel takes that address as an argument
__global__ void kernel(int* global_variable_addr) {
  printf("%d\n", *global_variable_addr);
}
int main() {
  kernel<<<1, 1>>>(&global_variable);
  ...
  return 0;
}
```

上例中，必须向 kernel 传入全局变量的*指针*，而不能在 kernel 中直接访问全局变量。这是因为没有 `__managed__` 修饰符的全局变量默认声明为仅供 `__host__` 使用，因此目前大多数编译器不允许在 device 代码中直接使用这类变量。

#### 4.1.1.1.1 文件支持的统一内存（File-backed Unified Memory）

由于具有完整 CUDA 统一内存支持的系统允许设备访问 host 进程拥有的任意内存，因此它们可以直接访问文件支持的内存。

下面把前一小节的初始示例改为使用文件支持的内存：GPU 直接从输入文件读取字符串并打印。示例中的内存由物理文件支持，但同样适用于由内存支持的文件。

```cpp
__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}

void test_file_backed() {
  int fd = open(INPUT_FILE_NAME, O_RDONLY);
  ASSERT(fd >= 0, "Invalid file handle");
  struct stat file_stat;
  int status = fstat(fd, &file_stat);
  ASSERT(status >= 0, "Invalid file stats");
  char* mapped = (char*)mmap(0, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ASSERT(mapped != MAP_FAILED, "Cannot map file into memory");
  kernel<<<1, 1>>>("file-backed", mapped);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  ASSERT(munmap(mapped, file_stat.st_size) == 0, "Cannot unmap file");
  ASSERT(close(fd) == 0, "Cannot close file");
}
```

请注意，在不具有 `hostNativeAtomicSupported` 属性的系统上（包括启用了 Linux HMM 的系统），不支持对文件支持的内存执行原子访问；该属性见[Host 原生原子操作](#4-1-1-2-3-host-原生原子操作)。

#### 4.1.1.1.2 使用统一内存进行进程间通信（Inter-Process Communication (IPC) with Unified Memory）

> **注意**
>
> 截至目前，使用统一内存进行 IPC 可能会带来显著的性能影响。

许多应用倾向于每个进程管理一块 GPU，但仍需要使用统一内存（例如为了支持超额订阅），并从多块 GPU 访问它。

CUDA IPC 不支持托管内存：这类内存的句柄不能通过本节讨论的任何机制共享。在具有完整 CUDA 统一内存支持的系统上，系统分配的内存具备 IPC 能力。一旦系统分配的内存访问权限已共享给其他进程，就可以使用与[文件支持的统一内存](#4-1-1-1-1-文件支持的统一内存)类似的编程模型。

如需了解 Linux 下创建可用于 IPC 的系统分配内存的不同方式，请参阅：

- [`mmap` 与 `MAP_SHARED`](https://man7.org/linux/man-pages/man2/mmap.2.html)；
- [POSIX IPC API](https://pubs.opengroup.org/onlinepubs/007904875/functions/shm_open.html)；
- [Linux `memfd_create`](https://man7.org/linux/man-pages/man2/memfd_create.2.html)。

请注意，无法使用这种技术在不同 host 及其设备之间共享内存。

### 4.1.1.2 性能调优（Performance Tuning）

为了让统一内存获得良好性能，需要：

- 理解系统上的分页工作方式，并避免不必要的页错误；
- 理解让数据保持在访问处理器本地的各种机制；
- 根据系统的内存传输粒度调优应用。

一般而言，性能提示（见[性能提示](#4-1-4-性能提示)）可能改善性能，但错误使用也可能比默认行为更慢。还要注意，每个提示都会在 host 上产生性能成本；有用的提示至少必须带来足够的性能收益，以抵消这项成本。

#### 4.1.1.2.1 内存分页与页面大小（Memory Paging and Page Sizes）

要理解统一内存的性能影响，必须理解虚拟寻址、内存页面和页面大小。本小节定义必要术语，并解释分页为什么会影响性能。

目前所有支持统一内存的系统都使用虚拟地址空间：应用使用的内存地址表示的是一个*虚拟*位置，该位置可能被*映射*到内存实际驻留的物理位置。

所有当前支持的处理器（包括 CPU 和 GPU）还使用内存*分页*。由于所有系统都使用虚拟地址空间，内存页面有两种：

- 虚拟页面：表示每个进程中由操作系统跟踪的、固定大小且连续的虚拟内存块，可以被*映射*到物理内存。虚拟页面与映射关联；例如，同一虚拟地址可以使用不同页面大小映射到物理内存。
- 物理页面：表示处理器的主内存管理单元（Memory Management Unit，MMU）支持的、固定大小且连续的内存块，虚拟页面可以映射到其中。

目前所有 x86_64 CPU 使用 4 KiB 的默认物理页面大小。Arm CPU 根据具体 CPU 支持 4 KiB、16 KiB、32 KiB 和 64 KiB 等多种物理页面大小。NVIDIA GPU 也支持多种物理页面大小，但优先使用 2 MiB 或更大的物理页面。这些大小将来可能随硬件变化。

虚拟页面的默认大小通常对应物理页面大小，但应用可以使用其他大小，只要操作系统和硬件支持。通常，支持的虚拟页面大小必须是 2 的幂，并且是物理页面大小的倍数。

跟踪虚拟页面到物理页面映射的逻辑实体称为*页表*；给定虚拟页面以给定虚拟大小映射到物理页面的每个映射称为*页表项（Page Table Entry，PTE）*。所有支持的处理器都有专用的页表缓存，以加快虚拟地址到物理地址的转换，这些缓存称为*转换后备缓冲区（Translation Lookaside Buffer，TLB）*。

应用性能调优有两个重要方面：

- 虚拟页面大小的选择；
- 系统是否提供由 CPU 和 GPU 共用的合并页表，还是为每个 CPU 和 GPU 分别提供独立页表。

##### 4.1.1.2.1.1 选择合适的页面大小（Choosing the Right Page Size）

通常，小页面会减少（虚拟）内存碎片，但带来更多 TLB 未命中；大页面会增加内存碎片，但减少 TLB 未命中。此外，由于通常迁移完整内存页面，较大页面的内存迁移成本通常高于较小页面，可能在使用大页面的应用中造成更大的延迟尖峰。页错误的更多细节见下一小节。

性能调优的一个重要方面是：相比 CPU，GPU 上的 TLB 未命中通常昂贵得多。如果 GPU 线程频繁随机访问使用足够小页面映射的统一内存，速度可能显著低于访问使用足够大页面映射的统一内存。CPU 线程随机访问使用小页面映射的大范围内存时也可能出现类似现象，但减速通常不明显，因此应用可以在较少内存碎片和这项减速之间权衡。

通常不应针对某个处理器的物理页面大小调优应用，因为物理页面大小取决于硬件，未来可能变化。上述建议只适用于虚拟页面大小。

##### 4.1.1.2.1.2 CPU 与 GPU 页表：硬件一致性与软件一致性（CPU and GPU Page Tables: Hardware Coherency vs. Software Coherency）

NVIDIA Grace Hopper 等硬件一致性系统为 CPU 和 GPU 提供逻辑上合并的页表。这很重要，因为 GPU 访问系统分配的内存时，会使用 CPU 为请求内存创建的页表项。如果该页表项使用 CPU 默认的 4 KiB 或 64 KiB 页面大小，访问大范围虚拟内存会造成大量 TLB 未命中，从而显著减速。

相反，在 CPU 和 GPU 各自拥有逻辑页表的软件一致性系统中，需要考虑不同的调优因素。为保证一致性，这类系统通常在某个处理器访问映射到另一处理器物理内存的地址时使用*页错误*。此页错误意味着：

- 必须确保当前拥有该页面的处理器（物理页面当前驻留的位置）不再访问它，例如删除或更新页表项；
- 必须确保请求访问的处理器能够访问该页面，例如创建新页表项或更新已有项，使其有效/激活；
- 必须把支持该虚拟页面的物理页面移动/迁移到请求访问的处理器；这可能很昂贵，工作量与页面大小成正比。

在 CPU 和 GPU 线程频繁并发访问同一内存页面时，硬件一致性系统相比软件一致性系统具有显著性能优势：

- 更少页错误：无需通过页错误模拟一致性或迁移内存；
- 更少竞争：一致性的粒度是缓存行而不是页面；多个处理器在同一缓存行上竞争时，只交换缓存行（它远小于最小页面），而不同处理器访问页面内不同缓存行时不会产生竞争。

这会影响以下场景的性能：

- CPU 和 GPU 同时对同一地址进行原子更新；
- CPU 线程向 GPU 线程发信号，或反过来。

##### 4.1.1.2.1.3 混合硬件一致性与软件一致性（Mixing Hardware and Software Coherency）

某些具有硬件一致性的系统（例如 NVIDIA DGX Station）也支持安装离散的、非一致性 GPU 硬件。硬件一致性 GPU 的访问仍使用[CPU 与 GPU 页表：硬件一致性与软件一致性](#4-1-1-2-1-2-cpu-与-gpu-页表硬件一致性与软件一致性)所述的硬件一致性；离散 GPU 的访问则使用软件一致性。

两类 GPU 可以共享统一地址空间，但访问不同 GPU 会产生不同的性能和迁移行为。软件一致性 GPU 的访问会有更多页错误和内存迁移；硬件一致性 GPU 的访问页错误更少，并在可能时进行远程映射。

为了获得最佳性能，应限制两类 GPU 之间的数据共享，或使用显式复制。也可以调用 `cudaMemAdviseSetPreferredLocation`，确保频繁共享的数据物理驻留在 CPU 或一致性 GPU 内存中，因为默认情况下访问软件一致性内存需要触发错误并迁移数据。

在混合一致性系统上，软件一致性 GPU 的 `cudaHostRegister` 和其他 host 内存访问 API 的行为也会改变。软件一致性 GPU 不使用固定映射，而是使用 CPU 页表的软件镜像。因此某些 GPU 访问可能触发页错误，而在非混合一致性系统上不会；不过这类错误很少见，通常只在内存压力下发生。

#### 4.1.1.2.2 从 host 直接访问统一内存（Direct Unified Memory Access from the Host）

某些设备支持 host 对驻留在 GPU 上的统一内存进行一致性读取、存储和原子访问，这些设备的 `cudaDevAttrDirectManagedMemAccessFromHost` 属性为 1。所有硬件一致性系统中与 NVLink 相连的设备都设置了该属性。在这些系统上，host 可以直接访问 GPU 驻留内存，不会发生页错误和数据迁移。对于 CUDA 托管内存，必须使用位置类型为 `cudaMemLocationTypeHost` 的 `cudaMemAdviseSetAccessedBy` 提示，才能启用无页错误的直接访问。

下面两个示例使用同一组 kernel，分别展示系统分配内存和托管内存：

```cpp
__global__ void write(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

__global__ void append(int *ret, int a, int b) {
  ret[threadIdx.x] += a + b + threadIdx.x;
}

void test_malloc() {
  int *ret = (int*)malloc(1000 * sizeof(int));
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);
  write<<< 1, 1000 >>>(ret, 10, 100);
  cudaDeviceSynchronize();
  for(int i = 0; i < 1000; i++)
      printf("%d: A+B = %d\n", i, ret[i]);
  append<<< 1, 1000 >>>(ret, 10, 100);
  cudaDeviceSynchronize();
  free(ret);
}
```

```cpp
void test_managed() {
  int *ret;
  cudaMallocManaged(&ret, 1000 * sizeof(int));
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);
  write<<< 1, 1000 >>>(ret, 10, 100);
  cudaDeviceSynchronize();
  for(int i = 0; i < 1000; i++)
      printf("%d: A+B = %d\n", i, ret[i]);
  append<<< 1, 1000 >>>(ret, 10, 100);
  cudaDeviceSynchronize();
  cudaFree(ret);
}
```

`write` kernel 完成后，`ret` 会在 GPU 内存中创建并初始化。随后 CPU 访问 `ret`，再由 `append` kernel 使用同一个 `ret` 内存。行为取决于系统架构和硬件一致性支持：

- `directManagedMemAccessFromHost=1`：CPU 访问托管缓冲区不会触发迁移；数据保持在 GPU 内存中，后续 GPU kernel 可以直接访问，不会产生错误或迁移；
- `directManagedMemAccessFromHost=0`：CPU 访问会触发页错误并启动数据迁移；首次访问同一数据的 GPU kernel 也会触发页错误，把页面迁回 GPU。

#### 4.1.1.2.3 Host 原生原子操作（Host Native Atomics）

某些设备（包括硬件一致性系统中通过 NVLink 连接的设备）支持对 CPU 驻留内存进行硬件加速的原子访问。这意味着对 host 内存的原子访问不必通过页错误模拟；这些设备的 `cudaDevAttrHostNativeAtomicSupported` 属性为 1。

#### 4.1.1.2.4 原子访问与同步原语（Atomic Accesses and Synchronization Primitives）

CUDA 统一内存支持 host 和 device 线程可用的全部原子操作，使所有线程能够并发访问同一共享内存位置并协作。`libcu++` 库提供了许多为 host/device 线程并发使用而调优的异构同步原语，包括 `cuda::atomic`、`cuda::atomic_ref`、`cuda::barrier` 和 `cuda::semaphore` 等。

在软件一致性系统上，device 对文件支持 host 内存进行原子访问可能触发页错误。下面的代码在硬件一致性系统上有效，在其他系统上则产生未定义行为：

```cpp
#include <cuda/atomic>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>

__global__ void kernel(int* ptr) {
  cuda::atomic_ref{*ptr}.store(2);
}

int main() {
  FILE* tmp_file = tmpfile64();
  int status = posix_fallocate(fileno(tmp_file), 0, 4096);
  int* ptr = (int*)mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileno(tmp_file), 0);
  *ptr = 1;
  printf("Atom value: %d\n", *ptr);
  kernel<<<1, 1>>>(ptr);
  while (cuda::atomic_ref{*ptr}.load() != 2);
  printf("Atom value: %d\n", *ptr);
  return EXIT_SUCCESS;
}
```

在软件一致性系统上，对统一内存的原子访问可能触发页错误，从而产生显著延迟。但这并不适用于这些系统上 GPU 对 CPU 内存的所有原子操作：`nvidia-smi -q | grep "Atomic Caps Outbound"` 列出的操作可能避免页错误。

在硬件一致性系统上，host 与 device 之间的原子操作不需要页错误，但由于其他原因，某些内存访问仍可能触发错误。

#### 4.1.1.2.5 `Memcpy()`/`Memset()` 与统一内存的行为（Memcpy()/Memset() Behavior With Unified Memory）

`cudaMemcpy*()` 和 `cudaMemset*()` 接受任何统一内存指针作为参数。

对 `cudaMemcpy*()` 而言，`cudaMemcpyKind` 指定的方向是性能提示；当任一参数是统一内存指针时，该提示可能产生更大的性能影响。建议遵循以下原则：

- 已知统一内存物理位置时，使用准确的 `cudaMemcpyKind` 提示；
- 相比不准确的 `cudaMemcpyKind`，优先使用 `cudaMemcpyDefault`；
- 始终使用已填充（已初始化）的缓冲区，避免用这些 API 初始化内存；
- 如果两个指针都指向系统分配内存，避免使用 `cudaMemcpy*()`；应启动 kernel，或使用 `std::memcpy` 等 CPU 内存复制算法。

#### 4.1.1.2.6 统一内存的内存分配器概览（Overview of Memory Allocators for Unified Memory）

具有完整 CUDA 统一内存支持的系统可以使用多种分配器分配统一内存。下表概览部分分配器及其特性；本节所有信息都可能在未来 CUDA 版本中变化。

| API | 放置策略 | 可访问方 | 是否依据访问迁移 | 页面大小 |
| --- | --- | --- | --- | --- |
| `malloc`、`new`、`mmap` | 首次触碰/提示 | CPU、GPU | 是 | 系统页面或大页面 |
| `cudaMallocManaged` | 首次触碰/提示 | CPU、GPU | 是 | CPU 驻留：系统页面；GPU 驻留：2 MB |
| `cudaMalloc` | GPU | GPU | 否 | GPU 页面：2 MB |
| `cudaMallocHost`、`cudaHostAlloc`、`cudaHostRegister` | CPU | CPU、GPU | 否 | CPU 映射：系统页面；GPU 映射：2 MB |
| 内存池（位置类型 host）：`cuMemCreate`、`cudaMemPoolCreate` | CPU | CPU、GPU | 否 | CPU 映射：系统页面；GPU 映射：2 MB |
| 内存池（位置类型 device）：`cuMemCreate`、`cudaMemPoolCreate`、`cudaMallocAsync` | GPU | GPU | 否 | 2 MB |

表中“是否依据访问迁移”是指是否会因为访问而迁移；“页面大小”可能受系统配置和未来 CUDA 版本影响。

对于 `mmap`，文件支持内存默认放置在 CPU 上，除非通过 `cudaMemAdviseSetPreferredLocation`（或 `mbind`）另行指定。

即使使用 `cudaMemAdvise` 禁用了基于访问的迁移，当后备内存空间已满时，内存仍可能迁移。文件支持内存不会依据访问迁移。

除非显式指定大页面（例如 `mmap MAP_HUGETLB` / `MAP_HUGE_SHIFT`），大多数系统的默认系统页面大小为 4 KiB 或 64 KiB；此时支持系统配置的任意大页面大小。GPU 驻留内存的页面大小可能在未来 CUDA 版本中变化；当前在将内存迁移到 GPU 或通过首次触碰放置到 GPU 时，可能无法保留大页面大小。

上表展示了多个分配器的语义差异。这些分配器都可能用于分配可同时被多个处理器（包括 host 和 device）访问的数据。`cudaMemPoolCreate` 的更多细节见[内存池](stream-ordered-memory-allocator.html#stream-ordered-memory-pools)，`cuMemCreate` 的更多细节见[虚拟内存管理](virtual-memory-management.html#virtual-memory-management)。

在 device 内存作为 NUMA 域暴露给系统的硬件一致性系统上，可以使用 `numa_alloc_on_node` 等专用分配器，把内存固定到给定 NUMA 节点（host 或 device）。这类内存可由 host 和 device 访问，且不会迁移。同样，`mbind` 可以把内存固定到给定 NUMA 节点，也可以让文件支持内存在首次访问前放置到给定 NUMA 节点。

对于可共享内存的分配器，还适用以下规则：

- `mmap` 等系统分配器允许通过 `MAP_SHARED` 标志在进程间共享内存。CUDA 支持这种方式，可用于在同一 host 连接的不同设备之间共享内存；但目前不支持在多个 host 之间共享，也不支持同时在多个 device 之间共享。详情见[使用统一内存进行进程间通信](#4-1-1-1-2-使用统一内存进行进程间通信)。
- 如需在多个 host 上通过网络访问统一内存或其他 CUDA 内存，请查阅所用通信库的文档，例如 [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)、[NVSHMEM](https://docs.nvidia.com/nvshmem/api/index.html)、[OpenMPI](https://www.open-mpi.org/faq/?category=runcuda) 和 [UCX](https://docs.mellanox.com/category/hpcx)。

#### 4.1.1.2.7 访问计数器迁移（Access Counter Migration）

在硬件一致性系统上，访问计数器功能跟踪 GPU 访问位于其他处理器上的内存的频率，以确保内存页面移动到最频繁访问它的处理器的物理内存中。它可以引导 CPU 与 GPU 之间以及 peer GPU 之间的迁移，这一过程称为访问计数器迁移。

从 CUDA 12.4 开始，访问计数器支持系统分配内存。文件支持内存不会根据访问迁移。对于系统分配内存，可以使用带有相应设备 ID 的 `cudaMemAdviseSetAccessedBy` 提示启用访问计数器迁移。如果访问计数器已启用，可以使用 `cudaMemAdviseSetPreferredLocation` 将位置设为 host，从而阻止迁移。默认情况下，`cudaMallocManaged` 使用“页错误并迁移”机制迁移内存。

驱动还可能使用访问计数器，更高效地缓解抖动或处理内存超额订阅场景。当前系统在设置 accessed-by 设备提示时允许托管内存使用访问计数器迁移，但这是实现细节，不应依赖它保证未来兼容性。

#### 4.1.1.2.8 避免 CPU 频繁写入 GPU 驻留内存（Avoid Frequent Writes to GPU-Resident Memory from the CPU）

host 访问统一内存时，缓存未命中可能导致 host 与 device 之间的流量超出预期。许多 CPU 架构要求所有内存操作（包括写操作）经过缓存层次结构。如果系统内存驻留在 GPU 上，CPU 频繁写入会导致缓存未命中，从 GPU 把数据先传回 CPU，然后才能写入请求的内存范围。在软件一致性系统上，这可能引入额外页错误；在硬件一致性系统上，可能增加 CPU 操作之间的延迟。

因此，如果要让 device 读取 host 产生的数据，应考虑写入 CPU 驻留内存，再由 device 直接读取：

```cpp
size_t data_size = sizeof(int);
int* data = (int*)malloc(data_size);
cudaMemLocation location = {.type = cudaMemLocationTypeHost};
cudaMemAdvise(data, data_size, cudaMemAdviseSetPreferredLocation, location);
cudaMemAdvise(data, data_size, cudaMemAdviseSetAccessedBy, location);
for (int i = 0; i < 10; ++i) {
  *data = 42 + i;
  kernel<<<1, 1>>>(data);
  cudaDeviceSynchronize();
}
free(data);
```

使用 `cudaMallocManaged` 时只需把分配与释放替换为：

```cpp
int* data;
size_t data_size = sizeof(int);
cudaMallocManaged(&data, data_size);
cudaMemLocation location = {.type = cudaMemLocationTypeHost};
cudaMemAdvise(data, data_size, cudaMemAdviseSetPreferredLocation, location);
cudaMemAdvise(data, data_size, cudaMemAdviseSetAccessedBy, location);
for (int i = 0; i < 10; ++i) {
  *data = 42 + i;
  kernel<<<1, 1>>>(data);
  cudaDeviceSynchronize();
}
cudaFree(data);
```

#### 4.1.1.2.9 利用对系统内存的异步访问（Exploiting Asynchronous Access to System Memory）

如果应用需要与 host 共享 device 工作的结果，有几种选择：

1. device 把结果写入 GPU 驻留内存，通过 `cudaMemcpy*` 传输结果，host 读取传输后的数据。
2. device 直接把结果写入 CPU 驻留内存，host 读取该数据。
3. device 把结果写入 GPU 驻留内存，host 直接访问该数据。

如果 device 可以在 host 传输/访问结果的同时调度独立工作，优先选择选项 1 或 3。如果 device 必须等到 host 访问结果后才能继续，选项 2 可能更合适；这是因为除非使用很多 host 线程读取数据，否则 device 通常能以高于 host 读取的带宽写入数据。

```cpp
void exchange_explicit_copy(cudaStream_t stream) {
  int* data, *host_data;
  size_t n_bytes = sizeof(int) * 16;
  host_data = (int*)malloc(n_bytes);
  cudaMallocManaged(&data, n_bytes);
  kernel<<<1, 16, 0, stream>>>(data);
  cudaMemcpyAsync(host_data, data, n_bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  printf("Got values %d - %d from GPU\n", host_data[0], host_data[15]);
  cudaFree(data);
  free(host_data);
}

void exchange_device_direct_write(cudaStream_t stream) {
  int* data;
  size_t n_bytes = sizeof(int) * 16;
  cudaMallocManaged(&data, n_bytes);
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetPreferredLocation, location);
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetAccessedBy, location);
  kernel<<<1, 16, 0, stream>>>(data);
  cudaStreamSynchronize(stream);
  printf("Got values %d - %d from GPU\n", data[0], data[15]);
  cudaFree(data);
}

void exchange_host_direct_read(cudaStream_t stream) {
  int* data;
  size_t n_bytes = sizeof(int) * 16;
  cudaMallocManaged(&data, n_bytes);
  cudaMemLocation device_loc = {};
  cudaGetDevice(&device_loc.id);
  device_loc.type = cudaMemLocationTypeDevice;
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetPreferredLocation, device_loc);
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetAccessedBy, device_loc);
  kernel<<<1, 16, 0, stream>>>(data);
  cudaStreamSynchronize(stream);
  printf("Got values %d - %d from GPU\n", data[0], data[15]);
  cudaFree(data);
}
```

在上面的显式复制示例中，也可以使用 host 或 device kernel 显式执行传输。对于连续数据，优先使用 CUDA copy engine，因为 copy engine 的操作可以与 host 和 device 上的工作重叠。`cudaMemcpy*` 和 `cudaMemPrefetchAsync` 可能使用 copy engine，但不能保证 `cudaMemcpy*` 一定使用它。出于同样原因，对于足够大的数据，显式复制优于 host 直接读取：如果 host 和 device 的工作都没有使各自的内存系统饱和，copy engine 可以同时执行传输和两者的工作。

copy engine 通常既用于 host/device 传输，也用于 NVLink 系统内 peer device 之间的传输。由于 copy engine 总数有限，某些系统使用 `cudaMemcpy*` 的带宽可能低于使用 device 显式执行传输。如果传输位于应用关键路径上，可以优先使用显式 device 传输。

## 4.1.2 仅支持 CUDA 托管内存的设备上的统一内存（Unified Memory on Devices with only CUDA Managed Memory Support）

对于计算能力为 6.x 或更高但不支持 pageable memory access 的设备，CUDA 托管内存得到完整支持且具有一致性，但 GPU 不能访问系统分配的内存。其编程模型和性能调优大体与[具有完整 CUDA 统一内存支持的设备上的统一内存](#4-1-1-具有完整-cuda-统一内存支持的设备上的统一内存)相同，区别是不能使用系统分配器分配内存。因此以下小节不适用：

- [统一内存：深入示例](#4-1-1-1-统一内存深入示例)；
- [CPU 与 GPU 页表：硬件一致性与软件一致性](#4-1-1-2-1-2-cpu-与-gpu-页表硬件一致性与软件一致性)；
- [原子访问与同步原语](#4-1-1-2-4-原子访问与同步原语)；
- [访问计数器迁移](#4-1-1-2-7-访问计数器迁移)；
- [避免 CPU 频繁写入 GPU 驻留内存](#4-1-1-2-8-避免-cpu-频繁写入-gpu-驻留内存)；
- [利用对系统内存的异步访问](#4-1-1-2-9-利用对系统内存的异步访问)。

## 4.1.3 Windows、WSL 和 Tegra 上的统一内存（Unified Memory on Windows, WSL, and Tegra）

> **注意**
>
> 本节只讨论计算能力低于 6.0 的设备、Windows 平台，以及 `concurrentManagedAccess` 属性为 0 的设备。

这些设备支持 CUDA 托管内存，但有以下限制：

- **数据迁移与一致性**：不支持按需将托管数据以细粒度移动到 GPU。每次启动 GPU kernel 时，通常都必须把所有托管内存传输到 GPU 内存，以避免内存访问错误；只有 CPU 端支持页错误。
- **GPU 内存超额订阅**：不能分配多于 GPU 物理内存大小的托管内存。
- **一致性与并发**：不支持同时访问托管内存。GPU kernel 运行时，如果 CPU 访问统一内存分配，一致性无法保证，因为缺少 GPU 页错误机制。

### 4.1.3.1 多 GPU（Multi-GPU）

在计算能力低于 6.0 的设备或 Windows 平台上，托管分配会通过 GPU 的 peer-to-peer 能力自动对系统内所有 GPU 可见。其行为类似使用 `cudaMalloc()` 分配的非托管内存：当前 active device 是物理分配的归属设备，但系统中其他 GPU 会通过 PCIe 总线以较低带宽访问该内存。

在 Linux 上，只要程序实际使用的所有 GPU 都支持 peer-to-peer，托管内存就会分配在 GPU 内存中。如果应用开始使用与已有托管分配所在其他 GPU 不支持 peer-to-peer 的 GPU，驱动会把所有托管分配迁移到系统内存；此时所有 GPU 都受 PCIe 带宽限制。

在 Windows 上，如果没有 peer 映射（例如不同架构的 GPU 之间），系统会自动退回使用映射内存，无论程序是否实际使用两块 GPU。如果实际上只使用一块 GPU，必须在启动程序前设置 `CUDA_VISIBLE_DEVICES`，限制可见 GPU，使托管内存能够分配在 GPU 内存中。

另一种方式是在 Windows 上将 `CUDA_MANAGED_FORCE_DEVICE_ALLOC` 设为非零值，强制驱动始终使用 device memory 作为物理存储。设置后，进程使用的所有支持托管内存的设备都必须彼此兼容 peer-to-peer。如果使用了一个支持托管内存但与进程此前使用的其他托管内存设备都不兼容 peer-to-peer 的设备，即使已经对这些设备调用 `::cudaDeviceReset`，也会返回 `::cudaErrorInvalidDevice`。这些环境变量见[CUDA 环境变量](../05-technical-appendices/environment-variables.html#cuda-environment-variables)。

### 4.1.3.2 一致性与并发（Coherency and Concurrency）

为了保证一致性，统一内存编程模型限制 CPU 与 GPU 并发执行时的数据访问。实际上，任何 kernel 操作执行期间，GPU 对所有托管数据拥有独占访问权，CPU 不得访问这些数据，即使特定 kernel 根本没有使用这些数据。即便访问的是不同的托管内存分配，并发 CPU/GPU 访问也会导致段错误，因为该页面被认为对 CPU 不可访问。

例如，下面的代码利用 GPU 页错误能力，在计算能力 6.x 设备上可以成功运行；但在 6.x 以前的架构和 Windows 平台上会失败，因为 CPU 访问 `y` 时 GPU kernel 仍在运行：

```cpp
__device__ __managed__ int x, y=2;
__global__ void kernel() { x = 10; }
int main() {
  kernel<<<1, 1>>>();
  y = 20;            // 不支持并发访问的 GPU 上会出错
  cudaDeviceSynchronize();
  return 0;
}
```

必须先与 GPU 显式同步，再访问 `y`（无论 GPU kernel 是否实际触碰 `y` 或任何托管数据）：

```cpp
__device__ __managed__ int x, y=2;
__global__ void kernel() { x = 10; }
int main() {
  kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
  y = 20;            // 不支持并发访问的 GPU 上成功
  return 0;
}
```

任何逻辑上保证 GPU 完成工作的函数调用都可以用于保证 GPU 工作已完成，例如[显式同步](../03-advanced-cuda/advanced-apis-and-features.html#_3-1-3-2-显式同步-explicit-synchronization)。如果 GPU 活跃期间通过 `cudaMallocManaged()` 或 `cuMemAllocManaged()` 动态分配内存，那么在启动更多工作或同步 GPU 前，该内存的行为未指定；此时 CPU 访问可能导致也可能不导致段错误。使用 `cudaMemAttachHost` 或 `CU_MEM_ATTACH_HOST` 标志分配的内存不受此规则影响。

### 4.1.3.3 与 stream 关联的统一内存（Stream Associated Unified Memory）

CUDA 编程模型提供 stream，让程序表达 kernel 启动之间的依赖和独立性。同一 stream 中启动的 kernel 保证连续执行，不同 stream 中的 kernel 可以并发执行，见[CUDA Streams](../02-programming-gpus/asynchronous-execution.html#cuda-streams)。

#### 4.1.3.3.1 Stream 回调（Stream Callbacks）

如果 GPU 上没有其他可能访问托管数据的活跃 stream，CPU 可以在 stream 回调中访问托管数据。此外，如果回调后没有 device 工作，也可以使用回调进行同步，例如在回调中通知条件变量；否则，CPU 访问只在回调执行期间有效。需要注意：

1. GPU 活跃时，CPU 始终可以访问非托管的、可映射内存数据。
2. 只要 GPU 正在运行任意 kernel，就视为 GPU 活跃，即使该 kernel 不使用托管数据。如果 kernel 可能使用数据，则禁止访问。
3. GPU 之间并发访问托管内存没有额外限制，只有非托管内存多 GPU 访问的限制适用。
4. 并发 GPU kernel 访问托管数据没有限制。

最后一点意味着 GPU kernel 之间可能发生竞态，这与当前非托管 GPU 内存的情况相同。从 GPU 角度看，托管内存函数与非托管内存完全相同。示例：

```cpp
int main() {
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  int *non_managed, *managed, *also_managed;
  cudaMallocHost(&non_managed, 4);
  cudaMallocManaged(&managed, 4);
  cudaMallocManaged(&also_managed, 4);
  kernel<<<1, 1, 0, stream1>>>(managed);
  *non_managed = 1;               // CPU 可以访问非托管数据
  *also_managed = 2;              // GPU 忙时会产生段错误
  kernel<<<1, 1, 0, stream2>>>(managed);
  cudaSetDevice(1);
  kernel<<<1, 1>>>(managed);      // 多 GPU 并发访问也允许
  return 0;
}
```

#### 4.1.3.3.2 与 stream 关联的托管内存提供更细粒度的控制（Managed memory associated to streams allows for finer-grained control）

统一内存在 stream 独立模型上进一步允许 CUDA 程序显式将托管分配与某个 CUDA stream 关联。这样，程序员可以根据 kernel 是否在指定 stream 中启动来表达其数据使用方式，并根据应用特有的数据访问模式获得并发机会。控制该行为的函数是：

```cpp
cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream,
                                     void *ptr,
                                     size_t length=0,
                                     unsigned int flags=0);
```

`cudaStreamAttachMemAsync()` 将从 `ptr` 开始的 `length` 字节内存与指定 stream 关联。只要该 stream 中的所有操作已完成，CPU 就可以访问这段内存，而不论其他 stream 是否活跃。也就是说，活跃 GPU 对托管内存区域的独占所有权从整个 GPU 细化到了单个 stream。最重要的是，如果分配没有与特定 stream 关联，它对所有运行中的 kernel（不论所在 stream）可见；这正是 `cudaMallocManaged()` 分配或 `__managed__` 变量的默认可见性，因此才有“任意 kernel 运行期间 CPU 不得触碰数据”的简单规则。

> **注意**
>
> 将分配与特定 stream 关联，就意味着程序保证只有在该 stream 中启动的 kernel 会触碰这段数据。统一内存系统不会进行错误检查。

> **注意**
>
> 除了允许更高并发，使用 `cudaStreamAttachMemAsync()` 还可以启用统一内存系统内部的数据传输优化，从而影响延迟和其他开销。

下面的示例将 `y` 显式关联为 host 可访问，从而让 CPU 始终可以访问它（注意 kernel 调用后没有 `cudaDeviceSynchronize()`）。此后，运行 kernel 的 GPU 对 `y` 的访问会产生未定义结果：

```cpp
__device__ __managed__ int x, y=2;
__global__ void kernel() { x = 10; }
int main() {
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cudaStreamAttachMemAsync(stream1, &y, 0, cudaMemAttachHost);
  cudaDeviceSynchronize();
  kernel<<<1, 1, 0, stream1>>>();
  y = 20;                           // kernel 正在运行，但 y 已关联到 host
  return 0;
}
```

#### 4.1.3.3.3 多线程 host 程序的更完整示例（A more elaborate example on multithreaded host programs）

`cudaStreamAttachMemAsync()` 的主要用途是启用 CPU 线程之间的独立任务并行。通常，每个 CPU 线程为它生成的所有工作创建自己的 stream，因为使用 CUDA NULL stream 会在线程之间产生依赖。托管数据默认对所有 GPU stream 可见，可能使多线程程序难以避免线程之间的相互影响。因此使用 `cudaStreamAttachMemAsync()` 将线程的托管分配与该线程自己的 stream 关联，并且该关联通常在线程生命周期内不变：

```cpp
void run_task(int *in, int *out, int length) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  int *data;
  cudaMallocManaged((void **)&data, length, cudaMemAttachHost);
  cudaStreamAttachMemAsync(stream, data);
  cudaStreamSynchronize(stream);
  for(int i=0; i<N; i++) {
    transform<<<100, 256, 0, stream>>>(in, data, length);
    cudaStreamSynchronize(stream);
    host_process(data, length);
    convert<<<100, 256, 0, stream>>>(out, data, length);
  }
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  cudaFree(data);
}
```

本例只建立一次分配与 stream 的关联，之后 host 和 device 重复使用数据。与显式在 host 和 device 之间复制数据相比，代码更简单，但结果相同。

`cudaMallocManaged()` 指定 `cudaMemAttachHost` 标志后，分配初始时对 device 端执行不可见（默认分配则对所有 stream 的 GPU kernel 可见）。这保证从分配数据到将数据获取到特定 stream 之间，不会意外与其他线程的执行交互。否则，如果另一个线程启动的 kernel 正在运行，新分配会被视为 GPU 正在使用，可能影响本线程在显式附加到私有 stream 前从 CPU 访问新数据的能力。为保证线程独立，应使用该标志进行分配。

另一种办法是在分配附加到 stream 后，在所有线程之间设置进程级 barrier；在启动任何 kernel 前，保证所有线程完成数据/stream 关联。销毁 stream 前还需要第二个 barrier，因为销毁 stream 会让分配恢复默认可见性。`cudaMemAttachHost` 标志既简化了该流程，也适用于无法在需要位置插入全局 barrier 的情况。

#### 4.1.3.3.4 与 stream 关联的统一内存的数据移动（Data Movement of Stream Associated Unified Memory）

在 `concurrentManagedAccess` 未设置的设备上，`Memcpy()`/`Memset()` 与 stream 关联统一内存的行为不同：

- 指定 `cudaMemcpyHostTo*` 且源数据为统一内存时，如果源数据在复制 stream 中可由 host 一致性访问，则从 host 访问；否则从 device 访问。指定 `cudaMemcpy*ToHost` 且目标是统一内存时，目标遵循同样规则。
- 指定 `cudaMemcpyDeviceTo*` 且源数据为统一内存时，从 device 访问；源数据必须在复制 stream 中可由 device 一致性访问，否则返回错误。指定 `cudaMemcpy*ToDevice` 且目标是统一内存时同理。
- 指定 `cudaMemcpyDefault` 时，如果统一内存在复制 stream 中无法由 device 一致性访问，或其首选位置是 `cudaCpuDeviceId` 且能由 host 一致性访问，则从 host 访问；否则从 device 访问。
- 对统一内存使用 `cudaMemset*()` 时，数据必须在 `cudaMemset()` 所用 stream 中可由 device 一致性访问，否则返回错误。

当 `cudaMemcpy*` 或 `cudaMemset*` 从 device 访问数据时，操作所在 stream 被视为 GPU 活跃。如果设备属性 `concurrentManagedAccess` 为 0，此期间 CPU 访问与该 stream 关联或具有全局可见性的数据会导致段错误。程序必须适当同步，确保操作完成后再从 CPU 访问关联数据。

在给定 stream 中，内存“可由 host 一致性访问”意味着它既没有全局可见性，也没有与该 stream 关联；“可由 device 一致性访问”意味着它具有全局可见性，或与该 stream 关联。

## 4.1.4 性能提示（Performance Hints）

性能提示允许程序员向 CUDA 提供更多统一内存使用信息。CUDA 使用这些提示更高效地管理托管内存，从而改善应用性能。性能提示绝不会影响应用正确性，只会影响性能。

> **注意**
>
> 只有在能够改善性能时，应用才应使用统一内存性能提示。

性能提示可以用于任意统一内存分配，包括 CUDA 托管内存。在具有完整 CUDA 统一内存支持的系统上，性能提示可以应用于所有系统分配的内存。

### 4.1.4.1 数据预取（Data Prefetching）

`cudaMemPrefetchAsync` 是一个异步、按 stream 顺序执行的 API，可以把数据迁移到更接近指定处理器的位置。预取期间仍可以访问数据。迁移要等 stream 中之前的操作全部完成后才开始，并在 stream 中之后的操作开始前完成。

```cpp
cudaError_t cudaMemPrefetchAsync(const void *devPtr,
                                 size_t count,
                                 struct cudaMemLocation location,
                                 unsigned int flags,
                                 cudaStream_t stream=0);
```

执行给定 `stream` 中的预取任务时，包含 `[devPtr, devPtr + count)` 的内存区域可以迁移到 `location.type` 为 `cudaMemLocationTypeDevice` 时指定的 `location.id` 设备，或迁移到 `location.type` 为 `cudaMemLocationTypeHost` 时的 CPU。`flags` 的细节见当前 [CUDA Runtime API 文档](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)。

系统分配内存的预取示例：

```cpp
void test_prefetch_sam(const cudaStream_t& s) {
  char *data = (char*)malloc(dataSizeBytes);
  init_data(data, dataSizeBytes);
  cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  const unsigned int flags = 0;
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);
  const unsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
  mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);
  location = {.type = cudaMemLocationTypeHost};
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);
  cudaStreamSynchronize(s);
  use_data(data, dataSizeBytes);
  free(data);
}
```

托管内存的预取示例：

```cpp
void test_prefetch_managed(const cudaStream_t& s) {
  char *data;
  cudaMallocManaged(&data, dataSizeBytes);
  init_data(data, dataSizeBytes);
  cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  const unsigned int flags = 0;
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);
  const unsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
  mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);
  location = {.type = cudaMemLocationTypeHost};
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);
  cudaStreamSynchronize(s);
  use_data(data, dataSizeBytes);
  cudaFree(data);
}
```

### 4.1.4.2 数据使用提示（Data Usage Hints）

当多个处理器同时访问同一数据时，可以用 `cudaMemAdvise` 提示 `[devPtr, devPtr + count)` 范围内数据将如何被访问：

```cpp
cudaError_t cudaMemAdvise(const void *devPtr,
                          size_t count,
                          enum cudaMemoryAdvise advice,
                          struct cudaMemLocation location);
```

`advice` 可以取以下值：

- `cudaMemAdviseSetReadMostly`：表示数据主要被读取、偶尔写入；通常可以在该区域用读取带宽换取写入带宽。
- `cudaMemAdviseSetPreferredLocation`：将数据首选位置设为指定 device 的物理内存，鼓励系统保持数据位于该位置，但不保证。将 `location.type` 设为 `cudaMemLocationTypeHost` 会把 CPU 内存作为首选位置；`cudaMemPrefetchAsync` 等其他提示可以覆盖该提示，让内存离开首选位置。
- `cudaMemAdviseSetAccessedBy`：在某些系统上，访问前建立到内存的映射有利于性能。当 `location.type` 为 `cudaMemLocationTypeDevice` 时，该提示告知系统数据将被 `location.id` 频繁访问，使系统可以预先建立映射。它不指定数据应驻留的位置，但可以与 `cudaMemAdviseSetPreferredLocation` 组合使用。在硬件一致性系统上，它会启用访问计数器迁移。

每种提示也可以用 `cudaMemAdviseUnsetReadMostly`、`cudaMemAdviseUnsetPreferredLocation` 和 `cudaMemAdviseUnsetAccessedBy` 取消。

下面的示例使用 `cudaMemAdviseSetReadMostly`：CPU 在每个外层循环中写入数据，随后把数据预取到所有 GPU；内层循环中的 kernel 只读取数据，因此预取会复制读取数据，而不是迁移数据。

```cpp
void test_advise_sam(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * threadsPerBlock;
  dataPtr = (char*)malloc(dataSize);
  cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);
  int outerLoopIter = 0;
  while (outerLoopIter < maxOuterLoopIter) {
    init_data(dataPtr, dataSize);
    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    for (int device = 0; device < maxDevices; device++) {
      location.id = device;
      cudaMemPrefetchAsync(dataPtr, dataSize, location, 0, stream);
    }
    int innerLoopIter = 0;
    while (innerLoopIter < maxInnerLoopIter) {
      mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
      innerLoopIter++;
    }
    outerLoopIter++;
  }
  free(dataPtr);
}
```

托管内存版本将分配替换为 `cudaMallocManaged(&dataPtr, dataSize)`，在具有完整 CUDA 统一内存支持的系统上也可以使用 `malloc`；其余的 `cudaMemAdvise`、多 GPU 预取和只读 kernel 循环相同，最后使用 `cudaFree(dataPtr)`。

完整的系统分配版本如下：

```cpp
void test_advise_sam(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * threadsPerBlock;  // 16 KiB
  dataPtr = (char*)malloc(dataSize);
  cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);
  int outerLoopIter = 0;
  while (outerLoopIter < maxOuterLoopIter) {
    init_data(dataPtr, dataSize);
    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    for (int device = 0; device < maxDevices; device++) {
      location.id = device;
      const unsigned int flags = 0;
      cudaMemPrefetchAsync(dataPtr, dataSize, location, flags, stream);
    }
    int innerLoopIter = 0;
    while (innerLoopIter < maxInnerLoopIter) {
      mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
      innerLoopIter++;
    }
    outerLoopIter++;
  }
  free(dataPtr);
}
```

托管内存版本如下：

```cpp
void test_advise_managed(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * threadsPerBlock;  // 16 KiB
  cudaMallocManaged(&dataPtr, dataSize);
  cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);
  int outerLoopIter = 0;
  while (outerLoopIter < maxOuterLoopIter) {
    init_data(dataPtr, dataSize);
    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    for (int device = 0; device < maxDevices; device++) {
      location.id = device;
      const unsigned int flags = 0;
      cudaMemPrefetchAsync(dataPtr, dataSize, location, flags, stream);
    }
    int innerLoopIter = 0;
    while (innerLoopIter < maxInnerLoopIter) {
      mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
      innerLoopIter++;
    }
    outerLoopIter++;
  }
  cudaFree(dataPtr);
}
```

### 4.1.4.3 丢弃内存（Memory Discarding）

`cudaMemDiscardBatchAsync` 允许应用通知 CUDA runtime：指定内存范围的内容已不再有用。统一内存驱动会因为基于错误的迁移或内存驱逐而自动传输数据，以支持 device memory 超额订阅；这些传输有时是多余的，会严重降低性能。将地址范围标记为“丢弃”后，驱动知道应用已经消费了该范围内容，无需在预取或页面驱逐时迁移这些数据来为其他分配腾出空间。丢弃后若没有后续写入或预取就读取页面，会得到不确定值；而丢弃后的任何新写入都保证能被随后读取看到。与丢弃操作并发的访问或预取会产生未定义行为。

```cpp
cudaError_t cudaMemDiscardBatchAsync(void **dptrs,
                                     size_t *sizes,
                                     size_t count,
                                     unsigned long long flags,
                                     cudaStream_t stream);
```

该函数对 `dptrs` 和 `sizes` 数组指定的地址范围批量执行丢弃。两个数组长度必须都等于 `count`；每个范围必须引用通过 `cudaMallocManaged` 分配的托管内存，或引用 `__managed__` 变量。

`cudaMemDiscardAndPrefetchBatchAsync` 同时执行丢弃和预取。它在语义上等价于先调用 `cudaMemDiscardBatchAsync`，再调用 `cudaMemPrefetchBatchAsync`，但更高效；当应用需要内存位于目标位置而不需要保留内容时很有用：

```cpp
cudaError_t cudaMemDiscardAndPrefetchBatchAsync(void **dptrs,
                                                size_t *sizes,
                                                size_t count,
                                                struct cudaMemLocation *prefetchLocs,
                                                size_t *prefetchLocIdxs,
                                                size_t numPrefetchLocs,
                                                unsigned long long flags,
                                                cudaStream_t stream);
```

`prefetchLocs` 指定预取目标，`prefetchLocIdxs` 指示每个预取位置适用于哪些操作。例如批次有 10 个操作，前 6 个预取到一个位置、后 4 个预取到另一个位置时，`numPrefetchLocs` 为 2，`prefetchLocIdxs` 为 `{0, 6}`，`prefetchLocs` 包含两个目标位置。

**重要注意事项：**

- 没有后续写入或预取就读取丢弃范围，会得到不确定值；
- 向范围写入或通过 `cudaMemPrefetchAsync` 预取可以撤销丢弃；
- 与丢弃操作同时发生的任何读、写或预取都会产生未定义行为；
- 所有设备的 `cudaDevAttrConcurrentManagedAccess` 都必须为非零值。

### 4.1.4.4 查询托管内存上的数据使用属性（Querying Data Usage Attributes on Managed Memory）

程序可以使用以下 API 查询通过 `cudaMemAdvise` 或 `cudaMemPrefetchAsync` 分配到 CUDA 托管内存上的内存范围属性：

```cpp
cudaMemRangeGetAttribute(void *data,
                         size_t dataSize,
                         enum cudaMemRangeAttribute attribute,
                         const void *devPtr,
                         size_t count);
```

该函数查询从 `devPtr` 开始、大小为 `count` 字节的托管内存范围属性。范围必须引用 `cudaMallocManaged` 分配的内存，或 `__managed__` 变量。可查询的属性包括：

- `cudaMemRangeAttributeReadMostly`：整个范围都设置了 `cudaMemAdviseSetReadMostly` 时返回 1，否则返回 0；
- `cudaMemRangeAttributePreferredLocation`：整个范围的首选位置是某个 GPU 时返回 GPU device ID，是 CPU 时返回 `cudaCpuDeviceId`，否则返回 `cudaInvalidDeviceId`；实际查询时内存所在位置可能不同于首选位置；
- `cudaMemRangeAttributeAccessedBy`：返回该范围设置了 accessed-by 提示的设备列表；
- `cudaMemRangeAttributeLastPrefetchLocation`：返回应用最近一次通过 `cudaMemPrefetchAsync` 显式请求预取到的位置，但不表示预取是否已经开始或完成；
- `cudaMemRangeAttributePreferredLocationType`：返回首选位置类型：`cudaMemLocationTypeDevice`、`cudaMemLocationTypeHost`、`cudaMemLocationTypeHostNuma`，或因页面首选位置不一致/缺失而返回 `cudaMemLocationTypeInvalid`；
- `cudaMemRangeAttributePreferredLocationId`：当首选位置类型为 device 时返回 device ordinal，当类型为 host NUMA 时返回 NUMA 节点 ID，其他情况应忽略该 ID；
- `cudaMemRangeAttributeLastPrefetchLocationType`：返回最近一次显式预取的类型，可能为 `cudaMemLocationTypeDevice`、`cudaMemLocationTypeHost`、`cudaMemLocationTypeHostNuma` 或 `cudaMemLocationTypeInvalid`；
- `cudaMemRangeAttributeLastPrefetchLocationId`：当最近一次预取位置类型为 device 时返回有效 device ordinal，当类型为 host NUMA 时返回有效 host NUMA 节点 ID，其他情况应忽略。

还可以使用对应的 `cudaMemRangeGetAttributes` 函数一次查询多个属性。

### 4.1.4.5 GPU 内存超额订阅（GPU Memory Oversubscription）

统一内存允许应用对任意单个处理器的内存进行*超额订阅*：也就是说，可以分配并共享大于系统中任一处理器内存容量的数组。这使得应用可以在不显著增加编程模型复杂度的情况下，对无法装入单块 GPU 的数据集执行 out-of-core 处理等任务。

此外，可以使用对应的 `cudaMemRangeGetAttributes` 函数查询多个属性。
