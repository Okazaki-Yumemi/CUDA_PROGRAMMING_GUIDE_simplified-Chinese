---
title: 5.4 C/C++ 语言扩展（C/C++ Language Extensions）
description: CUDA C/C++ 的函数与变量注解、内建类型、同步原语、原子操作、Warp 函数、宏和编译器扩展。
---



<span id="c-language-extensions"></span>

# 5.4. C/C++ 语言扩展（C/C++ Language Extensions）



<span id="function-variable-annotations"></span>

## 5.4.1. 函数和变量注释（Function and Variable Annotations）



<span id="id1"></span>

### 5.4.1.1. 执行空间说明符（Execution Space Specifiers）

执行空间说明符`__host__`, `__device__`, `__tile__`, `__global__`和`__tile_global__`指示函数是在主机、SIMT还是磁贴上下文中执行。



<table id="id22" class="table-no-stripes table" style="width:100%;">
<caption>表 39 执行空间说明符<a href="#id22" class="headerlink" title="Link to this table">#</a></caption>
<colgroup>
<col style="width: 35%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
</colgroup>
<thead>
<tr class="row-odd">
<th rowspan="2" class="head">执行空间说明符</th>
<th colspan="3" class="head">执行于</th>
<th colspan="3" class="head">可调用来自</th>
</tr>
<tr class="row-even">
<th class="head">Host</th>
<th class="head">SIMT</th>
<th class="head">Tile</th>
<th class="head">Host</th>
<th class="head">SIMT</th>
<th class="head">Tile</th>
</tr>
</thead>
<tbody>
<tr class="row-odd">
<td><code class="docutils literal notranslate">__host__</code>，否说明符</td>
<td>✅</td>
<td>❌</td>
<td>❌</td>
<td>✅</td>
<td>❌</td>
<td>❌</td>
</tr>
<tr class="row-even">
<td><code class="docutils literal notranslate">__device__</code></td>
<td>❌</td>
<td>✅</td>
<td>❌</td>
<td>❌</td>
<td>✅</td>
<td>❌</td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">__global__</code></td>
<td>❌</td>
<td>✅</td>
<td>❌</td>
<td>✅</td>
<td>✅</td>
<td>❌</td>
</tr>
<tr class="row-even">
<td><code class="docutils literal notranslate">__tile__</code></td>
<td>❌</td>
<td>❌</td>
<td>✅</td>
<td>❌</td>
<td>❌</td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">__tile_global__</code></td>
<td>❌</td>
<td>❌</td>
<td>✅</td>
<td>✅</td>
<td>❌</td>
<td>❌</td>
</tr>
<tr class="row-even">
<td><code class="docutils literal notranslate">__host__</code><code class="docutils literal notranslate"> </code><code class="docutils literal notranslate">__device__</code><code class="docutils literal notranslate"> </code><code class="docutils literal notranslate">__tile__</code></td>
<td>✅</td>
<td>✅</td>
<td>✅</td>
<td>✅</td>
<td>✅</td>
<td>✅</td>
</tr>
</tbody>
</table>



------------------------------------------------------------------------

 `__global__` 和 `__tile_global__` 函数的约束：

- 必须返回 `void`.

- 不能是 `class`, `struct`, or `union`.

- 需要执行配置如 <a href="#execution-configuration" class="reference internal"> 内核配置中所述。</a>.

- 不支持递归。

- 请参阅 `__global__` <a href="cpp-language-support.html#global-function-parameters" class="reference internal"> 函数参数 </a> 了解其他限制。

 对 `__global__` 和 `__tile_global__` 函数的调用是异步的。它们在设备完成执行之前返回到主机线程。

------------------------------------------------------------------------

 使用多个执行空间声明的函数（例如，`__host__`` ``__device__`）针对每个上下文进行编译。 `__CUDA_ARCH__` <a href="#cuda-arch-macro" class="reference internal">宏</a>可用于区分主机和设备代码路径：

```cuda


    __host__ __device__ void func() {
    #if defined(__CUDA_ARCH__)
        // Device code path
    #else
        // Host code path
    #endif
    }


```





<span id="id2"></span>

### 5.4.1.2. 内存空间说明符（Memory Space Specifiers）

内存空间说明符`__device__`, `__tile__`, `__managed__`, `__constant__`和`__shared__`指示变量在设备上的存储位置。

下表总结了内存空间属性：



|内存空间说明符|地点 |可以通过 | 访问终身|唯一实例 |
|----|----|----|----|----|
| `__device__` |设备全局内存|设备线程（网格）/CUDA 运行时 API |程序/<a href="../03-advanced-cuda/driver-api.html#driver-api-context" class="reference internal">CUDA 上下文</a>|每个设备 |
| `__tile__` |设备全局内存| Tile 块/CUDA 运行时 API |程序/<a href="../03-advanced-cuda/driver-api.html#driver-api-context" class="reference internal">CUDA 上下文</a> |每个设备 |
| `__constant__` |设备常量内存|设备线程（网格）/CUDA 运行时 API |程序/<a href="../03-advanced-cuda/driver-api.html#driver-api-context" class="reference internal">CUDA 上下文</a> |每个设备 |
| `__managed__` |主机和设备（自动）|主机/设备线程 |节目|每个程序 |
| `__shared__` |设备（流式多处理器）|块线程 |块|块|
|没有说明符 |设备（寄存器）|单线程 |单线程 |单线程 |

Table 40 内存空间说明符<a href="#id23" class="headerlink" title="Link to this table">#</a> {#id23 .table-no-stripes .table}



------------------------------------------------------------------------

- `__device__`, `__tile__` 和 `__constant__` 变量可以使用 <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html" class="reference external">CUDA 运行时 API</a> 从主机访问函数 `cudaGetSymbolAddress()`, `cudaGetSymbolSize()`, `cudaMemcpyToSymbol()` 和 `cudaMemcpyFromSymbol()`.

- `__constant__` 变量在设备代码中是只读的，只能使用 <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html" class="reference external">CUDA 运行时 API 从主机修改 </a>.

 以下示例说明如何使用这些 API：

```cuda


    __device__   float device_var       = 4.0f; // Variable in device memory
    __constant__ float constant_mem_var = 4.0f; // Variable in constant memory
                                                // For readability, the following example focuses on a device variable.
    int main() {
        float* device_ptr;
        cudaGetSymbolAddress((void**) &device_ptr, device_var);        // Gets address of device_var

        size_t symbol_size;
        cudaGetSymbolSize(&symbol_size, device_var);                   // Retrieves the size of the symbol (4 bytes).

        float host_var;
        cudaMemcpyFromSymbol(&host_var, device_var, sizeof(host_var)); // Copies from device to host.

        host_var = 3.0f;
        cudaMemcpyToSymbol(device_var, &host_var, sizeof(host_var));   // Copies from host to device.
    }


```

 请参阅 <a href="https://godbolt.org/z/vYjP8GGv3" class="reference external"> 编译器资源管理器中的示例 </a>.



<span id="shared-memory-specifier"></span>

#### 5.4.1.2.1. `__shared__` Memory（`__shared__` Memory）

`__shared__` 内存变量可以具有静态大小（在编译时确定）或动态大小（在内核启动时确定）。有关在运行时指定共享内存大小的详细信息，请参阅 <a href="#execution-configuration" class="reference internal"> 内核配置 </a> 部分。

 共享内存约束：

 - 具有动态大小的变量必须声明为外部数组或指针。

 - 具有静态大小的变量不能在其内部初始化。声明。

以下示例说明如何声明 `__shared__` 变量并确定其大小：

```cuda


    extern __shared__ char dynamic_smem_pointer[];
    // extern __shared__ char* dynamic_smem_pointer; alternative syntax

    __global__ void kernel() { // or a __device__ function
        __shared__ int smem_var1[4];                  // static size
        auto smem_var2 = (int*) dynamic_smem_pointer; // dynamic size
    }

    int main() {
        size_t shared_memory_size = 16;
        kernel<<<1, 1, shared_memory_size>>>();
        cudaDeviceSynchronize();
    }


```

请参阅 <a href="https://godbolt.org/z/nPjvd1frb" class="reference external"> 编译器资源管理器中的示例</a>.





<span id="managed-memory-specifier"></span>

#### 5.4.1.2.2. `__managed__` 内存（`__managed__` Memory）

`__managed__` 变量具有以下限制：

- `__managed__` 变量的地址不是常量表达。

- A `__managed__`变量不得具有引用类型 `T&`.

 - 当 CUDA 运行时可能不处于有效状态时，不得使用 `__managed__` 变量的地址或值，包括以下情况：

 - 在具有 `static` or `thread_local` 存储持续时间的对象的静态/动态初始化或销毁中。

 - 在之后执行的代码中`exit()` 已被调用。例如，标有 `__attribute__((destructor))`.

 的函数 - 在 CUDA 运行时可能未初始化时执行的代码中。例如，标有 `__attribute__((constructor))`.

- A `__managed__` 变量的函数不能用作 `decltype()` 表达式的无括号 id 表达式参数。

- `__managed__` 变量具有与为 <a href="../02-programming-gpus/unified-and-system-memory.html#_2-6-2-统一内存-unified-memory" class="reference internal"> 动态分配的托管内存指定的相同连贯性和一致性行为</a>.

 - 另请参阅 <a href="cpp-language-support.html#local-variables" class="reference internal">local 的限制变量</a>.

以下是合法和非法使用`__managed__`变量的示例：

```cuda


    #include <cassert>

    __device__ __managed__ int global_var = 10; // OK

    int* ptr = &global_var;                     // ERROR: use of a managed variable in static initialization

    struct MyStruct1 {
        int field;
        MyStruct1() : field(global_var) {};
    };

    struct MyStruct2 {
        ~MyStruct2() { global_var = 10; }
    };

    MyStruct1 temp1; // ERROR: use of managed variable in dynamic initialization

    MyStruct2 temp2; // ERROR: use of managed variable in the destructor of
                     //        object with static storage duration

    __device__ __managed__ const int const_var = 10;         // ERROR: const-qualified type

    __device__ __managed__ int&      reference = global_var; // ERROR: reference type

    template <int* Addr>
    struct MyStruct3 {};

    MyStruct3<&global_var> temp;     // ERROR: address of managed variable is not a constant expression

    __global__ void kernel(int* ptr) {
        assert(ptr == &global_var);  // OK
        global_var = 20;             // OK
    }

    int main() {
        int* ptr = &global_var;      // OK
        kernel<<<1, 1>>>(ptr);
        cudaDeviceSynchronize();
        global_var++;                // OK
        decltype(global_var) var1;   // ERROR: managed variable used as unparenthesized argument to decltype

        decltype((global_var)) var2; // OK
    }


```





#### 5.4.1.2.3. `__tile__`变量（`__tile__` Variables）

The `__tile__`内存空间说明符与`__device__`类似。标有 `__tile__` 的变量分配在设备全局内存中，并且可以由 CUDA 运行时 API 函数访问。与 `__device__` 变量不同，`__tile__` 变量可以从图块代码中直接访问。

 一般而言，`__device__` 变量可能无法通过 `__tile__` or `__tile_global__` 代码中的名称直接访问，并且 `__tile__` 变量也不能通过 `__device__` or `__global__` 代码中的名称直接访问。但是，这两个变量都分配在设备全局内存中，并且可以在任一上下文中访问指向该内存的指针。

 以下示例显示了图块变量的合法和非法使用：

```cuda


    __device__ int x_device = 0;
    __tile__   int x_tile = 0;

    // ERROR: Variable cannot be both device and tile
    __device__ __tile__ int x_device_tile = 0;

    __global__ void simt_function(int* in) {
      x_device = 1; // OK: __device__ variable accessed from SIMT code
      x_tile = 1;   // ERROR: __tile__ variable accessed from SIMT code

      *in = 1; // OK: May point to any device global memory, including x_tile.
    }

    __tile_global__ void tile_function(int* in) {
      x_device = 1; // ERROR: __device__ variable accessed from tile code
      x_tile = 1;   // OK: __tile__ variable accessed from tile code

      *in = 1; // OK: May point to any device global memory, including x_device.
    }

    int main() {
      int* x_device_ptr;
      cudaGetSymbolAddress((void**) &x_device_ptr, x_device);    // Gets address of x_device

      int* x_tile_ptr;
      cudaGetSymbolAddress((void**) &x_tile_ptr, x_tile);        // Gets address of x_tile

      // OK: Passing pointer to tile variable into SIMT kernel
      simt_function<<<1,1>>>(x_tile_ptr);
      cudaDeviceSynchronize();

      // OK: Passing pointer to device variable into tile kernel
      tile_function<<<1,1>>>(x_device_ptr);
      cudaDeviceSynchronize();
    }


```

A `__tile__` 变量可能不包含指针或引用类型的子对象。例如：

```cuda


    __tile__ int* ptr; // ERROR

    struct S1 { int* ptr; };
    __tile__ S1 val; // ERROR


```







<span id="inline-specifiers"></span>

### 5.4.1.3. 内联说明符（Inlining Specifiers）

以下说明符可用于控制 `__host__` 和 `__device__` 函数的内联：

- `__noinline__`：指示 `nvcc` 不内联该函数。

- `__forceinline__`：强制`nvcc`在单个翻译单元内内联该函数。

- `__inline_hint__`：使用 <a href="../02-programming-gpus/nvcc.html#nvcc-link-time-optimization" class="reference internal"> 链接时优化</a>.

 时启用跨翻译单元的积极内联。这些说明符是互斥的。当应用于 `__tile__` 函数时，这些说明符将被忽略。





<span id="restrict"></span>

### 5.4.1.4. `__restrict__` 指针（`__restrict__` Pointers）

`nvcc` 通过 `__restrict__` 关键字支持受限指针。当两个或多个指针引用重叠的内存区域时，会出现

 指针别名。这会抑制诸如代码重新排序和公共子表达式消除之类的优化。

A 限制限定指针是程序员的一个承诺，即在指针的生命周期内，它指向的内存只能通过该指针访问。这允许编译器执行更积极的优化。

-访问设备函数的所有线程仅从中读取；或

 - 最多有一个线程对其进行写入，并且没有其他线程从中读取。

 以下示例说明了别名问题，并演示了如何使用受限制的指针来帮助编译器减少指令数量：

```cuda


    __device__
    void device_function(const float* a, const float* b, float* c) {
        c[0] = a[0] * b[0];
        c[1] = a[0] * b[0];
        c[2] = a[0] * b[0] * a[1];
        c[3] = a[0] * a[1];
        c[4] = a[0] * b[0];
        c[5] = b[0];
        ...
    }


```

 因为指针 `a`, `b` 和 `c` 可能存在别名，因此任何写通`c` 可以修改 `a` or `b` 的元素。为了保证功能正确性，编译器无法将 `a[0]` 和 `b[0]` 加载到寄存器中，将它们相乘，并将结果存储在 `c[0]` 和 `c[1]` 中。这是因为如果 `a[0]` 和 `c[0]` 位于同一位置，结果将与抽象执行模型不同。编译器无法利用公共子表达式。同样，编译器无法将 `c[4]` 的计算与 `c[0]` 和 `c[1]` 的计算重新排序，因为之前对 `c[3]` 的写入可能会更改 `c[4]`.

 计算的输入通过声明 `a`, `b`，并且`c`作为受限指针，程序员通知编译器这些指针没有别名。这意味着写入 `c` 永远不会覆盖 `a` or `b` 的元素。这将函数原型更改如下：

```cuda


    __device__
    void device_function(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);


```

请注意，必须限制所有指针参数才能使编译器优化器有效。通过添加 `__restrict__` 关键字，编译器可以随意重新排序和执行公共子表达式消除，同时保持与抽象执行模型相同的功能。

```cuda


    __device__
    void device_function(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
        float t0 = a[0];
        float t1 = b[0];
        float t2 = t0 * t1;
        float t3 = a[1];
        c[0]     = t2;
        c[1]     = t2;
        c[4]     = t2;
        c[2]     = t2 * t3;
        c[3]     = t0 * t3;
        c[5]     = t1;
        ...
    }


```

请参阅 <a href="https://godbolt.org/z/6KeTqarnW" class="reference external">Compiler Explorer</a>.

 上的示例。结果是减少了内存访问和计算次数，通过缓存负载和公共子表达式带来的寄存器压力增加来平衡

由于寄存器压力是许多 CUDA 代码中的一个关键问题，因此使用受限指针会减少占用，从而对性能产生负面影响。

------------------------------------------------------------------------

 访问 `__global__` 函数 `const` 标记为 `__restrict__` 的指针被编译为只读缓存加载，类似于<a href="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc" class="reference external">PTX</a> `ld.global.nc` or `__ldg()` <a href="#low-level-load-store-functions" class="reference internal">低级加载和存储函数</a>指令。

```cuda


    __global__
    void kernel1(const float* in, float* out) {
        *out = *in; // PTX: ld.global
    }

    __global__
    void kernel2(const float* __restrict__ in, float* out) {
        *out = *in;  // PTX: ld.global.nc
    }


```

请参阅 <a href="https://godbolt.org/z/drsTEPa8s" class="reference external">编译器资源管理器中的示例</a>.





<span id="grid-constant"></span>

### 5.4.1.5. `__grid_constant__`参数（`__grid_constant__` Parameters）

使用 `__global__` 注释 `__grid_constant__` 函数参数可防止编译器创建该函数的每线程副本参数。相反，网格中的所有线程都将通过单个地址访问该参数，这可以提高性能。

The `__grid_constant__` 参数具有以下属性：

 - 它具有内核的生命周期。

 -它是单个内核私有的，这意味着来自其他网格（包括子网格）的线程无法访问该对象。

 -内核中的所有线程都看到相同的对象地址.

 - 它是只读的。修改 `__grid_constant__` 对象或其任何子对象（包括 `mutable` 成员）是未定义的行为。

 要求：

- 用注释的内核参数`__grid_constant__`必须有`const`- 限定的非引用类型。

- 所有函数声明必须与任何一致`__grid_constant__`参数。

- 函数模板特化必须与主模板声明相匹配`__grid_constant__`参数。

- 函数模板实例化还必须与主模板声明相匹配`__grid_constant__`参数。

- The `__grid_constant__`注释被忽略`__tile_global__`函数参数。

示例：

```cuda


    struct MyStruct {
        int         x;
        mutable int y;
    };

    __device__ void external_function(const MyStruct&);

    __global__ void kernel(const __grid_constant__ MyStruct s) {
        // s.x++; // Compile error: tried to modify read-only memory
        // s.y++; // Undefined Behavior: tried to modify read-only memory

        // Compiler will NOT create a per-thread local copy of "s":
        external_function(s);
    }


```

请参阅示例<a href="https://godbolt.org/z/Goq9jrEeo" class="reference external">编译器资源管理器</a>.





<span id="id3"></span>

### 5.4.1.6. 注释摘要（Annotation Summary）

下表总结了 CUDA 注释并报告了每个注释适用于哪些执行空间及其有效位置。



|注释|`__host__` / `__device__` / `__host__``  ``__device__` | `__global__` |
|----|----|----|
| <a href="#inline-specifiers" class="reference internal">__无内联__</a>, <a href="#inline-specifiers" class="reference internal">__力内联__</a>, <a href="#inline-specifiers" class="reference internal">__内联提示__</a>|功能| ❌ |
| <a href="#restrict" class="reference internal">__限制__</a>|指针参数 |指针参数 |
| <a href="#grid-constant" class="reference internal">__网格常数__</a>| ❌ |参数|
| <a href="#launch-bounds" class="reference internal">__启动边界__</a>| ❌ |功能|
| <a href="#maximum-number-of-registers-per-thread" class="reference internal">__maxnreg__</a>| ❌ |功能|
| <a href="#cluster-dimensions" class="reference internal">__cluster_dims__</a>| ❌ |功能|

表 41注释摘要<a href="#id24" class="headerlink" title="Link to this table">#</a>{#id24 .table-no-stripes .table}









## 5.4.2. 内置类型和变量（Built-in Types and Variables）



<span id="host-compiler-extensions"></span>

### 5.4.2.1. 主机编译器类型扩展（Host Compiler Type Extensions）

只要主机编译器支持，CUDA 就允许使用非标准算术类型。支持以下类型：

 - 128 位整数类型 `__int128`.

 - 当主机编译器定义 `__SIZEOF_INT128__` 宏时，在 Linux 上受支持。

 - 128 位浮点类型 `__float128` 和 `_Float128` 在计算能力为 10.0 及更高版本的 GPU 设备上可用。 `__float128` 类型的常量表达式可以由编译器以较低精度的浮点表示形式进行处理。

 - 当主机编译器定义 `__SIZEOF_FLOAT128__` or `__FLOAT128__` 宏时，在 Linux x86 上受支持。

- `_Complex` <a href="https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Complex-Data-Types.html" class="reference external"> 类型 </a> 仅在主机代码中受支持。

128 位整数和浮点图块代码中不支持点类型。





<span id="id4"></span>

### 5.4.2.2. 内置变量（Built-in Variables）

用于指定和检索沿 x、y 和 z 维度的网格和块的内核配置的值的类型为 `dim3`。用于获取块和线程索引的变量的类型为 `uint3`。 `dim3` 和 `uint3` 都是由三个名为 `x`, `y` 和 `z` 的无符号值组成的普通结构。在 C++11 及更高版本中，`dim3` is 1.

 内置设备变量的所有组件的默认值：

- `dim3`` ``gridDim`：包含网格的维度，即沿 x、y 和 z 维度的线程块的数量。

- `dim3`` ``blockDim`：包含线程块的维度，即沿 x、y 和 z 的线程数量Dimensions.

- `uint3`` ``blockIdx`：包含网格内沿 x、y 和 z 维度的块索引。

- `uint3`` ``threadIdx`：包含块内沿 x、y 和 z 维度的线程索引。

- `int`` ``warpSize`：定义为 warp 中线程数的运行时值，通常为 `32`。另请参阅 <a href="../01-introduction/programming-model.html#programming-model-warps-simt" class="reference internal">Warps 和 SIMT</a> 了解扭曲的定义。

 平铺代码中不支持这些变量。要检索图块代码中的块 ID 或网格大小，请使用 `cuda::tiles::bid()` 和`cuda::tiles::num_blocks()` APIs.





<span id="id5"></span>

### 5.4.2.3. 内置类型（Built-in Types）

CUDA 提供从主机和设备都支持的基本整数和浮点类型派生的向量类型。下表显示了可用的矢量类型。



| C++ 基本类型 |向量 X1 |向量X2 |向量X3 |向量X4 |
|----|----|----|----|----|
| `signed`` ``char` | `char1` | `char2` | `char3` | `char4` |
| `unsigned`` ``char` | `uchar1` | `uchar2` | `uchar3` | `uchar4` |
| `signed`` ``short` | `short1` | `short2` | `short3` | `short4` |
| `unsigned`` ``short` | `ushort1` | `ushort2` | `ushort3` | `ushort4` |
| `signed`` ``int` | `int1` | `int2` | `int3` | `int4` |
| `unsigned` | `uint1` | `uint2` | `uint3` | `uint4` |
| `signed`` ``long` | `long1` | `long2` | `long3` | `long4_16a/long4_32a` |
| `unsigned`` ``long` | `ulong1` | `ulong2` | `ulong3` | `ulong4_16a/ulong4_32a` |
| `signed`` ``long`` ``long` | `longlong1` | `longlong2` | `longlong3` | `longlong4_16a/longlong4_32a` |
| `unsigned`` ``long`` ``long` | `ulonglong1` | `ulonglong2` | `ulonglong3` | `ulonglong4_16a/ulonglong4_32a` |
| `float` | `float1` | `float2` | `float3` | `float4` |
| `double` | `double1` | `double2` | `double3` | `double4_16a/double4_32a` |

表 42  向量类型<a href="#id25" class="headerlink" title="Link to this table">#</a> {#id25 .table-no-stripes .table}



请注意，`long4`, `ulong4`, `longlong4`, `ulonglong4` 和 `double4` 已在 CUDA 13 中弃用，并且可能会在将来的版本中删除。

------------------------------------------------------------------------

 下表详细信息向量类型的字节大小和对齐要求：



|类型 |尺寸|对齐|
|----|----|----|
| `char1`, `uchar1` | 1 | 1 |
| `char2`, `uchar2` | 2 | 2 |
| `char3`, `uchar3` | 3 | 1 |
| `char4`, `uchar4` | 4 | 4 |
| `short1`, `ushort1` | 2 | 2 |
| `short2`, `ushort2` | 4 | 4 |
| `short3`, `ushort3` | 6 | 2 |
| `short4`, `ushort4` | 8 | 8 |
| `int1`, `uint1` | 4 | 4 |
| `int2`, `uint2` | 8 | 8 |
| `int3`, `uint3` | 12 | 4 |
| `int4`, `uint4` | 16 | 16 |
| `long1`, `ulong1` | 4/8 **\*** | 4/8 **\*** |
| `long2`, `ulong2` | 8/16 **\*** | 8/16 **\*** |
| `long3`, `ulong3` | 12/24 **\*** | 4/8 **\*** |
| `long4`, `ulong4`（已弃用）| 16/32 **\*** | 16 **\*** |
| `long4_16a`, `ulong4_16a` | 16/32 **\*** | 16 |
| `long4_32a`, `ulong4_32a` | 16/32 **\*** | 32 |
| `longlong1`, `ulonglong1` | 8 | 8 |
| `longlong2`, `ulonglong2` | 16 | 16 |
| `longlong3`, `ulonglong3` | 24 | 8 |
| `longlong4`, `ulonglong4`（已弃用）| 32 | 32 16 | 16
| `longlong4_16a`, `ulonglong4_16a` | 32 | 16 |
| `longlong4_32a`, `ulonglong4_32a` | 32 | 32 |
| `float1` | 4 | 4 |
| `float2` | 8 | 8 |
| `float3` | 12 | 4 |
| `float4` | 16 | 16 |
| `double1` | 8 | 8 |
| `double2` | 16 | 16 |
| `double3` | 24 | 8 |
| `double4`（已弃用）| 32 | 32 16 | 16
| `double4_16a` | 32 | 16 |
| `double4_32a` | 32 | 32 |

表 43对齐要求<a href="#vector-types-alignment-requirements-in-device-code" class="headerlink" title="Link to this table">#</a>{#vector-types-alignment-requirements-in-device-code .table-no-stripes .table}



**\*** `long`在 C++ LLP64 数据模型（Windows 64 位）上为 4 字节，而在 C++ LP64 数据模型（Linux 64 位）上为 8 字节。

------------------------------------------------------------------------

向量类型是结构。它们的第一、第二、第三和第四个组件可以通过`x`, `y`, `z`， 和`w`分别是字段。

```cuda


    int sum(int4 value) {
        return value.x + value.y + value.z + value.w;
    }


```

它们都具有以下形式的工厂函数`make_<type_name>()`;例如：

```cuda


    int4 add_one(int x, int y, int z, int w) {
        return make_int4(x + 1, y + 1, z + 1, w + 1);
    }


```

如果主机代码未编译`nvcc`，向量类型和相关函数可以通过包含来导入`cuda_runtime.h`CUDA 工具包中提供的标头。







<span id="execution-configuration"></span>

## 5.4.3. 内核配置（Kernel Configuration）

任何呼叫`__global__` or `__tile_global__`函数必须为该调用指定一个“执行配置”。此执行配置定义了将用于在设备上执行功能的网格和块的尺寸，以及相关联的<a href="../02-programming-gpus/asynchronous-execution.html#cuda-streams" class="reference internal">溪流</a>.

执行配置是通过在表格中插入表达式来指定的`<<<grid_dim,`` ``block_dim,`` ``dynamic_smem_bytes,`` ``stream>>>`位于函数名称和带括号的参数列表之间，其中：

- `grid_dim`属于类型<a href="#built-in-variables" class="reference internal">暗淡3</a>并指定网格的尺寸和大小，使得`grid_dim.x`` ``*`` ``grid_dim.y`` ``*`` ``grid_dim.z`等于正在启动的块数。

- `block_dim`属于类型<a href="#built-in-variables" class="reference internal">暗淡3</a>并指定每个块的尺寸和大小，使得`block_dim.x`` ``*`` ``block_dim.y`` ``*`` ``block_dim.z`等于每个块的线程数。为了`__tile_global__`内核，`block_dim`必须是`1`因为编译器会在启动tile内核时决定调度多少个线程。

- `dynamic_smem_bytes`是一个可选的`size_t`默认为零的参数。它指定除了静态分配的内存之外，为该调用每块动态分配的共享内存中的字节数。该内存被使用`extern`` ``__shared__`数组（参见<a href="#shared-memory-specifier" class="reference internal">__共享__内存</a>).

- `stream`类型为 `cudaStream_t`（指针）并指定关联的流。 `stream` 是一个可选参数，默认为 `NULL`.

 以下示例显示了内核函数声明和调用：

```cuda


    __global__ void kernel(float* parameter);

    kernel<<<grid_dim, block_dim, dynamic_smem_bytes>>>(parameter);


```

执行配置的参数在实际函数的参数之前进行评估。

如果 `grid_dim` or `block_dim` 超出了设备允许的最大大小（如 <a href="compute-capabilities.html#compute-capabilities" class="reference internal">Compute 中指定），则函数调用将失败功能</a>, or if `dynamic_smem_bytes` 在考虑静态分配内存后大于可用共享内存。



<span id="cluster-dimensions"></span>

### 5.4.3.1. 线程块群集（Thread Block Cluster）

计算功能 9.0 及更高版本允许用户指定编译时线程块群集维度，以便内核可以在 CUDA 中使用 <a href="../02-programming-gpus/intro-to-cuda-cpp.html#thread-block-clusters" class="reference internal"> 群集层次结构</a>。可以使用 `__cluster_dims__` 属性通过以下语法指定编译时簇维度：`__cluster_dims__([x,`` ``[y,`` ``[z]]])`。下面的示例显示编译时簇大小在 X 维度上为 2，在 Y 和 Z 维度上为 1。

```cuda


    __global__ void __cluster_dims__(2, 1, 1) kernel(float* parameter);


```

`__cluster_dims__()` 的默认形式指定内核将作为网格集群启动。如果未指定集群维度，用户可以在启动时指定。未能在启动时指定维度将导致启动时错误。不能在 `__cluster_dims__` 内核上指定 `__tile_global__` 属性，因为编译器决定如何跨集群调度切片内核。

 线程块集群的维度也可以在运行时指定，并且可以使用 `cudaLaunchKernelEx` API 启动具有集群的内核。此 API 采用 `cudaLaunchConfig_t` 类型的配置参数、内核函数指针和内核参数。下面的示例显示了运行时内核配置。

```cuda


    __global__ void kernel(float parameter1, int parameter2) {}

    int main() {
        cudaLaunchConfig_t config = {0};
        // The grid dimension is not affected by cluster launch, and is still enumerated
        // using the number of blocks.
        // The grid dimension should be a multiple of cluster size.
        config.gridDim          = dim3{4};  // 4 blocks
        config.blockDim         = dim3{32}; // 32 threads per block
        config.dynamicSmemBytes = 1024;     // 1 KB

        cudaLaunchAttribute attribute[1];
        attribute[0].id               = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs    = attribute;
        config.numAttrs = 1;

        float parameter1 = 3.0f;
        int   parameter2 = 4;
        cudaLaunchKernelEx(&config, kernel, parameter1, parameter2);
    }


```

请参阅 <a href="https://cuda.godbolt.org/z/M67r3a5zM" class="reference external">Compiler Explorer 上的示例</a>.





<span id="id6"></span>

### 5.4.3.2. Launch Bounds（Launch Bounds）

如 <a href="../02-programming-gpus/writing-simt-kernels.html#_2-3-3-7-kernel-启动与-occupancy-kernel-launch-and-occupancy" class="reference internal">Kernel Launch and Occupancy 中讨论的</a>部分，使用更少的寄存器允许更多线程和线程块驻留在多处理器上，从而提高性能。

因此，编译器使用启发式方法来最小化寄存器的使用，同时保持<a href="../02-programming-gpus/writing-simt-kernels.html#_2-3-3-3-registers" class="reference internal">寄存器溢出</a>并将指令数降至最低。应用程序可以选择通过以使用指定的启动边界的形式向编译器提供附加信息来帮助这些启发式方法`__launch_bounds__()`a 的定义中的限定符`__global__`功能：

```cuda


    __global__ void
    __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster)
    MyKernel(...) {
        ...
    }


```

- `maxThreadsPerBlock`指定应用程序启动时每块的最大线程数`MyKernel()`;它编译为`.maxntid`PTX 指令。

- `minBlocksPerMultiprocessor`是可选的，指定每个多处理器所需的最小驻留块数；它编译为`.minnctapersm`PTX 指令。

- `maxBlocksPerCluster`是可选的，指定应用程序启动时每个集群所需的最大线程块数`MyKernel()`;它编译为`.maxclusterrank`PTX 指令。

如果指定了启动边界，编译器首先得出上限，`L`，关于内核应使用的寄存器的数量。这确保了`minBlocksPerMultiprocessor`块（或单个块，如果`minBlocksPerMultiprocessor`未指定）的`maxThreadsPerBlock`线程可以驻留在多处理器上。请参阅<a href="../02-programming-gpus/writing-simt-kernels.html#_2-3-3-7-kernel-启动与-occupancy-kernel-launch-and-occupancy" class="reference internal">占用</a>内核使用的寄存器数量与每个块分配的寄存器数量之间关系的部分。然后编译器优化寄存器的使用，如下所示：

- 如果初始寄存器使用量超过`L`，编译器会减少它，直到它小于或等于`L`。这通常会导致本地内存使用量增加和/或指令数量增加。

- 如果初始寄存器使用率低于`L`

  - If `maxThreadsPerBlock`已指定，但是`minBlocksPerMultiprocessor`不是，编译器使用`maxThreadsPerBlock`确定寄存器使用阈值之间的转换`n`和`n`` ``+`` ``1`居民区。当少使用一个寄存器为额外的常驻块腾出空间时，就会发生这种情况。然后，编译器将应用与未指定启动边界时类似的启发式方法。

 - 如果同时指定了 `minBlocksPerMultiprocessor` 和 `maxThreadsPerBlock`，编译器可能会将寄存器使用量增加到 `L`，以减少指令数量并更好地隐藏单线程指令的延迟。如果执行

A 内核将无法启动其中：

 - 每个块的线程数超过其启动限制 `maxThreadsPerBlock`.

 - 每个集群的线程块数超过其启动限制 `maxBlocksPerCluster`.

 CUDA 内核所需的每线程资源可能会以不希望的方式限制最大块大小。为了保持与未来硬件和工具包的前向兼容性，并确保至少一个线程块可以在流式多处理器上运行，开发人员应包含单个参数 `__launch_bounds__(maxThreadsPerBlock)`，它指定内核启动时的最大块大小。如果不这样做，可能会导致“启动时请求的资源过多”错误。在某些情况下，提供 `__launch_bounds__(maxThreadsPerBlock,minBlocksPerMultiprocessor)` 的两个参数版本可以提高性能。 `minBlocksPerMultiprocessor` 的最佳值应通过对每个内核的详细分析来确定。

 内核的最佳启动范围通常因主要体系结构修订而异。以下代码示例说明了如何在设备代码中使用 `__CUDA_ARCH__` <a href="#cuda-arch-macro" class="reference internal">macro</a>.

```cuda


    #define THREADS_PER_BLOCK  256

    #if __CUDA_ARCH__ >= 900
        #define MY_KERNEL_MAX_THREADS  (2 * THREADS_PER_BLOCK)
        #define MY_KERNEL_MIN_BLOCKS   3
    #else
        #define MY_KERNEL_MAX_THREADS  THREADS_PER_BLOCK
        #define MY_KERNEL_MIN_BLOCKS   2
    #endif

    __global__ void
    __launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
    MyKernel(...) {
        ...
    }


```

 进行管理。当使用每个块的最大线程数（指定为 `MyKernel` 的第一个参数）调用 `__launch_bounds__()` 时，很容易使用 `MY_KERNEL_MAX_THREADS` 作为执行中每个块的线程数配置：

```cuda


    // Host code
    MyKernel<<<blocksPerGrid, MY_KERNEL_MAX_THREADS>>>(...);


```

但是，这不起作用，因为 `__CUDA_ARCH__` 在主机代码中未定义，如 <a href="#execution-space-specifiers" class="reference internal"> 执行空间说明符</a> 部分中所述。所以，`MyKernel`将以每块 256 个线程启动。相反，应确定每个块的线程数：

 - 在编译时使用不依赖于 `__CUDA_ARCH__` 的宏或常量，例如

  ```cuda


      // Host code
      MyKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(...);


  ```

 - 或者在运行时根据计算能力

  ```cuda


      // Host code
      cudaGetDeviceProperties(&deviceProp, device);
      int threadsPerBlock = (deviceProp.major >= 9) ? 2 * THREADS_PER_BLOCK : THREADS_PER_BLOCK;
      MyKernel<<<blocksPerGrid, threadsPerBlock>>>(...);


  ```

The `--resource-usage` 编译器选项报告寄存器使用情况。 <a href="https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#occupancy-calculator" class="reference external">CUDA 分析器</a> 报告占用率，可用于导出驻留块的数量。





<span id="id7"></span>

### 5.4.3.3. 每个线程的最大寄存器数（Maximum Number of Registers per Thread）

为了启用低级性能调整，CUDA C++ 提供了 `__maxnreg__()` 函数限定符，它将性能调整信息传递给后端优化编译器。 `__maxnreg__()` 限定符指定可以分配给线程块中单个线程的寄存器的最大数量。在`__global__`函数的定义中：

```cuda


    __global__ void
    __maxnreg__(maxNumberRegistersPerThread)
    MyKernel(...) {
        ...
    }


```

The `maxNumberRegistersPerThread`变量指定了内核`MyKernel()`的线程块中分配给单个线程的最大寄存器数量；它编译为 `.maxnreg` PTX 指令。

The `__launch_bounds__()` 和 `__maxnreg__()` 限定符不能一起应用于同一内核。

The `--maxrregcount`` ``<N>` 编译器选项可用于控制文件中所有 `__global__` 函数的寄存器使用。对于带有 `__maxnreg__` 限定符的内核函数，此选项将被忽略。







<span id="synchronization-functions"></span>

## 5.4.4. 同步原语（Synchronization Primitives）



### 5.4.4.1. 线程块同步函数（Thread Block Synchronization Functions）

```cuda


    void __syncthreads();
    int  __syncthreads_count(int predicate);
    int  __syncthreads_and(int predicate);
    int  __syncthreads_or(int predicate);


```

内在函数协调同一块内线程之间的通信。当块中的线程访问共享或全局内存中的相同地址时，可能会发生先读后写、先写后读或先写后写危险。通过在此类访问之间同步线程可以避免这些危险。

 内在函数具有以下语义：

- `__syncthreads*()` 等待，直到线程块中的所有未退出线程同时到达程序中相同的 `__syncthreads*()` 内在函数调用或退出。

- `__syncthreads*()`在参与线程之间提供内存排序：对 `__syncthreads*()` 内在函数的调用强烈发生在任何参与线程从等待或退出中解除阻塞之前（请参阅 <a href="https://eel.is/c++draft/intro.races" class="reference external">C++ 规范 [intro.races]</a>）。

以下示例演示如何使用 `__syncthreads()` 同步线程块内的线程，并安全地对共享数组的元素求和条件代码中允许使用threads:

```cuda


    #include <cuda_runtime_api.h>
    #include <memory.h>
    #include <cstdlib>
    #include <ctime>
    #include <stdio.h>


    // assuming blockDim.x is 128
    __global__ void example_syncthreads(int* input_data, int* output_data)
    {
        __shared__ int shared_data[128];
        shared_data[threadIdx.x] = input_data[blockDim.x*blockIdx.x + threadIdx.x];

        // All threads synchronize, guaranteeing all writes to 'shared_data' are ordered
        // before any thread is unblocked from '__syncthreads()':
        __syncthreads();

        // A single thread safely reads 'shared_data':
        if (threadIdx.x == 0) {
            float sum = 0;
            for (int i = 0; i < blockDim.x; ++i) {
                sum += shared_data[i];
            }
            output_data[blockIdx.x] = sum;
        }
    }


    void initArray(int* A, int length)
    {
         std::srand(std::time({}));
        for(int i=0; i<length; i++)
        {
            A[i] = int(10.f * (rand() / (float)RAND_MAX));
        }
    }


    int main(int argc, char** arg)
    {
        constexpr int block_size = 128;
        constexpr int input_length = 1024;
        constexpr int output_length = input_length / block_size;

        // Pointers to memory vectors
        int* input = nullptr;
        int* output = nullptr;
        int* comparisonResult = (int*)malloc(output_length*sizeof(int));

        // Use unified memory to allocate buffers
        cudaMallocManaged(&input, input_length*sizeof(int));
        cudaMallocManaged(&output, output_length*sizeof(int));

        initArray(input, input_length);
        int grid_size = input_length / block_size;

        example_syncthreads<<<grid_size, block_size>>>(input, output);
        cudaDeviceSynchronize();

        for(int i=0; i<output_length; i++)
        {
            comparisonResult[i] = 0;
            for(int j=0; j < block_size; j++)
            {
                comparisonResult[i] += input[i*block_size + j];
            }
        }

        for(int i=0; i< output_length; i++)
        {
            if(output[i] != comparisonResult[i])
            {
                printf("Results do not match at index %d: %d != %d\n", i, output[i], comparisonResult[i]);
                exit(-1);
            }
        }
        printf("Test passed\n");

        return 0;
    }


```

The `__syncthreads*()` 内在函数，但前提是条件在整个线程块中统一计算。否则，执行可能会挂起或产生意外的副作用。

以下示例演示了有效行为：

```cuda


    // assuming blockDim.x is 128
    __global__ void syncthreads_valid_behavior(int* input_data, int* output_data) {
        __shared__ int shared_data[128];
        shared_data[threadIdx.x] = input_data[threadIdx.x];
        if (blockIdx.x > 0) { // CORRECT, uniform condition across all block threads
            __syncthreads();
            output_data[threadIdx.x] = shared_data[128 - threadIdx.x];
        }
    }


```

，而以下示例则演示了无效行为，例如内核挂起或未定义的行为：

```cuda


    // assuming blockDim.x is 128
    __global__ void syncthreads_invalid_behavior1(int* input_data, int* output_data) {
        __shared__ int shared_data[256];
        shared_data[threadIdx.x] = input_data[threadIdx.x];
        if (threadIdx.x > 0) { // WRONG, non-uniform condition
            __syncthreads();   // Undefined Behavior
            output_data[threadIdx.x] = shared_data[128 - threadIdx.x];
        }
    }


```

```cuda


    // assuming blockDim.x is 128
    __global__ void syncthreads_invalid_behavior2(int* input_data, int* output_data) {
        __shared__ int shared_data[256];
        shared_data[threadIdx.x] = input_data[threadIdx.x];
        for (int i = 0; i < blockDim.x; ++i) {
            if (i == threadIdx.x) { // WRONG, non-uniform condition
                __syncthreads();    // Undefined Behavior
            }
        }
        output_data[threadIdx.x] = shared_data[128 - threadIdx.x];
    }


```

------------------------------------------------------------------------

`__syncthreads()` **带有谓词的变体**：

```cuda


    int __syncthreads_count(int predicate);


```

 与 `__syncthreads()` 相同，只是它为所有未退出线程计算谓词块并返回谓词计算结果为非零值的线程数。

```cuda


    int __syncthreads_and(int predicate);


```

 与 `__syncthreads()` 相同，只是它计算块中所有非退出线程的谓词。当且仅当谓词对所有线程计算结果都为非零值时，它才返回非零值。

```cuda


    int __syncthreads_or(int predicate);


```

 与 `__syncthreads()` 相同，只是它计算块中所有非退出线程的谓词。当且仅当谓词求值为其中一个或多个非零值时，它才返回非零值。





### 5.4.4.2. Warp 同步函数（Warp Synchronization Function）

```cuda


    void __syncwarp(unsigned mask = 0xFFFFFFFF);


```

内部函数 `__syncwarp()` 协调同一 warp 内的线程之间的通信。当 warp 中的某些线程访问共享或全局内存中的相同地址时，可能会发生潜在的先读后写、先写后读或先写后写危险。通过同步这些访问之间的线程可以避免这些数据危险。

调用`__syncwarp(mask)`在命名为 warp 的参与线程之间提供内存排序`mask`: 调用`__syncwarp(mask)`强烈地发生在之前（参见<a href="https://eel.is/c++draft/intro.races" class="reference external">C++ 规范 [intro.races]</a>）任何命名的经线`mask`解除等待或退出的阻塞。

这些功能受<a href="#warp-sync-intrinsic-constraints" class="reference internal">Warp __sync 内在约束</a>.

下面的例子演示了如何使用`__syncwarp()`同步扭曲中的线程以安全地访问共享内存数组：

```cuda


    __global__ void example_syncwarp(int* input_data, int* output_data) {
        if (threadIdx.x < warpSize) {
            __shared__ int shared_data[warpSize];
            shared_data[threadIdx.x] = input_data[threadIdx.x];

            __syncwarp(); // equivalent to __syncwarp(0xFFFFFFFF)
            if (threadIdx.x == 0)
                output_data[0] = shared_data[1];
        }
    }


```





<span id="id8"></span>

### 5.4.4.3. 内存栅栏功能（Memory Fence Functions）

CUDA 编程模型假设弱有序内存模型。换句话说，CUDA 线程将数据写入共享内存、全局内存、页锁定主机内存或对等设备内存的顺序不一定是另一个 CUDA 或主机线程观察正在写入的数据的顺序。在没有内存围栏或同步的情况下读取或写入同一内​​存位置会导致未定义的行为。

在下面的示例中，线程 1 执行`writeXY()`，当线程 2 执行时`readXY()`.

```cuda


    __device__ int X = 1, Y = 2;

    __device__ void writeXY() {
        X = 10;
        Y = 20;
    }

    __device__ void readXY() {
        int B = Y;
        int A = X;
    }


```

两个线程同时读取和写入相同的内存位置，`X`和`Y`。任何数据竞争都会导致未定义的行为并且没有定义的语义。因此，所得值`A`和`B`可以是任何东西。

内存栅栏和同步功能强制执行<a href="https://en.cppreference.com/w/cpp/atomic/memory_order" class="reference external">顺序一致的顺序</a>内存访问。这些功能的不同之处在于<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#thread-scopes" class="reference external">线程范围</a>其中强制执行顺序，但独立于访问的内存空间，包括共享内存、全局内存、页锁定主机内存和对等设备的内存。



暗示

建议使用`cuda::atomic_thread_fence`由提供<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic/atomic_thread_fence.html" class="reference external">库++</a>出于安全和便携性原因，只要有可能。



**块级内存栅栏**



CUDA C++



```cuda


    // <cuda/atomic> header
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block);


```

确保：

- 在调用之前，调用线程对所有内存进行的所有写入`cuda::atomic_thread_fence()`被调用线程块中的所有线程观察到发生在调用线程在调用之后对所有内存进行的所有写入之前`cuda::atomic_thread_fence()`;

- 在调用之前，调用线程对所有内存进行的所有读取`cuda::atomic_thread_fence()`在调用之后，在调用线程对所有内存进行的所有读取之前排序`cuda::atomic_thread_fence()`.



内在因素



```cuda


    void __threadfence_block();


```

确保：

- 在调用之前，调用线程对所有内存进行的所有写入`__threadfence_block()`被调用线程块中的所有线程观察到发生在调用线程在调用之后对所有内存进行的所有写入之前`__threadfence_block()`;

- 在调用之前，调用线程对所有内存进行的所有读取`__threadfence_block()`在调用之后，在调用线程对所有内存进行的所有读取之前排序`__threadfence_block()`.





**设备级内存栅栏**



CUDA C++



```cuda


    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);


```

确保：

- 在调用后，调用线程不会写入所有内存`cuda::atomic_thread_fence()`设备中的任何线程都会观察到在调用线程之前对所有内存进行任何写入之前发生的情况`cuda::atomic_thread_fence()`.



内在因素



```cuda


    void __threadfence();


```

确保：

- 在调用后，调用线程不会写入所有内存`__threadfence()`设备中的任何线程都会观察到在调用线程之前对所有内存进行任何写入之前发生的情况`__threadfence()`.





**系统级内存栅栏**



CUDA C++



```cuda


    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);


```

确保：

- 在调用之前，调用线程对所有内存进行的所有写入`cuda::atomic_thread_fence()`设备中的所有线程、主机线程和对等设备中的所有线程都会观察到，在调用线程之后对所有内存进行所有写入之前发生`cuda::atomic_thread_fence()`.



内在因素



```cuda


    void __threadfence_system();


```

确保：

- 在调用之前，调用线程对所有内存进行的所有写入`__threadfence_system()`设备中的所有线程、主机线程和对等设备中的所有线程都会观察到，在调用线程之后对所有内存进行所有写入之前发生`__threadfence_system()`.





在前面的代码示例中，我们可以在代码中插入内存栅栏，如下所示：



CUDA C++



```cuda


    #include <cuda/atomic>

    __device__ int X = 1, Y = 2;

    __device__ void writeXY() {
        X = 10;
        cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
        Y = 20;
    }

    __device__ void readXY() {
        int B = Y;
        cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
        int A = X;
    }


```



内在因素



```cuda


    __device__ int X = 1, Y = 2;

    __device__ void writeXY() {
        X = 10;
        __threadfence();
        Y = 20;
    }

    __device__ void readXY() {
        int B = Y;
        __threadfence();
        int A = X;
    }


```





对于此代码，可以观察到以下结果：

- `A`等于 1 且`B`等于2，即`readXY()`之前执行过`writeXY()`,

- `A`等于 10 且`B`等于20，即`writeXY()`之前执行过`readXY()`.

- `A`等于 10 且`B`等于2。

- 情况`A`是 1 并且`B`是 20 是不可能的，因为内存栅栏确保写入`X`在写入之前可见`Y`.

如果线程1和线程2属于同一个块，那么使用块级栅栏就足够了。如果线程 1 和 2 不属于同一块，则如果它们是来自同一设备的 CUDA 线程，则必须使用设备级栅栏；如果它们是来自两个不同设备的 CUDA 线程，则必须使用系统级栅栏。

以下代码示例说明了一个常见用例，其中线程消耗其他线程生成的数据。该内核在一次调用中计算 N 个数字数组的总和。

- 每个块首先对数组的子集求和并将结果存储在全局内存中。

- 当所有块完成后，最后一个块从全局内存中读取这些部分和中的每一个，并将它们加在一起以获得最终结果。

- 为了确定哪个块最后完成，每个块自动递增一个计数器以表示计算完成并存储其部分和（请参阅<a href="#atomic-functions" class="reference internal">原子函数</a>部分了解更多详情）。最后一个块接收的计数器值等于`gridDim.x`` ``-`` ``1`.

如果在存储部分和和递增计数器之间没有栅栏，则计数器可以在存储部分和之前递增。这可能会导致计数器达到`gridDim.x`` ``-`` ``1`并允许最后一个块在内存中更新之前开始读取部分和。



笔记

内存栅栏仅影响内存操作的执行顺序；它不保证这些操作对其他线程的可见性。



在下面的代码示例中，内存操作的可见性`result`通过将变量声明为来确保变量`volatile`。有关更多详细信息，请参阅`volatile`-<a href="cpp-language-support.html#volatile-qualifier" class="reference internal">限定变量</a>部分。

```cuda


    #include <cuda/atomic>

    __device__ int count = 0;

    __global__ void sum(const float*    array,
                        int             N,
                        volatile float* result) {
        __shared__ bool isLastBlockDone;
        // Each block sums a subset of the input array.
        float partialSum = calculatePartialSum(array, N);

        if (threadIdx.x == 0) {
            // Thread 0 of each block stores the partial sum to global memory.
            // The compiler will use a store operation that bypasses the L1 cache
            // since the "result" variable is declared as volatile.
            // This ensures that the threads of the last block will read the correct
            // partial sums computed by all other blocks.
            result[blockIdx.x] = partialSum;

            // Thread 0 makes sure that the increment of the "count" variable is
            // only performed after the partial sum has been written to global memory.
            cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);

            // Thread 0 signals that it is done.
            int count_old = atomicInc(&count, gridDim.x);

            // Thread 0 determines if its block is the last block to be done.
            isLastBlockDone = (count_old == (gridDim.x - 1));
        }
        // Synchronize to make sure that each thread reads the correct value of
        // isLastBlockDone.
        __syncthreads();

        if (isLastBlockDone) {
            // The last block sums the partial sums stored in result[0 .. gridDim.x-1]
            float totalSum = calculateTotalSum(result);

            if (threadIdx.x == 0) {
                // Thread 0 of last block stores the total sum to global memory and
                // resets the count variable, so that the next kernel call works
                // properly.
                result[0] = totalSum;
                count     = 0;
            }
        }
    }


```







<span id="id9"></span>

## 5.4.5. 原子函数（Atomic Functions）

原子函数对共享数据执行读取-修改-写入操作，使它们看起来像是一步执行的。原子性确保每个操作要么完全完成，要么根本不完成，从而为所有参与线程提供一致的数据视图。

CUDA 通过五种方式提供原子函数：

扩展 CUDA C++ 原子函数，<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic.html" class="reference external">CUDA::原子</a>和<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html" class="reference external">CUDA::atomic_ref</a>.
- 它们在主机和设备代码中都是允许的。

- 他们遵循<a href="https://en.cppreference.com/w/cpp/atomic/atomic.html" class="reference external">C++ 标准原子操作</a>语义。

- 他们允许指定<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes" class="reference external">线程范围</a>的原子操作。

标准 C++ 原子函数，<a href="https://en.cppreference.com/w/cpp/atomic/atomic.html" class="reference external">cuda::std::原子</a>和<a href="https://en.cppreference.com/w/cpp/atomic/atomic_ref.html" class="reference external">cuda::std::atomic_ref</a>.
- 它们在主机和设备代码中都是允许的。

- 他们遵循<a href="https://en.cppreference.com/w/cpp/atomic/atomic.html" class="reference external">C++ 标准原子操作</a>语义。

- 他们不允许指定<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes" class="reference external">线程范围</a>的原子操作。

编译器<a href="#built-in-atomic-functions" class="reference internal">内置原子函数</a>, `__nv_atomic_<op>()`.
- 它们自 CUDA 12.8 起可用。

- 它们仅允许在设备代码中使用。

- 他们遵循<a href="https://en.cppreference.com/w/cpp/atomic/memory_order.html" class="reference external">C++ 标准原子内存顺序</a>语义。

- 他们允许指定<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes" class="reference external">线程范围</a>的原子操作。

- 它们具有相同的内存排序语义<a href="https://en.cppreference.com/w/cpp/atomic/atomic.html" class="reference external">C++ 标准原子操作</a>.

- 它们支持以下允许的数据类型的子集<a href="https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html" class="reference external">cuda::std::原子</a>和<a href="https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html" class="reference external">cuda::std::atomic_ref</a>，128 位数据类型除外。

- 它们在图块代码中不受支持。

CUDA Tile C++ 原子函数（例如，`cuda::tiles::atomic_load`):
- 仅在图块代码中允许使用它们。

- 他们遵循<a href="https://en.cppreference.com/w/cpp/atomic/memory_order.html" class="reference external">C++ 标准原子内存顺序</a>语义。

- 它们允许通过指定线程范围`cuda::tiles::thread_scope`.

<a href="#legacy-atomic-functions" class="reference internal">遗留原子函数</a>, `atomic<Op>()`.
- 它们仅允许在设备代码中使用。

- 他们只支持`memory_order_relaxed` <a href="https://en.cppreference.com/w/cpp/atomic/memory_order.html" class="reference external">C++ 原子内存语义</a>.

- 他们允许指定<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes" class="reference external">线程范围</a>原子操作作为函数名称的一部分。

- 不像<a href="#built-in-atomic-functions" class="reference internal">内置原子函数</a>，遗留原子函数仅保证原子性，不会引入同步点（栅栏）。

- 它们支持以下允许的数据类型的子集<a href="#built-in-atomic-functions" class="reference internal">内置原子函数</a>。原子`add`操作支持其他数据类型。



暗示

使用<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives.html" class="reference external">扩展 CUDA C++ 原子函数</a>由提供`libcu++`出于效率、安全性和便携性的考虑，建议使用。





<span id="id10"></span>

### 5.4.5.1. 遗留原子函数（Legacy Atomic Functions）

传统原子函数对存储在全局或共享内存中的 32 位、64 位或 128 位字执行原子读-修改-写操作。例如，`atomicAdd()`函数读取全局或共享内存中特定地址的字，向其添加一个数字，然后将结果写回同一地址。

- 原子函数只能在设备函数中使用。

- 对于矢量类型，例如`__half2`, `__nv_bfloat162`, `float2`， 和`float4`，对向量的每个元素执行读-修改-写操作。不保证整个向量在单次访问中是原子的。

本节中描述的原子函数有一个<a href="https://en.cppreference.com/w/cpp/atomic/memory_order" class="reference external">内存排序</a> of `cuda::std::memory_order_relaxed`并且仅在特定的情况下是原子的<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#thread-scopes" class="reference external">线程范围</a>:

- 不带后缀的原子 API（例如 `atomicAdd`）在 `cuda::thread_scope_device`.

 范围内是原子的 - 具有 `_block` 后缀的原子 API（例如 `atomicAdd_block`）在 `cuda::thread_scope_block`.

 范围内是原子的 - 具有 `_system` 后缀的原子 API，例如， `atomicAdd_system`，如果满足特定的 `cuda::thread_scope_system` 条件，则在 <a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#atomicity" class="reference external"> 范围内是原子的 </a>.

 以下示例显示 CPU 和 GPU 以原子方式更新地址 `addr`:

```cuda


    #include <cuda_runtime.h>

    __global__ void atomicAdd_kernel(int* addr) {
        atomicAdd_system(addr, 10);
    }

    void test_atomicAdd(int device_id) {
        int* addr;
        cudaMallocManaged(&addr, 4);
        *addr = 0;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device_id);
        if (deviceProp.concurrentManagedAccess != 1) {
            return; // the device does not coherently access managed memory concurrently with the CPU
        }

        atomicAdd_kernel<<<...>>>(addr);
        __sync_fetch_and_add(addr, 10);  // CPU atomic operation
    }


```

------------------------------------------------------------------------

 处的整数值 请注意，任何原子操作都可以基于 `atomicCAS()` 实现（比较和交换）。例如，单精度浮点数的 `atomicAdd()` 可以实现如下：

```cuda


    #include <cuda/memory>
    #include <cuda/std/bit>

    __device__ float customAtomicAdd(float* d_ptr, float value) {
        volatile unsigned* d_ptr_unsigned = reinterpret_cast<unsigned*>(d_ptr);
        unsigned  old_value      = *d_ptr_unsigned;
        unsigned  assumed;
        do {
            assumed                          = old_value;
            float    assumed_float           = cuda::std::bit_cast<float>(assumed);
            float    expected_value          = assumed_float + value;
            unsigned expected_value_unsigned = cuda::std::bit_cast<unsigned>(expected_value);
            old_value                        = atomicCAS(d_ptr_unsigned, assumed, expected_value_unsigned);
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old_value);
        return cuda::std::bit_cast<float>(old_value);
    }


```

请参阅 <a href="https://godbolt.org/z/676e5bc7a" class="reference external">Compiler Explorer</a>.



<span id="id11"></span>

#### 5.4.5.1.1. `atomicAdd()`（`atomicAdd()`）

```cuda


    T atomicAdd(T* address, T val);


```

 上的示例，该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 计算 `old`` ``+`` ``val`.

3. 将结果存储回内存中的同一地址。

该函数返回 `old` 值。

`atomicAdd()` 支持以下数据类型：计算能力 8.x 及更高版本的设备上的

- `int`, `unsigned`, `unsigned`` ``long`` ``long`, `float`, `double`, `__half2`, `__half`.

- `__nv_bfloat16`, `__nv_bfloat162`。计算能力 9.x 及更高版本的设备上的

- `float2`, `float4`，并且仅支持全局内存地址。

应用于矢量类型的 `atomicAdd()` 的原子性（例如 `__half2` or `float4`）是针对每个组件单独保证的；不保证整个向量作为单次访问是原子的。





#### 5.4.5.1.2. `atomicSub()`（`atomicSub()`）

```cuda


    T atomicSub(T* address, T val);


```

 该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 `old` 的 `address` 值。

2. 计算 `old`` ``-`` ``val`.

3. 将结果存储回内存中的同一地址。

 该函数返回 `old` 值。

`atomicSub()` 支持以下数据类型：

- `int`, `unsigned`





#### 5.4.5.1.3. `atomicInc()`（`atomicInc()`）

```cuda


    unsigned atomicInc(unsigned* address, unsigned val);


```

 该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 计算 `old`` ``>=`` ``val`` ``?`` ``0`` ``:`` ``(old`` ``+`` ``1)`.

3. 将结果存储回内存中的同一地址。

 函数返回 `old` 值。





#### 5.4.5.1.4. `atomicDec()`（`atomicDec()`）

```cuda


    unsigned atomicDec(unsigned* address, unsigned val);


```

 函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 计算 `(old`` ``==`` ``0`` ``||`` ``old`` ``>`` ``val)`` ``?`` ``val`` ``:`` ``(old`` ``-`` ``1)`.

3. 将结果存储回内存中的同一地址。

 函数返回 `old` 值。





#### 5.4.5.1.5. `atomicAnd()`（`atomicAnd()`）

```cuda


    T atomicAnd(T* address, T val);


```

 函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 计算 `old`` ``&`` ``val`.

3. 将结果存储回内存中的同一地址。

 该函数返回 `old` 值。

`atomicAnd()` 支持以下数据类型：

- `int`, `unsigned`, `unsigned`` ``long`` ``long`.





#### 5.4.5.1.6. `atomicOr()`（`atomicOr()`）

```cuda


    T atomicOr(T* address, T val);


```

 该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 计算 `old`` ``|`` ``val`.

3. 将结果存储回内存中的同一地址。

函数返回`old`

`atomicOr()` 支持以下数据类型：

- `int`, `unsigned`, `unsigned`` ``long`` ``long`.





#### 5.4.5.1.7. `atomicXor()`（`atomicXor()`）

```cuda


    T atomicXor(T* address, T val);


```

 该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 计算 `old`` ``^`` ``val`.

3. 将结果存储回内存中的同一地址。

 该函数返回 `old` 值。

`atomicXor()` 支持以下数据类型：

- `int`, `unsigned`, `unsigned`` ``long`` ``long`.





#### 5.4.5.1.8. `atomicMin()`（`atomicMin()`）

```cuda


    T atomicMin(T* address, T val);


```

 该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 计算 `old` 和 `val`.

3 中的最小值。  将结果存储回内存中的同一地址。

 该函数返回 `old` 值。

`atomicMin()` 支持以下数据类型：

- `int`, `unsigned`, `unsigned`` ``long`` ``long`, `long`` ``long`.





#### 5.4.5.1.9. `atomicMax()`（`atomicMax()`）

```cuda


    T atomicMax(T* address, T val);


```

 该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 计算 `old` 和 `val`.

3 的最大值。  将结果存储回内存中的同一地址。

 函数返回 `old` 值。

`atomicMax()` 支持以下数据类型：

- `int`, `unsigned`, `unsigned`` ``long`` ``long`, `long`` ``long`.





#### 5.4.5.1.10. `atomicExch()`（`atomicExch()`）

```cuda


    T atomicExch(T* address, T val);


```

```cuda


    template<typename T>
    T atomicExch(T* address, T val); // only 128-bit types, compute capability 9.x and higher


```

该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 将 `val` 存储回内存中的同一地址。

 该函数返回 `old` 值。

`atomicExch()` 支持以下数据类型：

- `int`, `unsigned`, `unsigned`` ``long`` ``long`, `float`.

 C++ 模板函数 `atomicExch()` 支持 128 位类型，并具有以下要求：

 - 计算能力 9.x

- `T` 必须对齐到 16 字节，即 `alignof(T)`` ``>=`` ``16`.

- `T` 必须是普通可复制的，即 `std::is_trivially_copyable_v<T>`.

 - 对于 C++03 及更早版本：`T` 必须是普通可构造的，即 `std::is_default_constructible_v<T>`.





#### 5.4.5.1.11. `atomicCAS()`（`atomicCAS()`）

```cuda


    T atomicCAS(T* address, T compare, T val);


```

```cuda


    template<typename T>
    T atomicCAS(T* address, T compare, T val);  // only 128-bit types, compute capability 9.x and higher


```

 该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 计算 `old`` ``==`` ``compare`` ``?`` ``val`` ``:`` ``old`.

3. 将结果存储回内存中的同一地址。

 函数返回 `old` 值。

`atomicCAS()` 支持以下数据类型：

- `int`, `unsigned`, `unsigned`` ``long`` ``long`, `unsigned`` ``short`.

 C++ 模板函数 `atomicCAS()` 支持 128 位类型，具有以下要求：

 - 计算能力 9.x 和

- `T` 必须对齐到 16 个字节，即 `alignof(T)`` ``>=`` ``16`.

- `T` 必须是 trivially 可复制的，即 `std::is_trivially_copyable_v<T>`.

- 对于 C++03 及更早版本：`T` 必须是 trivially 可构造的，即`std::is_default_constructible_v<T>`.







<span id="id12"></span>

### 5.4.5.2. 内置原子函数（Built-in Atomic Functions）

CUDA 12.8 及更高版本支持原子操作的 CUDA 编译器内置函数，遵循与 <a href="https://en.cppreference.com/w/cpp/atomic/atomic.html" class="reference external">C++ 标准原子操作</a> 和 CUDA <a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes" class="reference external"> 线程作用域</a> 相同的内存排序语义。这些函数遵循 <a href="https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html" class="reference external">GNU 的原子内置函数签名 </a>，并带有线程作用域的额外参数。当支持内置原子函数时，

`nvcc` 定义宏 `__CUDACC_DEVICE_ATOMIC_BUILTINS__`。

 下面列出了 <a href="https://en.cppreference.com/w/cpp/atomic/atomic.html" class="reference external"> 内存的原始枚举器命令</a>和<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes" class="reference external">线程作用域</a>，用作内置原子函数的`order`和`scope`参数：

```cuda


    // atomic memory orders
    enum {
       __NV_ATOMIC_RELAXED,
       __NV_ATOMIC_CONSUME,
       __NV_ATOMIC_ACQUIRE,
       __NV_ATOMIC_RELEASE,
       __NV_ATOMIC_ACQ_REL,
       __NV_ATOMIC_SEQ_CST
    };


```

```cuda


    // thread scopes
    enum {
       __NV_THREAD_SCOPE_THREAD,
       __NV_THREAD_SCOPE_BLOCK,
       __NV_THREAD_SCOPE_CLUSTER,
       __NV_THREAD_SCOPE_DEVICE,
       __NV_THREAD_SCOPE_SYSTEM
    };


```

-内存顺序对应于<a href="https://en.cppreference.com/w/cpp/atomic/memory_order" class="reference external">C++标准原子操作内存order</a>.

- 线程作用域遵循 `cuda::thread_scope` <a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#thread-scopes" class="reference external"> 定义</a>.

- `__NV_ATOMIC_CONSUME` 内存顺序当前使用更强的 `__NV_ATOMIC_ACQUIRE` 内存顺序来实现。

- `__NV_THREAD_SCOPE_THREAD` 线程作用域当前使用更宽的 `__NV_THREAD_SCOPE_BLOCK` 线程作用域来实现。

 示例:

```cuda


    __device__ T __nv_atomic_load_n(T*  pointer,
                                    int memory_order,
                                    int thread_scope = __NV_THREAD_SCOPE_SYSTEM);


```

Atomic内置函数有以下限制：

 - 它们只能在设备函数中使用。

 - 它们不能在本地内存上操作。

 - 不能获取这些函数的地址。

- The `order` 和 `scope` 参数必须是整数文字；它们不能是变量。

 - `__NV_THREAD_SCOPE_CLUSTER` 及更高版本的体系结构支持线程作用域 `sm_90`。

 不支持的情况示例：

```cuda


     // Not permitted in a host function
     __host__ void bar() {
         unsigned u1 = 1, u2 = 2;
         __nv_atomic_load(&u1, &u2, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
     }

     // Not permitted to be applied to local memory
    __device__ void foo() {
       unsigned a = 1, b;
       __nv_atomic_load(&a, &b, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
    }

     // Not permitted as a template default argument.
     // The function address cannot be taken.
     template<void *F = __nv_atomic_load_n>
     class X {
         void *f = F; // The function address cannot be taken.
     };

     // Not permitted to be called in a constructor initialization list.
     class Y {
         int a;
     public:
         __device__ Y(int *b): a(__nv_atomic_load_n(b, __NV_ATOMIC_RELAXED)) {}
     };


```



#### 5.4.5.2.1. `__nv_atomic_fetch_add()`, `__nv_atomic_add()`（`__nv_atomic_fetch_add()`, `__nv_atomic_add()`）

```cuda


    __device__ T    __nv_atomic_fetch_add(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_add      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);


```

 这些函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 计算 `old`` ``+`` ``val`.

3. 将结果存储回内存中的同一地址。

- `__nv_atomic_fetch_add` 返回 `old` 值。

- `__nv_atomic_add`无返回值。

 该函数支持以下数据类型：

- `int`, `unsigned`, `unsigned`` ``long`` ``long`, `float`, `double`.





#### 5.4.5.2.2. `__nv_atomic_fetch_sub()`, `__nv_atomic_sub()`（`__nv_atomic_fetch_sub()`, `__nv_atomic_sub()`）

```cuda


    __device__ T    __nv_atomic_fetch_sub(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_sub      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);


```

 该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 计算 `old`` ``-`` ``val`.

3. 将结果存储回内存中的同一地址。

- `__nv_atomic_fetch_sub` 返回 `old` 值。

- `__nv_atomic_sub` 无返回值。

 该函数支持以下数据类型：

- `int`, `unsigned`, `unsigned`` ``long`` ``long`, `float`, `double`.





#### 5.4.5.2.3. `__nv_atomic_fetch_and()`, `__nv_atomic_and()`（`__nv_atomic_fetch_and()`, `__nv_atomic_and()`）

```cuda


    __device__ T    __nv_atomic_fetch_and(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_and      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);


```

 该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 计算 `old`` ``&`` ``val`.

3. 将结果存储回内存中的同一地址。

- `__nv_atomic_fetch_and` 返回 `old` 值。

- `__nv_atomic_and` 无返回值。

 该函数支持以下数据类型：

 - 大小为 4 或 8 字节的任何整型。





#### 5.4.5.2.4. `__nv_atomic_fetch_or()`, `__nv_atomic_or()`（`__nv_atomic_fetch_or()`, `__nv_atomic_or()`）

```cuda


    __device__ T    __nv_atomic_fetch_or(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_or      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);


```

 该函数以一个原子的方式执行以下操作交易：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 计算 `old`` ``|`` ``val`.

3. 将结果存储回内存中的同一地址。

- `__nv_atomic_fetch_or`返回 `old` 值。

- `__nv_atomic_or` 没有返回值。

 该函数支持以下数据类型：

 - 大小为 4 或 8 字节的任何整型。





#### 5.4.5.2.5. `__nv_atomic_fetch_xor()`, `__nv_atomic_xor()`（`__nv_atomic_fetch_xor()`, `__nv_atomic_xor()`）

```cuda


    __device__ T    __nv_atomic_fetch_xor(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_xor      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);


```

 该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 计算 `old`` ``^`` ``val`.

3. 将结果存储回内存中的同一地址。

- `__nv_atomic_fetch_xor` 返回 `old` 值。

- `__nv_atomic_xor` 无返回值。

 该函数支持以下数据类型：

 - 大小为 4 或 8 字节的任何整型。





#### 5.4.5.2.6. `__nv_atomic_fetch_min()`, `__nv_atomic_min()`（`__nv_atomic_fetch_min()`, `__nv_atomic_min()`）

```cuda


    __device__ T    __nv_atomic_fetch_min(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_min      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);


```

 该函数以一个原子的方式执行以下操作交易：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 计算 `old` 和 `val`.

3 中的最小值。  将结果存储回内存中的同一地址。

- `__nv_atomic_fetch_min` 返回 `old` 值。

- `__nv_atomic_min` 无返回值。

 该函数支持以下数据类型：

- `unsigned`, `int`, `unsigned`` ``long`` ``long`, `long`` ``long`.





#### 5.4.5.2.7. `__nv_atomic_fetch_max()`, `__nv_atomic_max()`（`__nv_atomic_fetch_max()`, `__nv_atomic_max()`）

```cuda


    __device__ T    __nv_atomic_fetch_max(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_max      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);


```

 该函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 计算 `old` 和 `val`.

3 的最大值。  将结果存储回内存中的同一地址。

- `__nv_atomic_fetch_max`返回 `old` 值。

- `__nv_atomic_max` 无返回值。

 该函数支持以下数据类型：

- `unsigned`, `int`, `unsigned`` ``long`` ``long`, `long`` ``long`





#### 5.4.5.2.8. `__nv_atomic_exchange()`, `__nv_atomic_exchange_n()`（`__nv_atomic_exchange()`, `__nv_atomic_exchange_n()`）

```cuda


    __device__ T    __nv_atomic_exchange_n(T* address, T val,          int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_exchange  (T* address, T* val, T* ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);


```

 该函数在一个原子事务中执行以下操作：

1. 读取全局或共享内存中位于地址 `old` 处的 `address` 值。

2.

    `__nv_atomic_exchange_n` 将 `val` 存储到 `address` 指向的位置。





    `__nv_atomic_exchange` 将 `old` 存储到 `ret` 指向的位置，并存储位于该地址的值`val` 到 `address` 指向的位置。



- `__nv_atomic_exchange_n` 返回 `old` 值。

- `__nv_atomic_exchange` 无返回值。

 该函数支持以下数据类型：

 - 大小为 4、8 或 16 的任何数据类型字节。

 - 计算能力 9.x 及更高版本的设备支持 16 字节数据类型。





#### 5.4.5.2.9. `__nv_atomic_compare_exchange()`, `__nv_atomic_compare_exchange_n()`（`__nv_atomic_compare_exchange()`, `__nv_atomic_compare_exchange_n()`）

```cuda


    __device__ bool __nv_atomic_compare_exchange  (T* address, T* expected, T* desired, bool weak, int success_order, int failure_order,
                                                   int scope = __NV_THREAD_SCOPE_SYSTEM);

    __device__ bool __nv_atomic_compare_exchange_n(T* address, T* expected, T desired, bool weak, int success_order, int failure_order,
                                                   int scope = __NV_THREAD_SCOPE_SYSTEM);


```

 这些函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中的地址 `old` 处的 `address` 值。

2. 将 `old` 与 `expected` 指向的值进行比较。

3. 如果它们相等，则返回值为 `true`，并且 `desired` 存储到 `address` 指向的位置。否则，它返回 `false`，并且 `old` 存储到 `expected` 指向的位置。

 参数 `weak` 被忽略，它会选择之间更强的内存顺序`success_order`和 `failure_order` 执行比较和交换操作。

 该函数支持以下数据类型：

 - 大小为 2、4、8 或 16 字节的任何数据类型。

 - 具有计算能力 9.x 及更高版本的设备支持 16 字节数据类型。





#### 5.4.5.2.10. `__nv_atomic_load()`, `__nv_atomic_load_n()`（`__nv_atomic_load()`, `__nv_atomic_load_n()`）

```cuda


    __device__ void __nv_atomic_load  (T* address, T* ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ T    __nv_atomic_load_n(T* address,         int order, int scope = __NV_THREAD_SCOPE_SYSTEM);


```

 该函数执行一个原子事务中的以下操作：

1. 读取位于全局或共享内存中 `old` 地址处的 `address` 值。

2.

    `__nv_atomic_load` 将 `old` 存储到 `ret` 指向的位置。





    `__nv_atomic_load_n` 返回 `old`.



 该函数支持以下数据类型：

 - 任何大小的数据类型1、2、4、8 或 16 字节。

`order` 不能是 `__NV_ATOMIC_RELEASE` or `__NV_ATOMIC_ACQ_REL`.





#### 5.4.5.2.11. `__nv_atomic_store()`, `__nv_atomic_store_n()`（`__nv_atomic_store()`, `__nv_atomic_store_n()`）

```cuda


    __device__ void __nv_atomic_store  (T* address, T* val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_store_n(T* address, T  val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);


```

 函数在一个原子事务中执行以下操作：

1. 读取全局或共享内存中位于地址 `old` 处的 `address` 值。

2.

    `__nv_atomic_store` 读取 `val` 指向的值并存储到 `address` 指向的位置。





    `__nv_atomic_store_n` 将 `val` 存储到 `address` 指向的位置to.



`order` 不能是 `__NV_ATOMIC_CONSUME`, `__NV_ATOMIC_ACQUIRE` or `__NV_ATOMIC_ACQ_REL`.





#### 5.4.5.2.12. `__nv_atomic_thread_fence()`（`__nv_atomic_thread_fence()`）

```cuda


    __device__ void __nv_atomic_thread_fence(int order, int scope = __NV_THREAD_SCOPE_SYSTEM);


```

 该原子函数根据指定的内存顺序在此线程请求的内存访问之间建立排序。线程范围参数指定可以观察此操作的排序效果的线程集。









<span id="id13"></span>

## 5.4.6. Warp 函数（Warp Functions）

以下部分介绍了允许扭曲中的线程相互通信并执行计算的扭曲函数。



Hint

建议尽可能使用 `CUB` <a href="https://nvidia.github.io/cccl/unstable/cub/api_docs/warp_wide.html#warp-wide-collective-primitives" class="reference external">Warp-Wide“集体”基元</a> 来执行扭曲操作，以提高效率、安全性和可移植性





<span id="warp-activemask-functions"></span>

### 5.4.6.1. Warp Active Mask（Warp Active Mask）

```cuda


    unsigned __activemask();


```

函数返回一个32位整数掩码，表示调用warp中所有当前活动的线程。如果调用 `__activemask()` 时扭曲中的第 N 个通道处于活动状态，则设置第 N 位。 <a href="../03-advanced-cuda/advanced-kernel-programming.html#simt-architecture-notes" class="reference internal">非活动线程</a> 在返回的掩码中由 0 位表示。已退出程序的线程始终标记为非活动状态。



Warning

`__activemask()` 不能用于确定哪些扭曲通道执行给定分支。此函数旨在用于机会性扭曲级编程，并且仅提供扭曲内活动线程的瞬时快照。

```cuda


    // Check whether at least one thread's predicate evaluates to true
    if (pred) {
        // Invalid: the value of 'at_least_one' is non-deterministic
        // and could vary between executions.
        at_least_one = __activemask() > 0;
    }


```



请注意，在 `__activemask()` 调用处收敛的线程不能保证在后续指令处保持收敛，除非这些指令是扭曲同步内在函数（`__sync`).

例如，编译器可以重新排序指令，并且活动线程集可能不会保留：

```cuda


    unsigned mask      = __activemask();              // Assume mask == 0xFFFFFFFF (all bits set, all threads active)
    int      predicate = threadIdx.x % 2 == 0;        // 1 for even threads, 0 for odd threads
    int      result    = __any_sync(mask, predicate); // Active threads might not be preserved


```





<span id="id14"></span>

### 5.4.6.2. Warp 投票函数（Warp Vote Functions）

```cuda


    int      __all_sync   (unsigned mask, int predicate);
    int      __any_sync   (unsigned mask, int predicate);
    unsigned __ballot_sync(unsigned mask, int predicate);


```

 warp 投票函数使给定 <a href="../01-introduction/programming-model.html#programming-model-warps-simt" class="reference internal">warp</a> 的线程执行缩减和广播操作。这些函数将整数 `predicate` 作为来自 warp 中每个未退出线程的输入，并将这些值与零进行比较，然后组合比较结果。通过以下方式之一跨 <a href="../03-advanced-cuda/advanced-kernel-programming.html#simt-architecture-notes" class="reference internal"> 活动线程 </a> 进行（减少），向每个参与线程广播单个返回值：

`__all_sync(unsigned`` ``mask,`` ``predicate)`:
E 为 `predicate` 中的所有非退出线程评估 `mask`，如果 `predicate` 的所有线程评估为非零，则返回非零他们。

`__any_sync(unsigned`` ``mask,`` ``predicate)`:
E 评估`predicate`对于所有未退出的线程`mask`并返回非零如果`predicate`其中一个或多个的计算结果为非零。

`__ballot_sync(unsigned`` ``mask,`` ``predicate)`:
评价`predicate`对于所有未退出的线程`mask`并返回一个整数，其第 N 位被设置，如果`predicate`对于 warp 的第 N 个线程，计算结果为非零，并且第 N 个线程处于活动状态。否则，第 N 位为零。

这些功能受<a href="#warp-sync-intrinsic-constraints" class="reference internal">Warp __sync 内在约束</a>.



警告

这些内在函数不提供任何内存排序。







<span id="id15"></span>

### 5.4.6.3. 扭曲匹配函数（Warp Match Functions）



暗示

建议使用<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/warp/warp_match_all.html" class="reference external">库++</a> `cuda::device::warp_match_all()`作为通用且更安全的替代品`__match_all_sync`功能。



```cuda


    unsigned __match_any_sync(unsigned mask, T value);
    unsigned __match_all_sync(unsigned mask, T value, int *pred);


```

warp 匹配函数在一个线程内的非退出线程之间执行变量的广播和比较操作。<a href="../01-introduction/programming-model.html#programming-model-warps-simt" class="reference internal">经</a>.

`__match_any_sync`
返回具有相同按位的未退出线程的掩码`value` in `mask`.

`__match_all_sync`
退货`mask`如果所有未退出的线程都在`mask`按位相同`value`;否则返回 0。谓词`pred`设置为`true`如果所有未退出的线程都在`mask`按位相同`value`;否则谓词设置为 false。

`T`可以是`int`, `unsigned`, `long`, `unsigned`` ``long`, `long`` ``long`, `unsigned`` ``long`` ``long`, `float` or `double`.

这些功能受<a href="#warp-sync-intrinsic-constraints" class="reference internal">Warp __sync 内在约束</a>.



警告

这些内在函数不提供任何内存排序。







<span id="id16"></span>

### 5.4.6.4. 减少变形功能（Warp Reduce Functions）



暗示

建议使用`CUB` <a href="https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1WarpReduce.html#_CPPv4I0_iEN3cub10WarpReduceE" class="reference external">曲速范围“集体”基元</a>出于效率、安全性和便携性的原因，尽可能执行扭曲减少。



受计算能力 8.x 或更高版本的设备支持。

```cuda


    T        __reduce_add_sync(unsigned mask, T value);
    T        __reduce_min_sync(unsigned mask, T value);
    T        __reduce_max_sync(unsigned mask, T value);

    unsigned __reduce_and_sync(unsigned mask, unsigned value);
    unsigned __reduce_or_sync (unsigned mask, unsigned value);
    unsigned __reduce_xor_sync(unsigned mask, unsigned value);


```

The `__reduce_<op>_sync`内在函数对中提供的数据执行归约操作`value`同步所有未退出的线程后`mask`.

`__reduce_add_sync`, `__reduce_min_sync`, `__reduce_max_sync`
返回对中提供的值应用算术加法、最小或最大归约运算的结果`value`由每个未退出的线程指定`mask`. `T`可以是一个`unsigned` or `signed`整数。

`__reduce_and_sync`, `__reduce_or_sync`, `__reduce_xor_sync`
返回对中提供的值应用按位 AND、OR 或 XOR 归约运算的结果`value`由每个未退出的线程指定`mask`.

这些功能受<a href="#warp-sync-intrinsic-constraints" class="reference internal">Warp __sync 内在约束</a>.



警告

这些内在函数不提供任何内存排序。







<span id="id17"></span>

### 5.4.6.5. 扭曲随机播放功能（Warp Shuffle Functions）



暗示

建议使用<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/warp/warp_shuffle.html#libcudacxx-extended-api-warp-warp-shuffle" class="reference external">库++</a> `cuda::device::warp_shuffle()`作为通用且更安全的替代品`__shfl_sync()`和`__shfl_<op>_sync()`内在函数。



```cuda


    T __shfl_sync     (unsigned mask, T value, int      srcLane,  int width=warpSize);
    T __shfl_up_sync  (unsigned mask, T value, unsigned delta,    int width=warpSize);
    T __shfl_down_sync(unsigned mask, T value, unsigned delta,    int width=warpSize);
    T __shfl_xor_sync (unsigned mask, T value, int      laneMask, int width=warpSize);


```

Warp shuffle 函数在未退出的线程之间交换值<a href="../01-introduction/programming-model.html#programming-model-warps-simt" class="reference internal">经</a>不使用共享内存。

`__shfl_sync()`：从索引通道直接复制。
内在函数返回的值`value`由其ID由以下给出的线程持有`srcLane`.

- If `width`小于`warpSize`，那么扭曲的每个子部分都表现为一个单独的实体，其起始逻辑通道 ID 为 0。

- If `srcLane`超出范围`[0,`` ``width`` ``-`` ``1]`，结果对应于所持有的值`srcLane`` ``%`` ``width`，它位于同一小节内。

------------------------------------------------------------------------

`__shfl_up_sync()`：从 ID 低于调用者 ID 的通道复制。
内在函数通过减去来计算源通道 ID`delta`来自呼叫者的车道 ID。的价值`value`返回结果车道 ID 所持有的内容：实际上，`value`经线向上移动`delta`车道。

- If `width`小于 `warpSize`，则扭曲的每个子部分都表现为起始逻辑通道 ID 为 0 的单独实体。

 - 源通道索引不会环绕 `width` 的值，因此较低的 `delta` 通道将保持不变。\

------------------------------------------------------------------------

`__shfl_down_sync()`：从 ID 高于该通道的通道进行复制来电者”。
 内部函数通过将 `delta` 添加到调用者的通道 ID 来计算源通道 ID。返回结果通道 ID 所保存的 `value` 的值：这具有将 `value` 在扭曲中向下移动 `delta` 通道的效果。

- If `width` 小于 `warpSize`，则扭曲的每个分段都表现为一个单独的实体，其起始逻辑通道 ID 为 0。

- 至于`__shfl_up_sync()`，源通道的 ID 号不会环绕宽度值，因此上面的 `delta` 通道将有效保持不变。\

------------------------------------------------------------------------

`__shfl_xor_sync()`：根据自己通道 ID 的按位异或从通道复制。
 该内在函数通过对调用者的通道 ID 和 `laneMask` 执行按位异或来计算源通道 ID：返回结果通道 ID 所保存的 `value` 的值。该模式实现了蝶形寻址模式，用于树缩减和广播。

- If `width` 小于 `warpSize`，则每组 `width` 连续线程都能够访问较早组中的元素。但是，如果它们尝试访问后面的线程组中的元素，则将返回它们自己的 `value` 值。

------------------------------------------------------------------------

`T` 可以是：包含

- `int`, `unsigned`, `long`, `unsigned`` ``long`, `long`` ``long`, `unsigned`` ``long`` ``long`, `float` or `double`.

- `__half` 标头的 `__half2` 和 `cuda_fp16.h`。包含

- `__nv_bfloat16` 标头的 `__nv_bfloat162` 和 `cuda_bf16.h` include.

Threads 只能从主动参与内在函数的另一个线程读取数据。如果目标线程是 <a href="../03-advanced-cuda/advanced-kernel-programming.html#simt-architecture-notes" class="reference internal">inactive</a>，则检索到的值未定义。

`width` 必须是 `[1,`` ``warpSize]` 范围内的 2 的幂，即 1、2、4、8、16 或 32。其他值将产生未定义的结果。

这些函数受 <a href="#warp-sync-intrinsic-constraints" class="reference internal">Warp __sync 内在约束</a>.

有效 warp shuffle 使用示例：

```cuda


    int laneId = threadIdx.x % warpSize;
    int data   = ...

    // all warp threads get 'data' from lane 0
    int result1 = __shfl_sync(0xFFFFFFFF, data, 0);

    if (laneId < 4) {
        // lanes 0, 1, 2, 3 get 'data' from lane 1
        int result2 = __shfl_sync(0xb1111, data, 1);
    }

    // lanes [0 - 15] get 'data' from lane 0
    // lanes [16 - 31] get 'data' from lane 16
    int result3 = __shfl_sync(0xFFFFFFFF, value, warpSize / 2);

    // each lane gets 'data' from the lane two positions above
    // lanes 30, 31 get their original value
    int result4 = __shfl_down_sync(0xFFFFFFFF, data, 2);


```

 无效 warp shuffle 使用示例：

```cuda


    int laneId = threadIdx.x % warpSize;
    int value  = ...
     // undefined behavior: lane 0 does not participate in the call
    int result = (laneId > 0) ? __shfl_sync(0xFFFFFFFF, value, 0) : 0;

    if (laneId <= 4) {
        // undefined behavior: destination lanes 5, 6 are not active for lanes 3, 4
        result = __shfl_down_sync(0b11111, value, 2);
    }

    // undefined behavior: width is not a power of 2
    __shfl_sync(0xFFFFFFFF, value, 0, /*width=*/31);


```



Warning

 这些内在函数并不意味着内存屏障。它们不保证任何内存排序。



示例 1：跨 warp 广播单个值



CUDA C++



```cuda


    #include <cassert>
    #include <cuda/warp>

    __global__ void warp_broadcast_kernel(int input) {
        int laneId = threadIdx.x % 32;
        int value;
        if (laneId == 0) { // unused variable for all threads except lane 0
            value = input;
        }
        value = cuda::device::warp_shuffle_idx(value, 0); // Synchronize all threads in warp, and get "value" from lane 0
        assert(value == input);
    }

    int main() {
        warp_broadcast_kernel<<<1, 32>>>(1234);
        cudaDeviceSynchronize();
        return 0;
    }


```



Intrinsics



```cuda


    #include <assert.h>

    __global__ void warp_broadcast_kernel(int input) {
        int laneId = threadIdx.x % 32;
        int value;
        if (laneId == 0) { // unused variable for all threads except lane 0
            value = input;
        }
        value = __shfl_sync(0xFFFFFFFF, value, 0); // Synchronize all threads in warp, and get "value" from lane 0
        assert(value == input);
    }

    int main() {
        warp_broadcast_kernel<<<1, 32>>>(1234);
        cudaDeviceSynchronize();
        return 0;
    }


```





请参阅 <a href="https://cuda.godbolt.org/z/E3E3Y5e4e" class="reference external">Compiler Explorer 上的示例</a>.

示例 2：跨 8 个子分区的包含加扫描线程



Hint

建议使用<a href="https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1WarpScan.html" class="reference external">cub::WarpScan</a>函数来实现高效且通用的warp扫描函数。





CUDA C++



```cuda


    #include <cstdio>
    #include <cub/cub.cuh>

    __global__ void scan_sub_partition_with_8_threads_kernel() {
        using WarpScan    = cub::WarpScan<int, 8>;
        using TempStorage = typename WarpScan::TempStorage;
        __shared__ TempStorage temp_storage;

        int laneId = threadIdx.x % 32;
        int value  = 31 - laneId; // starting value to accumulate
        int partial_sum;
        WarpScan(temp_storage).InclusiveSum(value, partial_sum);
        printf("Thread %d final value = %d\n", threadIdx.x, partial_sum);
    }

    int main() {
        scan_sub_partition_with_8_threads_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        return 0;
    }


```



Intrinsics



```cuda


    #include <stdio.h>

    __global__ void scan_sub_partition_with_8_threads_kernel() {
        int laneId = threadIdx.x % 32;
        int value  = 31 - laneId; // starting value to accumulate
        // Loop to accumulate scan within my partition.
        // Scan requires log2(8) == 3 steps for 8 threads
        for (int delta = 1; delta <= 4; delta *= 2) {
            int tmp         = __shfl_up_sync(0xFFFFFFFF, value, delta, /*width=*/8); // read from laneId - delta
            int source_lane = laneId % 8 - delta;
            if (source_lane >= 0) // lanes with 'source_lane < 0' have their value unchanged
                value += tmp;
        }
        printf("Thread %d final value = %d\n", threadIdx.x, value);
    }

    int main() {
        scan_sub_partition_with_8_threads_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        return 0;
    }


```





请参阅<a href="https://cuda.godbolt.org/z/Tohd38edc" class="reference external">Compiler上的示例Explorer</a>.

示例 3：跨扭曲减少



Hint

建议使用 <a href="https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1WarpReduce.html" class="reference external">cub::WarpReduce</a> 函数来实现高效且通用的扭曲减少函数。





CUDA C++



```cuda


    #include <cstdio>
    #include <cub/cub.cuh>
    #include <cuda/warp>

    __global__ void warp_reduce_kernel() {
        using WarpReduce  = cub::WarpReduce<int>;
        using TempStorage = typename WarpReduce::TempStorage;
        __shared__ TempStorage temp_storage;

        int laneId     = threadIdx.x % 32;
        int value      = 31 - laneId; // starting value to accumulate
        auto aggregate = WarpReduce(temp_storage).Sum(value);
        aggregate      = cuda::device::warp_shuffle_idx(aggregate, 0);
        printf("Thread %d final value = %d\n", threadIdx.x, aggregate);
    }

    int main() {
        warp_reduce_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        return 0;
    }


```



Intrinsics



```cuda


    #include <stdio.h>

    __global__ void warp_reduce_kernel() {
        int laneId = threadIdx.x % 32;
        int value  = 31 - laneId; // starting value to accumulate
        // Use XOR mode to perform butterfly reduction
        // A full-warp reduction requires log2(32) == 5 steps
        for (int i = 1; i <= 16; i *= 2)
            value += __shfl_xor_sync(0xFFFFFFFF, value, i);
        // "value" now contains the sum across all threads
        printf("Thread %d final value = %d\n", threadIdx.x, value);
    }

    int main() {
        warp_reduce_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        return 0;
    }


```





请参阅示例<a href="https://cuda.godbolt.org/z/T94nfGMzG" class="reference external">Compiler Explorer</a>.





<span id="id18"></span>

### 5.4.6.6. Warp `__sync` 内在约束（Warp `__sync` Intrinsic Constraints）

所有 warp `__sync` 内在函数，例如：

- `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`

- `__match_any_sync`, `__match_all_sync`

- `__reduce_add_sync`, `__reduce_min_sync`, `__reduce_max_sync`, `__reduce_and_sync`, `__reduce_or_sync`, `__reduce_xor_sync`

- `__syncwarp`

 使用 `mask` 参数来指示哪些 warp 线程参与调用。该参数确保硬件执行内在函数之前正确收敛。



 `mask` 中的每个位对应于线程的通道 ID (`threadIdx.x`` ``%`` ``warpSize`)。内在函数会等待，直到 `mask` 中指定的所有未退出的扭曲线程到达调用。





 必须满足以下约束才能正确执行：



- 每个调用线程必须在`mask`.

- 每个非调用线程必须将其相应位设置为零`mask`。退出的线程将被忽略。

- 中指定的所有未退出的线程`mask`必须以相同的方式执行内在函数`mask`价值。

- Warp 线程可能会同时调用不同的内在函数`mask`值，前提是掩码不相交。即使在发散的控制流中，这种条件也是有效的。

扭曲行为`__sync`函数无效，例如内核挂起或未定义，如果：

- 调用线程中未指定`mask`.

- 指定的非退出线程`mask`无法最终退出或在同一程序点使用相同的内容调用内在函数`mask`价值。

- 在条件代码中，所有条件必须在指定的所有非退出线程中进行相同的评估`mask`.



笔记

当所有 warp 线程都参与调用时，即当`mask`设置为`0xFFFFFFFF`.



有效的扭曲内在函数用法示例：

```cuda


    __global__ void valid_examples() {
        if (threadIdx.x < 4) {        // threads 0, 1, 2, 3 are active
            __all_sync(0b1111, pred); // CORRECT, threads 0, 1, 2, 3 participate in the call
        }

        if (threadIdx.x == 0)
            return; // exit
        // CORRECT, all non-exited threads participate in the call
        __all_sync(0xFFFFFFFF, pred);
    }


```

不相交`mask`例子：

```cuda


    __global__ void example_syncwarp_with_mask(int* input_data, int* output_data) {
        if (threadIdx.x < warpSize) {
            __shared__ int shared_data[warpSize];
            shared_data[threadIdx.x] = input_data[threadIdx.x];

            unsigned mask = threadIdx.x < 16 ? 0xFFFF : 0xFFFF0000; // CORRECT
            __syncwarp(mask);
            if (threadIdx.x == 0 || threadIdx.x == 16)
                output_data[threadIdx.x] = shared_data[threadIdx.x + 1];
        }
    }


```

```cuda


    __global__ void example_syncwarp_with_mask_branches(int* input_data, int* output_data) {
        if (threadIdx.x < warpSize) {
            __shared__ int shared_data[warpSize];
            shared_data[threadIdx.x] = input_data[threadIdx.x];

            if (threadIdx.x < 16) {
                unsigned mask = 0xFFFF; // CORRECT
                __syncwarp(mask);
                output_data[threadIdx.x] = shared_data[15 - threadIdx.x];
            }
            else {
                unsigned mask = 0xFFFF0000; // CORRECT
                __syncwarp(mask);
                output_data[threadIdx.x] = shared_data[31 - threadIdx.x];
            }
        }
    }


```

无效的 warp 内在函数用法示例：

```cuda


    if (threadIdx.x < 4) {           // threads 0, 1, 2, 3 are active
        __all_sync(0b0000011, pred); // WRONG, threads 2, 3 are active but not set in mask
        __all_sync(0b1111111, pred); // WRONG, threads 4, 5, 6 are not active but set in mask
    }

    // WRONG, participating threads have a different and overlapping mask
    __all_sync(threadIdx.x == 0 ? 1 : 0xFFFFFFFF, pred);


```







## 5.4.7. CUDA 特定宏（CUDA-Specific Macros）



<span id="cuda-arch-macro"></span>

### 5.4.7.1. `__CUDA_ARCH__`（`__CUDA_ARCH__`）

宏观`__CUDA_ARCH__`代表<a href="https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architecture-macros" class="reference external">虚拟架构</a>正在为其编译代码的 NVIDIA GPU 的名称。其值可能与设备的实际计算能力不同。该宏允许编写专门针对特定 GPU 架构的代码路径，这可能是获得最佳性能或使用特定于架构的功能和指令所必需的。该宏还可用于区分主机代码和设备代码。

`__CUDA_ARCH__`仅在设备代码中定义，即在`__device__`, `__host__`` ``__device__`， 和`__global__`功能。宏的值与`nvcc`选项`compute_<version>`，与关系`__CUDA_ARCH__`` ``=`` ``<version>`` ``*`` ``10`.

例子：

```bash


    nvcc --generate-code arch=compute_80,code=sm_90 prog.cu


```

定义`__CUDA_ARCH__` as `800`.

------------------------------------------------------------------------

`__CUDA_ARCH__`**约束**

**1.** 以下实体的类型签名不应取决于是否定义了 `__CUDA_ARCH__`，也不取决于其值。

- `__global__` 函数和函数模板。

- `__device__` 和 `__constant__` 变量。

- 纹理和Surfaces.

示例：

```cuda


    #if !defined(__CUDA_ARCH__)
        typedef int my_type;
    #else
        typedef double my_type;
    #endif

    __device__ my_type my_var;           // ERROR: my_var's type depends on __CUDA_ARCH__

    __global__ void kernel(my_type in) { // ERROR: kernel's type depends on __CUDA_ARCH__
        ...
    }


```

**2.** `__global__` 函数模板的实例化不得依赖于 `__CUDA_ARCH__` 是否已定义或其值。也就是说，具有等效模板参数的相同实例化必须出现在*所有*设备程序和主机程序中（无论这些实例化中的任何一个是否在运行时启动）。

E示例：

```cuda


    __device__ int result;

    template <typename T>
    __global__ void kernel(T in) {
        result = in;
    }

    __host__ __device__ void host_device_function(void) {
    #if !defined(__CUDA_ARCH__)
        kernel<<<1, 1>>>(1); // ERROR: "kernel<int>" instantiation only
                                //        when __CUDA_ARCH__ is undefined!
    #endif
    }

    int main(void) {
        host_device_function();
        cudaDeviceSynchronize();
        return 0;
    }


```

可以通过将计算从 `__global__` 函数移动到由前者调用的 `__device__` 函数模板来避免此问题。在 `__global__` 函数内，`__CUDA_ARCH__` 可用于有条件地实例化具有不同参数的 `__device__` 函数模板。

**3.** 在单独编译模式下，具有外部链接的函数或变量定义的存在或不存在不应取决于 `__CUDA_ARCH__` 的定义或其value.

示例：

```cuda


    #if !defined(__CUDA_ARCH__)
        void host_function(void) {} // ERROR: The definition of host_function()
                                    //        is only present when __CUDA_ARCH__
                                    //        is undefined
    #endif


```

**4.** 在单独编译中，不得在标头中使用预处理器宏 `__CUDA_ARCH__`，以防止对象具有不同的行为。或者，所有对象必须针对同一虚拟架构进行编译。如果在标头中定义了弱函数或模板函数，并且其行为取决于 `__CUDA_ARCH__`，则如果针对不同的计算体系结构编译这些对象，则不同对象中的该函数的实例可能会发生冲突。

例如，如果标头文件 `a.h` 包含：

```cuda


    template<typename T>
    __device__ T* get_ptr() {
    #if __CUDA_ARCH__ == 900
        return nullptr; /* no address */
    #else
        __shared__ T arr[256];
        return arr;
    #endif
    }


```

，则如果 `a.cu` 和 `b.cu` 都包含`a.h` 和实例化 `get_ptr()` 为相同类型，并且 `b.cu` 需要非 `NULL` 地址，并使用以下命令进行编译：

```text


    nvcc -arch=compute_70 -dc a.cu
    nvcc -arch=compute_80 -dc b.cu
    nvcc -arch=sm_80 a.o b.o

    Only one version of the ``get_ptr()`` function is used at link time, so the behavior depends on which version is chosen. To avoid this issue, either ``a.cu`` and ``b.cu`` must be compiled for the same compute architecture, or ``__CUDA_ARCH__`` should not be used in the shared header function.


```

 编译器不保证将为上述 `__CUDA_ARCH__` 的不支持用途生成诊断。





### 5.4.7.2. `__CUDA_ARCH_SPECIFIC__` 和`__CUDA_ARCH_FAMILY_SPECIFIC__`（`__CUDA_ARCH_SPECIFIC__` and `__CUDA_ARCH_FAMILY_SPECIFIC__`）

宏`__CUDA_ARCH_SPECIFIC__`和`__CUDA_ARCH_FAMILY_SPECIFIC__`定义为识别具有<a href="compute-capabilities.html#compute-capabilities-architecture-specific-features" class="reference internal">架构-</a>和<a href="compute-capabilities.html#compute-capabilities-family-specific-features" class="reference internal">家庭-</a>分别具有具体特征。看<a href="compute-capabilities.html#compute-capabilities-feature-set-compiler-targets" class="reference internal">功能集编译器目标</a>部分了解更多信息。

类似于`__CUDA_ARCH__`, `__CUDA_ARCH_SPECIFIC__`和`__CUDA_ARCH_FAMILY_SPECIFIC__`仅在设备代码中定义，即在`__device__`, `__host__`` ``__device__`， 和`__global__`功能。这些宏与`nvcc`选项`compute_<version>a`和`compute_<version>f`.

```bash


    nvcc --generate-code arch=compute_100a,code=sm_100a prog.cu


```

- `__CUDA_ARCH__`` ``==`` ``1000`.

- `__CUDA_ARCH_SPECIFIC__`` ``==`` ``1000`.

- `__CUDA_ARCH_FAMILY_SPECIFIC__`` ``==`` ``1000`.

```bash


    nvcc --generate-code arch=compute_100f,code=sm_103f prog.cu


```

- `__CUDA_ARCH__`` ``==`` ``1000`.

- `__CUDA_ARCH_FAMILY_SPECIFIC__`` ``==`` ``1000`.

- `__CUDA_ARCH_SPECIFIC__`没有定义。

```bash


    nvcc -arch=sm_100 prog.cu


```

- `__CUDA_ARCH__`` ``==`` ``1000`.

- `__CUDA_ARCH_FAMILY_SPECIFIC__`没有定义。

- `__CUDA_ARCH_SPECIFIC__`没有定义。

```bash


    nvcc -arch=sm_100a prog.cu
    # equivalent to:
    nvcc --generate-code arch=sm_100a,compute_100,compute_100a prog.cu


```

- `__CUDA_ARCH__`` ``==`` ``1000`.

- `__CUDA_ARCH_FAMILY_SPECIFIC__`没有定义。

- `__CUDA_ARCH_SPECIFIC__`` ``==`` ``1000`和`__CUDA_ARCH_SPECIFIC__`未定义的都生成。





### 5.4.7.3. CUDA 功能测试宏（CUDA Feature Testing Macros）

`nvcc`提供以下预处理器宏用于功能测试。当 CUDA 前端编译器支持特定功能时定义宏。

- `__CUDACC_DEVICE_ATOMIC_BUILTINS__`：支持<a href="#built-in-atomic-functions" class="reference internal">设备原子编译器内置</a>.

- `__NVCC_DIAG_PRAGMA_SUPPORT__`：支持<a href="#nv-diagnostic-pragmas" class="reference internal">诊断控制指令</a>.

- `__CUDACC_EXTENDED_LAMBDA__`：支持<a href="cpp-language-support.html#extended-lambdas" class="reference internal">扩展 lambda</a>。启用者`--expt-extended-lambda` or `--extended-lambda`旗帜。

- `__CUDACC_RELAXED_CONSTEXPR__`：支持<a href="cpp-language-support.html#constexpr-functions" class="reference internal">宽松的 constexpr 函数</a>。由 `--expt-relaxed-constexpr` 标志启用。





### 5.4.7.4. `__nv_pure__` 属性（`__nv_pure__` Attribute）

在 C/C++ 中，纯函数对其参数没有副作用，并且可以访问全局变量，但不会修改它们。

CUDA 提供了主机和设备函数都支持的 `__nv_pure__` 属性。编译器将 `__nv_pure__` 转换为 `pure` GNU 属性或 Microsoft Visual Studio `noalias` 属性。

```cuda


    __device__ __nv_pure__
    int add(int a, int b) {
        return a + b;
    }


```







## 5.4.8. CUDA 特定函数（CUDA-Specific Functions）



### 5.4.8.1. 地址空间谓词函数（Address Space Predicate Functions）

地址空间谓词函数用于确定指针。



Hint

 建议使用 `cuda::device::is_address_from()`libcu++`cuda::device::is_object_from()` 提供的 <a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory/is_address_from.html" class="reference external"> 和 </a> 函数作为地址空间谓词内在函数的可移植且更安全的替代方案。



```cuda


    __device__ unsigned __isGlobal      (const void* ptr);
    __device__ unsigned __isShared      (const void* ptr);
    __device__ unsigned __isConstant    (const void* ptr);
    __device__ unsigned __isGridConstant(const void* ptr);
    __device__ unsigned __isLocal       (const void* ptr);


```

 函数返回 `1` if `ptr` 包含的通用地址指定地址空间中的对象，否则为 `0`。如果参数是 `NULL` 指针，则它们的行为未指定。

- `__isGlobal()`：全局内存空间。

- `__isShared()`：共享内存空间。

- `__isConstant()`：常量内存空间。

- `__isGridConstant()`：用 `__grid_constant__`.

- `__isLocal()` 注释的内核参数：本地内存空间。





### 5.4.8.2. 地址空间转换函数（Address Space Conversion Functions）

CUDA 指针(`T*`) 可以访问对象，无论对象存储在何处。例如，`int*` 可以访问 `int` 对象，无论它们驻留在全局内存还是共享内存中。

 地址空间转换函数用于在通用地址和特定地址空间中的地址之间进行转换。当编译器无法确定指针的地址空间时，例如，在跨越翻译单元或与 PTX 指令交互时，这些函数非常有用。

```cuda


    __device__ size_t __cvta_generic_to_global  (const void* ptr); // PTX: cvta.to.global
    __device__ size_t __cvta_generic_to_shared  (const void* ptr); // PTX: cvta.to.shared
    __device__ size_t __cvta_generic_to_constant(const void* ptr); // PTX: cvta.to.const
    __device__ size_t __cvta_generic_to_local   (const void* ptr); // PTX: cvta.to.local


```

```cuda


    __device__ void* __cvta_global_to_generic  (size_t raw_ptr); // PTX: cvta.global
    __device__ void* __cvta_shared_to_generic  (size_t raw_ptr); // PTX: cvta.shared
    __device__ void* __cvta_constant_to_generic(size_t raw_ptr); // PTX: cvta.const
    __device__ void* __cvta_local_to_generic   (size_t raw_ptr); // PTX: cvta.local


```

 作为与 PTX 指令互操作的示例，`ld.shared.s32`` ``r0,`` ``[ptr];` PTX 指令期望 `ptr` 引用共享内存地址空间。带有指向对象的 `int*` 指针的 CUDA 程序`__shared__`内存需要将此指针转换为共享地址空间，然后通过调用 `__cvta_generic_to_shared` 将其传递到 PTX 指令，如下所示：

```cuda


    __shared__ int smem_var;
    smem_var        = 42;
    size_t smem_ptr = __cvta_generic_to_shared(&smem_var);
    int    output;
    asm volatile("ld.shared.s32 %0, [%1];" : "=r"(output) : "l"(smem_ptr) : "memory");
    assert(output == 42);


```

 利用这些地址表示的常见优化是通过利用共享空间、本地空间和常量空间的地址范围小于 32 位的事实来减小数据结构大小，这允许存储 32 位地址而不是 64 位指针并保存寄存器。此外，32 位算术比 64 位算术更快。要获取这些地址的 32 位整数表示形式，请通过从无符号 64 位整数转换为无符号 32 位整数来将 64 位值截断为 32 位：

```cuda


    __shared__ int smem_var;
    uint32_t       smem_ptr_32bit = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_var));


```

 要从此类 32 位表示形式恢复通用地址，请将地址零扩展回无符号 64 位整数，然后调用相应的地址空间转换函数：

```cuda


    size_t smem_ptr_64bit = static_cast<size_t>(smem_ptr_32bit); // zero-extend to 64 bits
    void*  generic_ptr    = __cvta_shared_to_generic(smem_ptr_64bit);
    assert(generic_ptr == &smem_var);


```

------------------------------------------------------------------------





<span id="low-level-load-store-functions"></span>

### 5.4.8.3. 低级加载和存储函数（Low-Level Load and Store Functions）

```cuda


    T __ldg(const T* address);


```

函数`__ldg()`执行只读L1/Tex缓存加载。它支持所有 C++ 基本类型、CUDA 向量类型（x3 组件除外）和扩展浮点类型，例如 `__half`, `__half2`, `__nv_bfloat16` 和 `__nv_bfloat162`.

------------------------------------------------------------------------

```cuda


    T __ldcg(const T* address);
    T __ldca(const T* address);
    T __ldcs(const T* address);
    T __ldlu(const T* address);
    T __ldcv(const T* address);


```

 函数使用 <a href="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators" class="reference external">PTX ISA</a> 指南中指定的缓存运算符执行加载。它们支持所有 C++ 基本类型、CUDA 向量类型（x3 组件除外）和扩展浮点类型，例如 `__half`, `__half2`, `__nv_bfloat16` 和 `__nv_bfloat162`.

------------------------------------------------------------------------

```cuda


    void __stwb(T* address, T value);
    void __stcg(T* address, T value);
    void __stcs(T* address, T value);
    void __stwt(T* address, T value);


```

 函数使用 <a href="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators" class="reference external">PTX ISA</a> 指南中指定的缓存运算符执行存储。它们支持所有 C++ 基本类型、CUDA 向量类型（x3 分量除外）和扩展浮点类型，例如 `__half`, `__half2`, `__nv_bfloat16` 和 `__nv_bfloat162`.





<span id="trap-function"></span>

### 5.4.8.4. `__trap()`（`__trap()`）



Hint

 建议使用 `cuda::std::terminate()`libcu++<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/standard_api.html" class="reference external">C++ 参考提供的 </a> (<a href="https://en.cppreference.com/w/cpp/error/terminate.html" class="reference external"> 函数</a>）作为 `__trap()`.



A 的可移植替代方案，可以通过从任何设备线程调用 `__trap()` 函数来启动陷阱操作。

```cuda


    void __trap();


```

 内核的执行被中止，在主机程序中引发中断。调用 `__trap()` 会导致 CUDA 上下文损坏，从而导致后续 CUDA 调用和内核调用失败。





<span id="nanosleep-function"></span>

### 5.4.8.5. `__nanosleep()`（`__nanosleep()`）

```cuda


    __device__ void __nanosleep(unsigned nanoseconds);


```

 函数 `__nanosleep(ns)` 将线程挂起大约 `ns` 纳秒的睡眠持续时间。最大睡眠持续时间约为一毫秒。

示例：

以下代码实现具有指数回退的互斥体。

```cuda


    __device__ void mutex_lock(unsigned* mutex) {
        unsigned ns = 8;
        while (atomicCAS(mutex, 0, 1) == 1) {
            __nanosleep(ns);
            if (ns < 256) {
                ns *= 2;
            }
        }
    }

    __device__ void mutex_unlock(unsigned *mutex) {
        atomicExch(mutex, 0);
    }


```





<span id="dpx-instructions"></span>

### 5.4.8.6. 动态编程扩展 (DPX) 指令（Dynamic Programming eXtension (DPX) Instructions）

DPX 函数集可以查找最小值和最大值，以及最多三个 16 或32 位有符号或无符号整数参数。有一个可选的ReLU，即钳位到零​​，功能。

 比较函数：

- 三个参数。语义：`max(a,`` ``b,`` ``c)`, `min(a,`` ``b,`` ``c)`.

```cuda


         int __vimax3_s32  (     int,      int,      int);
    unsigned __vimax3_s16x2(unsigned, unsigned, unsigned);
    unsigned __vimax3_u32  (unsigned, unsigned, unsigned);
    unsigned __vimax3_u16x2(unsigned, unsigned, unsigned);

         int __vimin3_s32  (     int,      int,      int);
    unsigned __vimin3_s16x2(unsigned, unsigned, unsigned);
    unsigned __vimin3_u32  (unsigned, unsigned, unsigned);
    unsigned __vimin3_u16x2(unsigned, unsigned, unsigned);


```

-两个参数，带有ReLU。语义：`max(a,`` ``b,`` ``0)`, `max(min(a,`` ``b),`` ``0)`.

```cuda


         int __vimax_s32_relu  (     int,      int);
    unsigned __vimax_s16x2_relu(unsigned, unsigned);

         int __vimin_s32_relu  (     int,      int);
    unsigned __vimin_s16x2_relu(unsigned, unsigned);


```

-三个参数，带有ReLU。语义： `max(a,`` ``b,`` ``c,`` ``0)`, `max(min(a,`` ``b,`` ``c),`` ``0)`.

```cuda


         int __vimax3_s32_relu  (     int,      int,      int);
    unsigned __vimax3_s16x2_relu(unsigned, unsigned, unsigned);

         int __vimin3_s32_relu  (     int,      int,      int);
    unsigned __vimin3_s16x2_relu(unsigned, unsigned, unsigned);


```

- 两个参数，还返回哪个参数较小/较大：

```cuda


         int __vibmax_s32  (     int,      int, bool* pred);
    unsigned __vibmax_u32  (unsigned, unsigned, bool* pred);
    unsigned __vibmax_s16x2(unsigned, unsigned, bool* pred);
    unsigned __vibmax_u16x2(unsigned, unsigned, bool* pred);

         int __vibmin_s32  (     int,      int, bool* pred);
    unsigned __vibmin_u32  (unsigned, unsigned, bool* pred);
    unsigned __vibmin_s16x2(unsigned, unsigned, bool* pred);
    unsigned __vibmin_u16x2(unsigned, unsigned, bool* pred);


```

 融合加法和最小值/最大值：

- 三个参数，将（第一个 + 第二个）与第三个进行比较。语义：`max(a`` ``+`` ``b,`` ``c)`, `min(a`` ``+`` ``b,`` ``c)`

```cuda


         int __viaddmax_s32  (     int,     int,       int);
    unsigned __viaddmax_s16x2(unsigned, unsigned, unsigned);
    unsigned __viaddmax_u32  (unsigned, unsigned, unsigned);
    unsigned __viaddmax_u16x2(unsigned, unsigned, unsigned);

         int __viaddmin_s32  (     int,     int,       int);
    unsigned __viaddmin_s16x2(unsigned, unsigned, unsigned);
    unsigned __viaddmin_u32  (unsigned, unsigned, unsigned);
    unsigned __viaddmin_u16x2(unsigned, unsigned, unsigned);


```

- 三个参数，使用 ReLU，将（第一个 + 第二个）与第三个和零进行比较。语义：`max(a`` ``+`` ``b,`` ``c,`` ``0)`, `max(min(a`` ``+`` ``b,`` ``c),`` ``0)`

```cuda


         int __viaddmax_s32_relu  (     int,      int,      int);
    unsigned __viaddmax_s16x2_relu(unsigned, unsigned, unsigned);

         int __viaddmin_s32_relu  (     int,      int,      int);
    unsigned __viaddmin_s16x2_relu(unsigned, unsigned, unsigned);


```

 这些指令是硬件加速的或软件模拟的，具体取决于计算能力。有关计算能力要求，请参阅 <a href="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions" class="reference external"> 算术指令 </a> 部分。

 完整的 API 可以在 <a href="https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html" class="reference external">CUDA 数学 API 文档中找到</a>.

------------------------------------------------------------------------

DPX 是一种非常有用的工具，用于实现动态编程算法，例如基因组学中的 Smith-Waterman 和 Needleman-Wunsch 算法以及路线优化中的 Floyd-Warshall 算法。

 三个有符号 32 位整数的最大值，带 ReLU：

```cuda


    int a           = -15;
    int b           = 8;
    int c           = 5;
    int max_value_0 = __vimax3_s32_relu(a, b, c); // max(-15, 8, 5, 0) = 8
    int d           = -2;
    int e           = -4;
    int max_value_1 = __vimax3_s32_relu(a, d, e); // max(-15, -2, -4, 0) = 0


```

 两个 32 位有符号整数、另一个 32 位有符号整数和一个零之和的最小值(ReLU):

```cuda


    int a           = -5;
    int b           = 6;
    int c           = -2;
    int max_value_0 = __viaddmax_s32_relu(a, b, c); // max(-5 + 6, -2, 0) = max(1, -2, 0) = 1
    int d           = 4;
    int max_value_1 = __viaddmax_s32_relu(a, d, c); // max(-5 + 4, -2, 0) = max(-1, -2, 0) = 0


```

两个无符号 32 位整数的最小值并确定哪个值较小：

```cuda


    unsigned a = 9;
    unsigned b = 6;
    bool     smaller_value;
    unsigned min_value = __vibmin_u32(a, b, &smaller_value); // min_value is 6, smaller_value is true


```

三对无符号 16 位整数的最大值：

```cuda


    unsigned a         = 0x00050002;
    unsigned b         = 0x00070004;
    unsigned c         = 0x00020006;
    unsigned max_value = __vimax3_u16x2(a, b, c); // max(5, 7, 2) and max(2, 4, 6), so max_value is 0x00070006


```







## 5.4.9. 编译器优化提示（Compiler Optimization Hints）

编译器优化提示用附加信息装饰代码，以帮助编译器优化生成的代码代码。

- 内置函数始终在设备代码中可用。

- 主机代码支持取决于主机编译器。



<span id="id19"></span>

### 5.4.9.1. `#pragma`` ``unroll`（`#pragma`` ``unroll`）

 默认情况下，编译器展开具有已知行程计数的小循环。但是，`#pragma`` ``unroll` 指令可用于控制任何给定循环的展开。该伪指令必须紧邻循环之前放置，并且仅适用于该循环。

 可以选择跟随整型常量表达式。以下是整型常量表达式的情况：

 - 如果不存在，则在其行程计数恒定的情况下，循环将完全展开。

 - 如果计算结果为 `0` or `1`，则不会展开循环。

 - 如果它是非正整数或大于 `INT_MAX`，则编译指示将为

示例：

```cuda


    struct MyStruct {
        static constexpr int value = 4;
    };

    inline constexpr int Count = 4;

    __device__ void foo(int* p1, int* p2) {
        // no argument specified, the loop will be completely unrolled
        #pragma unroll
        for (int i = 0; i < 12; ++i)
            p1[i] += p2[i] * 2;

        // unroll value = 5
        #pragma unroll (Count + 1)
        for (int i = 0; i < 12; ++i)
            p1[i] += p2[i] * 4;

        // unroll value = 1, loop unrolling disabled
        #pragma unroll 1
        for (int i = 0; i < 12; ++i)
            p1[i] += p2[i] * 8;

        // unroll value = 4
        #pragma unroll (MyStruct::value)
        for (int i = 0; i < 12; ++i)
            p1[i] += p2[i] * 16;

        // negative value, pragma unroll ignored
        #pragma unroll -1
        for (int i = 0; i < 12; ++i)
            p1[i] += p2[i] * 2;
    }


```

参见<a href="https://godbolt.org/z/fPMK55PxE" class="reference external">编译器资源管理器上的示例</a>.





### 5.4.9.2. `__builtin_assume_aligned()`（`__builtin_assume_aligned()`）



Hint

建议使用`cuda::std::assume_aligned()`libcu++<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/standard_api.html" class="reference external">C++提供的</a> (<a href="https://en.cppreference.com/w/cpp/memory/assume_aligned.html" class="reference external">函数参考</a>）作为内置函数的可移植且更安全的替代方案。



```cuda


    void* __builtin_assume_aligned(const void* ptr, size_t align)
    void* __builtin_assume_aligned(const void* ptr, size_t align, <integral type> offset)


```

内置函数使编译器能够假设返回的指针至少与`align`字节对齐。

- 三参数版本使编译器能够假定 `(char*)`` ``ptr`` ``-`` ``offset` 至少与 `align` 字节对齐。

`align` 必须是 2 的幂和整数文字。

E 示例：

```cuda


    void* res1 = __builtin_assume_aligned(ptr, 32);    // compiler can assume 'res1' is at least 32-byte aligned
    void* res2 = __builtin_assume_aligned(ptr, 32, 8); // compiler can assume 'res2 = (char*) ptr - 8' is at least 32-byte aligned


```





### 5.4.9.3. `__builtin_assume()` 和 `__assume()`（`__builtin_assume()` and `__assume()`）

```cuda


    void __builtin_assume(bool predicate)
    void __assume        (bool predicate) // only with Microsoft Compiler


```

 内置函数使编译器能够假定布尔参数为 true。如果运行时参数为 false，则行为未定义。请注意，如果参数有副作用，则行为未指定。

示例：

```cuda


    __device__ bool is_greater_than_zero(int value) {
        return value > 0;
    }

    __device__ bool f(int value) {
        __builtin_assume(value > 0);
        return is_greater_than_zero(value); // returns true, without evaluating the condition
    }


```





### 5.4.9.4. `__builtin_expect()`（`__builtin_expect()`）

```cuda


    long __builtin_expect(long input, long expected)


```

内置函数告诉编译器 `input` 预计等于 `expected`，并返回 `input` 的值。它通常用于向编译器提供分支预测信息。它的行为类似于 C++20 `[[likely]]` 和 `[[unlikely]]` <a href="https://en.cppreference.com/w/cpp/language/attributes/likely" class="reference external"> 属性 </a>.

 示例：

```cuda


    // indicate to the compiler that likely "var == 0"
    if (__builtin_expect(var, 0))
        doit();


```





### 5.4.9.5. `__builtin_unreachable()`（`__builtin_unreachable()`）

```cuda


    void __builtin_unreachable(void)


```

 内置函数告诉编译器控制流永远不会到达调用该函数的点。如果控制流在运行时确实达到此点，则程序具有未定义的行为。

此函数对于避免生成无法访问的分支的代码以及禁用针对无法访问的代码的编译器警告非常有用。

示例：

```cuda


    // indicates to the compiler that the default case label is never reached.
    switch (in) {
        case 1:  return 4;
        case 2:  return 10;
        default: __builtin_unreachable();
    }


```





<span id="nv-abi-pragmas"></span>

### 5.4.9.6. 自定义 ABI Pragmas（Custom ABI Pragmas）

The `#pragma`` ``nv_abi` 指令启用在 <a href="../02-programming-gpus/nvcc.html#nvcc-separate-compilation" class="reference internal"> 中编译的应用程序编译</a>模式，以实现与<a href="../02-programming-gpus/nvcc.html#nvcc-separate-compilation" class="reference internal">类似的性能整个程序编译</a>，通过保留函数使用的寄存器数量。

使用该编译指示的语法如下，其中`EXPR`指任何整型常量表达式：

```cuda


    #pragma nv_abi preserve_n_data(EXPR) preserve_n_control(EXPR)


```

-后面的参数`#pragma`` ``nv_abi` 是可选的，可以按任何顺序提供；但是，至少需要一个参数。

- The `preserve_n` 参数限制函数调用期间保留的寄存器数量：

  - `preserve_n_data(EXPR)` 限制数据寄存器的数量。

  - `preserve_n_control(EXPR)` 限制控制寄存器的数量。

The `#pragma`` ``nv_abi`指令可以紧邻设备函数声明或定义之前放置。

```cuda


    #pragma nv_abi preserve_n_data(16)
    __device__ void dev_func();

    #pragma nv_abi preserve_n_data(16) preserve_n_control(8)
    __device__ int dev_func() {
        return 0;
    }


```

或者，它可以直接放置在设备函数内的 C++ 表达式语句内的间接函数调用之前。请注意，虽然支持对自由函数的间接函数调用，但不支持对函数引用或类成员函数的间接调用。

```cuda


    __device__ int dev_func1();

    struct MyStruct {
        __device__ int member_func2();
    };

    __device__ void test() {
        auto* dev_func_ptr = &dev_func1; // type: int (*)(void)
        #pragma nv_abi preserve_n_control(8)
        int v1 = dev_func_ptr();         // CORRECT, indirect call

        #pragma nv_abi preserve_n_control(8)
        int v2 = dev_func1();            // WRONG, direct call; the pragma has no effect
                                         // dev_func1 has type: int(void)

        auto& dev_func_ref = &dev_func1; // type: int (&)(void)
        #pragma nv_abi preserve_n_control(8)
        int v3 = dev_func_ref();         // WRONG, call to a reference
                                         // the pragma has no effect

        auto member_function_ptr = &MyStruct::member_func2; // type: int (MyStruct::*)(void)
        #pragma nv_abi preserve_n_control(8)
        int v4 = member_function_ptr();  // WRONG, indirect call to member function
                                         // the pragma has no effect
    }


```

当应用于设备函数的声明或定义时，编译指示会修改对该函数的任何调用的自定义 ABI 属性。当放置在间接函数调用站点时，它仅影响该特定调用的 ABI 属性。请注意，编译指示仅在放置在调用站点时影响间接函数调用；它对直接函数调用没有影响。

```cuda


    #pragma nv_abi preserve_n_control(8)
    __device__ int dev_func3();

    __device__ int dev_func4();

    __device__ void test() {
        int v1 = dev_func3();            // CORRECT, the pragma affects the direct call

        auto* dev_func_ptr = &dev_func4; // type: int (*)(void)
        #pragma nv_abi preserve_n_control(8)
        int v2 = dev_func_ptr();         // CORRECT, the pragma affects the indirect call

        int v3 = dev_func_ptr();         // WRONG, the pragma has no effect
    }


```

请注意，如果函数声明的编译指示参数与其相应的定义不匹配，则程序的格式不正确。





<span id="nv-mma-throughput-pragma"></span>

### 5.4.9.7. MMA 吞吐量编译指示（MMA Throughput Pragma）

The `nv_mma_throughput` 编译指示是一个指令，用于启用编译器优化，专门针对矩阵乘法和累加操作进行调整。它仅在入口函数范围内有效，即在 `__global__` 函数上有效。通过沿调用路径传播的优化，效果可以扩展到从该入口函数调用的 `__device__` 函数。

E 示例：

```cuda


    #pragma nv_mma_throughput
    __global__  void kernel(){
    ...
    }


```



注意

The `nv_mma_throughput` 编译指示仍处于实验阶段。它启用的设置是针对一组有限的内部 NVIDIA 工作负载进行调整的。无法保证每个内核都能提高性能。









## 5.4.10. 调试和诊断（Debugging and Diagnostics）



<span id="id20"></span>

### 5.4.10.1. Assertion（Assertion）

```cuda


    #define assert(expression) /* unspecified */


```

The `assert()` 如果 `expression` 等于 0，宏将停止内核执行。如果程序在调试器中运行，则会触发断点，从而允许使用调试器检查设备的当前状态。否则，每个线程`expression`等于 0 在通过与主机同步后将消息打印到 stderr`cudaDeviceSynchronize()`, `cudaStreamSynchronize()`, or `cudaEventSynchronize()`。该消息的格式如下：

```text


    <filename>:<line number>:<function>:
    block: [blockIdx.x,blockIdx.y,blockIdx.z],
    thread: [threadIdx.x,threadIdx.y,threadIdx.z]
    Assertion `<expression>` failed.


```

内核的执行被中止，在主机程序中引发中断。这`assert()`宏会导致 CUDA 上下文损坏，导致任何后续 CUDA 调用或内核调用失败`cudaErrorAssert`.

内核执行不受影响，如果`expression`与零不同。

例如，源文件中的以下程序`test.cu`

```cuda


    #include <assert.h>

     __global__ void testAssert(void) {
         int is_one        = 1;
         int should_be_one = 0;

         // This will have no effect
         assert(is_one);

         // This will halt kernel execution
         assert(should_be_one);
     }

     int main(void) {
         testAssert<<<1,1>>>();
         cudaDeviceSynchronize();
         return 0;
     }


```

将输出：

```text


    test.cu:11: void testAssert(): block: [0,0,0], thread: [0,0,0] Assertion `should_be_one` failed.


```

断言旨在用于调试目的。由于它们会影响性能，因此建议在生产代码中禁用它们。可以通过定义在编译时禁用它们`NDEBUG`包含之前的预处理器宏`assert.h` or `<cassert>`，或者通过使用编译器标志`-DNDEBUG`。注意表达式不应该有副作用；否则，禁用断言将影响代码的功能。

The `assert()`宏在两者中都可用`__device__`和`__tile__`代码。





<span id="id21"></span>

### 5.4.10.2. 断点功能（Breakpoint Function）

可以通过调用以下函数来暂停内核函数的执行`__brkpt()`来自任何设备线程的函数。

```cuda


    void __brkpt();


```





<span id="nv-diagnostic-pragmas"></span>

### 5.4.10.3. 诊断指令（Diagnostic Pragmas）

以下编译指示可用于管理引发特定诊断消息时触发的错误的严重性。

```cuda


    #pragma nv_diag_suppress
    #pragma nv_diag_warning
    #pragma nv_diag_error
    #pragma nv_diag_default
    #pragma nv_diag_once


```

这些编译指示的用途如下：

```cuda


    #pragma nv_diag_xxx <error_number1>, <error_number2> ...


```

使用警告消息中显示的错误号指定受影响的诊断。任何诊断都可以更改为错误，但只有警告可以在更改为错误后抑制其严重性或恢复其严重性。这`nv_diag_default`pragma 将诊断的严重性返回到发出任何其他 pragma 之前有效的严重性，即由任何命令行选项修改的消息的正常严重性。以下示例抑制`declared`` ``but`` ``never`` ``referenced`警告`foo()`:

```cuda


    #pragma nv_diag_suppress 177 // "declared but never referenced"
    void foo() {
        int i = 0;
    }

    #pragma nv_diag_default 177
    void bar() {
        int i = 0;
    }


```

以下编译指示可用于保存和恢复当前诊断编译指示状态：

```cuda


    #pragma nv_diagnostic push
    #pragma nv_diagnostic pop


```

示例：

```cuda


    #pragma nv_diagnostic push
    #pragma nv_diag_suppress 177 // "declared but never referenced"
    void foo() {
        int i = 0;
    }

    #pragma nv_diagnostic pop
    void bar() {
        int i = 0; // raise a warning
    }


```

请注意，这些指令仅影响`nvcc`CUDA前端编译器。它们对主机编译器没有影响。

`nvcc`定义宏`__NVCC_DIAG_PRAGMA_SUPPORT__`当支持诊断指令时。







<span id="wmma"></span>

## 5.4.11. 扭曲矩阵函数（Warp Matrix Functions）

C++ 扭曲矩阵运算利用 Tensor Core 来加速以下形式的矩阵问题`D=A*B+C`。计算能力 7.0 或更高的设备的混合精度浮点数据支持这些运算。这需要一个线程中所有线程的合作<a href="../01-introduction/programming-model.html#programming-model-warps-simt" class="reference internal">经</a>。此外，只有当条件在整个代码中的计算结果相同时，才允许在条件代码中执行这些操作。<a href="../01-introduction/programming-model.html#programming-model-warps-simt" class="reference internal">经</a>，否则代码执行可能会挂起。



<span id="wmma-description"></span>

### 5.4.11.1. 描述（Description）

以下所有函数和类型都在命名空间中定义`nvcuda::wmma`。子字节操作被视为预览版，即它们的数据结构和 API 可能会发生变化，并且可能与未来版本不兼容。这个额外的功能定义在`nvcuda::wmma::experimental`命名空间。

```cuda


    template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;

    void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);
    void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
    void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
    void fill_fragment(fragment<...> &a, const T& v);
    void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);


```

`fragment`
一个重载类，包含分布在扭曲中所有线程上的矩阵的一部分。将矩阵元素映射为`fragment`内部存储未指定，并且可能会在未来架构中发生变化。

仅允许某些模板参数组合。第一个模板参数指定片段将如何参与矩阵运算。可接受的值`Use`是：

- `matrix_a`当片段用作第一个被乘数时，`A`,

- `matrix_b`当片段用作第二个被乘数时，`B`, or

- `accumulator`当片段用作源或目标累加器时（`C` or `D`， 分别）。

  The `m`, `n`和`k`大小描述参与乘法累加运算的扭曲宽度矩阵块的形状。每块瓷砖的尺寸取决于它的作用。对于 `matrix_a`，图块的尺寸为 `m`` ``x`` ``k`；对于 `matrix_b`，维度为 `k`` ``x`` ``n`，`accumulator` 图块为 `m`` ``x`` ``n`.

。数据类型 `T` 对于被乘数可以是 `double`, `float`, `__half`, `__nv_bfloat16`, `char`, or `unsigned`` ``char`，对于累加器可以是 `double`, `float`, `int`, or `__half`。如 <a href="#wmma-type-sizes" class="reference internal"> 元素类型和矩阵大小</a> 中所述，支持累加器和被乘数类型的有限组合。必须为 `matrix_a` 和 `matrix_b` 片段指定 Layout 参数。 `row_major` or `col_major` 分别表示矩阵行或列中的元素在内存中是连续的。 `Layout` 矩阵的 `accumulator` 参数应保留 `void` 的默认值。仅当如下所述加载或存储累加器时才指定行或列布局。

`load_matrix_sync`
 等待，直到所有扭曲通道都到达 load_matrix_sync，然后从内存加载矩阵片段 a。 `mptr` 必须是指向内存中矩阵的第一个元素的 256 位对齐指针。 `ldm` 描述连续行（对于行主要布局）或列（对于列主要布局）之间的元素跨度，对于 `__half` 元素类型必须是 8 的倍数，对于 `float` 元素类型必须是 4 的倍数。 （即，两种情况都是 16 字节的倍数）。如果片段是 `accumulator`，则 `layout` 参数必须指定为 `mem_row_major` or `mem_col_major`. For `matrix_a` 和 `matrix_b` 片段，布局是从片段的 `layout` 参数推断的。对于 warp 中的所有线程，`mptr`, `ldm`, `layout` 和 `a` 的所有模板参数的值必须相同。该函数必须由 warp 中的所有线程调用，否则结果未定义。

`store_matrix_sync`
 等待所有 warp Lane 到达 store_matrix_sync，然后将矩阵片段 a 存储到内存。`mptr`必须是指向内存中矩阵第一个元素的 256 位对齐指针。`ldm`描述连续行（对于行主要布局）或列（对于列主要布局）之间元素的跨度，并且必须是 8 的倍数`__half`元素类型或 4 的倍数`float`元素类型。 （即，两种情况都是 16 字节的倍数）。输出矩阵的布局必须指定为`mem_row_major` or `mem_col_major`。的价值观`mptr`, `ldm`, `layout`并且 a 的所有模板参数对于 warp 中的所有线程都必须相同。

`fill_fragment`
用常量值填充矩阵片段`v`。由于矩阵元素到每个片段的映射是未指定的，因此该函数通常由扭曲中的所有线程调用，并具有共同的值`v`.

`mma_sync`
等待所有warp Lane到达mma_sync，然后执行warp同步矩阵乘法累加操作`D=A*B+C`。就地操作，`C=A*B+C`，也支持。的价值`satf`对于扭曲中的所有线程，每个矩阵片段的模板参数必须相同。另外，模板参数`m`, `n`和`k`片段之间必须匹配`A`, `B`, `C`和`D`。该函数必须由 warp 中的所有线程调用，否则结果是未定义的。

If `satf`（饱和到有限值）模式是`true`，以下附加数值属性适用于目标累加器：

- 如果元素结果为+Infinity，则相应的累加器将包含`+MAX_NORM`

- 如果元素结果为 -Infinity，则相应的累加器将包含`-MAX_NORM`

- 如果元素结果为 NaN，则相应的累加器将包含`+0`

因为矩阵元素到每个线程的映射`fragment`未指定，调用后必须从内存（共享或全局）访问各个矩阵元素`store_matrix_sync`。在扭曲中的所有线程将对所有片段元素统一应用逐元素操作的特殊情况下，可以使用以下 `fragment` 类成员来实现直接元素访问。以

```cuda


    enum fragment<Use, m, n, k, T, Layout>::num_elements;
    T fragment<Use, m, n, k, T, Layout>::x[num_elements];


```

 为例，以下代码将 `accumulator` 矩阵图块缩放一半。

```cuda


    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag;
    float alpha = 0.5f; // Same value for all threads in warp
    /*...*/
    for(int t=0; t<frag.num_elements; t++)
    frag.x[t] *= alpha;


```





<span id="wmma-altfp"></span>

### 5.4.11.2. 备用浮点（Alternate Floating Point）

Tensor Core 支持备用类型具有计算能力 8.0 及更高版本的设备上的浮点运算。

`__nv_bfloat16`
此数据格式是替代的 fp16 格式，其范围与 f32 相同，但精度降低（7 位）。您可以直接将此数据格式与 `__nv_bfloat16` 中提供的 `cuda_bf16.h` 类型一起使用。具有 `__nv_bfloat16` 数据类型的矩阵片段需要由 `float` 类型的累加器组成。支持的形状和操作与 `__half`.

`tf32`
 相同，该数据格式是 Tensor Cores 支持的特殊浮点格式，范围与 f32 相同，但精度降低（\>=10 位）。该格式的内部布局是实现定义的。要在 WMMA 运算中使用此浮点格式，必须手动将输入矩阵转换为 tf32 精度。

 为了方便转换，提供了新的内在 `__float_to_tf32`。虽然内在函数的输入和输出参数为 `float` 类型，但输出将以数字形式表示为 `tf32`。此新精度仅适用于 Tensor Core，如果与其他 `float` 类型运算混合使用，结果的精度和范围将不确定。

 一旦输入矩阵 (`matrix_a` or `matrix_b`) 转换为 tf32 精度，`fragment` 与 `precision::tf32` 精度的组合以及数据类型`float` to `load_matrix_sync` 将利用这一新功能。两个累加器片段都必须具有 `float` 数据类型。唯一支持的矩阵大小是 16x16x8 (m-n-k)。

 片段的元素表示为 `float`，因此映射来自`element_type<T>` to `storage_element_type<T>` is:

```cuda


    precision::tf32 -> float


```





<span id="wmma-double"></span>

### 5.4.11.3. 双精度（Double Precision）

Tensor 核心支持计算能力 8.0 及更高版本的设备上的双精度浮点运算。要使用此新功能，必须使用具有 `fragment` 类型的 `double`。 `mma_sync` 操作将使用 .rn（舍入到最接近的偶数）舍入修饰符执行。





<span id="wmma-subbyte"></span>

### 5.4.11.4. 子字节操作（Sub-byte Operations）

 子字节 WMMA 操作提供了一种访问 Tensor Core 的低精度功能的方法。它们被视为预览功能，即它们的数据结构和 API 可能会发生变化，并且可能与未来版本不兼容。此功能可通过 `nvcuda::wmma::experimental` 命名空间使用：

```cuda


    namespace experimental {
        namespace precision {
            struct u4; // 4-bit unsigned
            struct s4; // 4-bit signed
            struct b1; // 1-bit
       }
        enum bmmaBitOp {
            bmmaBitOpXOR = 1, // compute_75 minimum
            bmmaBitOpAND = 2  // compute_80 minimum
        };
        enum bmmaAccumulateOp { bmmaAccumulateOpPOPC = 1 };
    }


```

 对于 4 位精度，可用的 API 保持不变，但必须将 `experimental::precision::u4` or `experimental::precision::s4` 指定为片段数据类型。由于片段的元素打包在一起，因此 `num_storage_elements` 将小于该片段的 `num_elements`。因此，子字节片段的 `num_elements` 变量返回子字节类型 `element_type<T>` 的元素数量。对于单位精度也是如此，在这种情况下，来自 `element_type<T>` to `storage_element_type<T>` 的映射如下：

```cuda


    experimental::precision::u4 -> unsigned (8 elements in 1 storage element)
    experimental::precision::s4 -> int (8 elements in 1 storage element)
    experimental::precision::b1 -> unsigned (32 elements in 1 storage element)
    T -> T  //all other types


```

 子字节片段允许的布局始终为 `row_major`（对于 `matrix_a`）和 `col_major`（对于 `matrix_b`.

）对于子字节操作，`ldm` in `load_matrix_sync` 的值应该是对于元素类型 `experimental::precision::u4` 和 `experimental::precision::s4` 为 32 的倍数，或者对于元素类型 `experimental::precision::b1` 为 128 的倍数（即，两种情况下都是 16 字节的倍数）。



Note

 已弃用对 MMA 指令的以下变体的支持，并将在sm_90:

> - `experimental::precision::u4`
> - `experimental::precision::s4`
> - `experimental::precision::b1`，其中 `bmmaBitOp` 设置为 `bmmaBitOpXOR`



`bmma_sync`
 等待，直到所有扭曲通道都执行了 bmma_sync，然后执行扭曲同步位矩阵乘法累加操作 `D`` ``=`` ``(A`` ``op`` ``B)`` ``+`` ``C`，其中 `op` 由逻辑运算 `bmmaBitOp` 组成，后跟由下式定义的累加：`bmmaAccumulateOp`。可用的操作有：

`bmmaBitOpXOR`、`matrix_a` 中的行与 `matrix_b`

`bmmaBitOpAND` 的 128 位列的 128 位 XOR、`matrix_a` 中的行与 `matrix_b` 的 128 位列的 128 位 AND，可在计算能力为 8.0 和 8.0 的设备上使用

累加操作始终是 `bmmaAccumulateOpPOPC`，它对设置的位数进行计数。





<span id="wmma-restrictions"></span>

### 5.4.11.5. 限制（Restrictions）

张量核心所需的特殊格式对于每个主要和次要设备架构可能有所不同。由于线程只保存整个矩阵的一个片段（不透明的特定于体系结构的 ABI 数据结构），情况变得更加复杂，开发人员不允许假设各个参数如何映射到参与矩阵乘法累加的寄存器。

由于片段是特定于体系结构的，因此如果函数已针对不同的链接兼容体系结构进行编译并链接在一起到同一设备可执行文件中，则将它们从函数 A 传递到函数 B 是不安全的。在这种情况下，片段的大小和布局将特定于一种体系结构，而在另一种体系结构中使用 WMMA API 将导致不正确的结果或潜在的损坏。

sm_70 和 sm_75 是两个链接兼容体系结构的示例，其中片段的布局不同。

```cuda


    fragA.cu: void foo() { wmma::fragment<...> mat_a; bar(&mat_a); }
    fragB.cu: void bar(wmma::fragment<...> *mat_a) { // operate on mat_a }


```

```cuda


    // sm_70 fragment layout
    $> nvcc -dc -arch=compute_70 -code=sm_70 fragA.cu -o fragA.o
    // sm_75 fragment layout
    $> nvcc -dc -arch=compute_75 -code=sm_75 fragB.cu -o fragB.o
    // Linking the two together
    $> nvcc -dlink -arch=sm_75 fragA.o fragB.o -o frag.o


```

 这种未定义的行为在编译时和运行时工具也可能无法检测到，因此需要额外小心以确保片段是一致的。当与为不同的链接兼容架构构建并期望传递 WMMA 片段的遗留库链接时，最有可能出现这种链接危险。

请注意，在弱链接的情况下（例如 CUDA C++ 内联函数），链接器可能会选择任何可用的函数定义，这可能会导致编译单元之间的隐式传递。

为了避免此类问题，矩阵应始终存储到内存中，以便通过外部接口（例如 `wmma::store_matrix_sync(dst,`` ``鈥?;`）传输，然后可以将其作为指针类型安全地传递给 `bar()`。 `float`` ``*dst`\].

请注意，由于 sm_70 可以在 sm_75 上运行，因此上面的示例 sm_75 代码可以更改为 sm_70 并在 sm_75 上正确运行。不过，当与其他 sm_75 单独编译的二进制文件链接时，建议在应用程序中使用 sm_75 本机代码。





<span id="wmma-type-sizes"></span>

### 5.4.11.6. 元素类型和矩阵大小（Element Types and Matrix Sizes）

Tensor 核心支持各种元素类型和矩阵大小。下表列出了支持的 `matrix_a`, `matrix_b` 和 `accumulator` 矩阵的各种组合：



|矩阵A|矩阵B|蓄能器|矩阵大小 (m-n-k) |
|---------------|---------------|-------------|---------------------|
| \_\_一半 | \_\_一半 |浮动| 16x16x16 |
| \_\_一半 | \_\_一半 |浮动| 32x8x16 |
| \_\_一半 | \_\_一半 |浮动| 8x32x16 |
| \_\_一半 | \_\_一半 | \_\_一半 | 16x16x16 |
| \_\_一半 | \_\_一半 | \_\_一半 | 32x8x16 |
| \_\_一半 | \_\_一半 | \_\_一半 | 8x32x16 |
|无符号字符 |无符号字符 |整数 | 16x16x16 |
|无符号字符 |无符号字符 |整数 | 32x8x16 |
|无符号字符 |无符号字符 |整数 | 8x32x16 |
|签名字符 |签名字符 |整数 | 16x16x16 |
|签名字符 |签名字符 |整数 | 32x8x16 |
|签名字符 |签名字符 |整数 | 8x32x16 |



备用浮点支持：



|矩阵A|矩阵B|蓄能器|矩阵大小 (m-n-k) |
|-----------------|-----------------|-------------|---------------------|
| \_\_nv_bfloat16 | \_\_nv_bfloat16 |浮动| 16x16x16 | 16x16x16
| \_\_nv_bfloat16 | \_\_nv_bfloat16 |浮动| 32x8x16 |
| \_\_nv_bfloat16 | \_\_nv_bfloat16 |浮动| 8x32x16 |
|精度::tf32 |精度::tf32 |浮动| 16x16x8 |



双精度支撑：



|矩阵A|矩阵B|蓄能器|矩阵大小 (m-n-k) |
|----------|----------|-------------|---------------------|
|双|双|双| 8x8x4 |



 实验性支持子字节操作：



|矩阵A|矩阵B|蓄能器|矩阵大小 (m-n-k) |
|---------------|---------------|-------------|---------------------|
|精度::u4 |精度::u4 |整数 | 8x8x32 |
|精度::s4 |精度::s4 |整数 | 8x8x32 |
|精度::b1 |精度::b1 |整数 | 8x8x128 |







<span id="wmma-example"></span>

### 5.4.11.7. 示例（Example）

以下代码在单个扭曲中实现 16x16x16 矩阵乘法。

```cuda


    #include <mma.h>
    using namespace nvcuda;

    __global__ void wmma_ker(half *a, half *b, float *c) {
       // Declare the fragments
       wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
       wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
       wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

       // Initialize the output to zero
       wmma::fill_fragment(c_frag, 0.0f);

       // Load the inputs
       wmma::load_matrix_sync(a_frag, a, 16);
       wmma::load_matrix_sync(b_frag, b, 16);

       // Perform the matrix multiplication
       wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

       // Store the output
       wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
    }


```
