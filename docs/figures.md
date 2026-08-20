---
title: PDF 图版
description: CUDA Programming Guide Release 13.3 代表性图版的中文解释
---

# PDF 图版

<div class="source-note">
  下列图片是从项目内 Release 13.3 PDF 的代表性页面按页渲染得到的原版图版截图。图下分为“图上内容”和“编程含义”两层，避免只看装饰性的硬件示意图而忽略它对 CUDA 代码的约束。
</div>

## 第一部分：CUDA 编程模型

<div class="figure-grid">
  <div class="figure-card" id="图-1-gpu-为数据处理投入更多晶体管"><h3>图 1：GPU 为数据处理投入更多晶体管</h3><img src="/figures/page-020.png" alt="PDF 第 20 页，图 1" /><p><strong>PDF p.20。</strong>图中把 CPU 与 GPU 的芯片资源分配放在一起：CPU 更强调缓存和控制流，GPU 把更多晶体管用于并行数据处理单元。<strong>编程含义：</strong>适合 GPU 的工作通常具有大量相同或相近的操作，能让许多线程同时推进。</p></div>
  <div class="figure-card" id="图-2-gpu-由-sm-组成"><h3>图 2：GPU 由许多 SM 组成</h3><img src="/figures/page-022.png" alt="PDF 第 22 页，图 2" /><p><strong>PDF p.22。</strong>GPU 由 GPC 组织多个 SM，SM 内含寄存器、缓存/共享内存和功能单元，设备内存通过互连连接到 GPU。<strong>编程含义：</strong>线程块会被调度到某一个 SM 上执行，资源用量会影响同时驻留的线程块数量。</p></div>
  <div class="figure-card" id="图-3-线程块网格"><h3>图 3：线程块组成的网格</h3><img src="/figures/page-022.png" alt="PDF 第 22 页，图 3" /><p><strong>PDF p.22。</strong>网格由许多具有相同形状的线程块组成，箭头表示线程。网格、线程块都可以是一维、二维或三维。<strong>编程含义：</strong>使用 `blockIdx`、`threadIdx`、`blockDim` 和 `gridDim` 计算每个线程负责的数据坐标。</p></div>
  <div class="figure-card" id="图-4-sm-上的线程块调度"><h3>图 4：SM 上的线程块调度</h3><img src="/figures/page-024.png" alt="PDF 第 24 页，图 4" /><p><strong>PDF p.24。</strong>多个线程块可以同时驻留在一个 SM 上，但不同线程块在网格中的调度顺序没有保证。<strong>编程含义：</strong>跨线程块不能默认存在同步或先后关系；需要全局同步时，通常拆成多个 kernel launch 或使用专门的协作机制。</p></div>
  <div class="figure-card" id="图-5-线程块集群"><h3>图 5：线程块集群</h3><img src="/figures/page-025.png" alt="PDF 第 25 页，图 5" /><p><strong>PDF p.25。</strong>cluster 在 grid 中保持线程块的位置，同时增加了 cluster 内坐标。<strong>编程含义：</strong>支持 compute capability 9.0 及以上的设备可用 cluster 提供更强的组级协作，但必须检查目标硬件和启动配置。</p></div>
  <div class="figure-card" id="图-6-gpc-内的集群调度"><h3>图 6：GPC 内的集群调度</h3><img src="/figures/page-026.png" alt="PDF 第 26 页，图 6" /><p><strong>PDF p.26。</strong>同一 cluster 的线程块以 cluster 形状组织，并同时调度到一个 GPC 的多个 SM。<strong>编程含义：</strong>cluster 级同步和 distributed shared memory 扩展了线程块之间的协作边界，但也带来更具体的资源约束。</p></div>
  <div class="figure-card" id="图-7-warp-分歧"><h3>图 7：warp 分歧与掩码线程</h3><img src="/figures/page-027.png" alt="PDF 第 27 页，图 7" /><p><strong>PDF p.27。</strong>示意图中只有满足条件的线程执行 `if` 体，其他 lane 暂时被 mask。<strong>编程含义：</strong>同一 warp 走不同控制流会产生 warp divergence；条件分支不是绝对不能用，但要关注活跃 lane 数和分支路径的成本。</p></div>
  <div class="figure-card" id="图-8-simt-与-tile-编程模型"><h3>图 8：SIMT 与 tile 编程模型</h3><img src="/figures/page-027.png" alt="PDF 第 27 页，图 8" /><p><strong>PDF p.27。</strong>SIMT 让程序员从逐线程角度编写代码，tile 编程让程序员从逐 block 角度描述 tile 操作，再由编译器映射到线程。<strong>编程含义：</strong>两种模型共享 SM、block、grid 和 device memory，但线程控制粒度不同。</p></div>
  <div class="figure-card" id="图-9-tile-空间与数据移动"><h3>图 9：tile 空间与数据移动</h3><img src="/figures/page-028.png" alt="PDF 第 28 页，图 9" /><p><strong>PDF p.28。</strong>二维 array 被概念上划分为 tile space，load 根据 tile-space 索引取出 tile，边界外元素可填零；store 把 tile 写回对应区域。<strong>编程含义：</strong>tile 的形状、边界处理和 gather/scatter 语义决定了数据移动的正确性。</p></div>
  <div class="figure-card" id="图-10-fatbin-与-ptx"><h3>图 10：可执行文件中的 fatbin 与 PTX</h3><img src="/figures/page-033.png" alt="PDF 第 33 页，图 10" /><p><strong>PDF p.33。</strong>CPU 二进制和 GPU 的 fatbin 共存，fatbin 可以包含针对具体架构的 cubin，也可以包含可被 JIT 编译的 PTX。<strong>编程含义：</strong>发布 CUDA 程序时，架构覆盖范围、兼容性和首次启动编译时间都与打包内容有关。</p></div>
</div>

## 第二部分：访存与性能

<div class="figure-grid">
  <div class="figure-card" id="图-11-线程块网格与内存空间"><h3>图 11：第二部分中的线程块网格</h3><img src="/figures/page-063.png" alt="PDF 第 63 页，图 11" /><p><strong>PDF p.63。</strong>第二部分再次用 grid/block/thread 的层次把 CUDA C++ 代码和线程组织对应起来。<strong>编程含义：</strong>这套层次会直接影响索引表达式、内存访问布局以及 kernel 的可扩展性。</p></div>
  <div class="figure-card" id="图-14-全局内存中的矩阵转置"><h3>图 14：全局内存中的矩阵转置</h3><img src="/figures/page-075.png" alt="PDF 第 75 页，图 14" /><p><strong>PDF p.75。</strong>矩阵转置把读写方向交换，线程在一侧访问连续元素，另一侧可能产生跨步访问。<strong>编程含义：</strong>仅仅让线程索引“看起来连续”还不够，需要分别分析 load 和 store 是否合并，必要时用 shared memory 重排。</p></div>
  <div class="figure-card" id="图-15-共享内存中的步长访问"><h3>图 15：共享内存中的步长访问</h3><img src="/figures/page-077.png" alt="PDF 第 77 页，图 15" /><p><strong>PDF p.77。</strong>共享内存被划分为 bank，步长为 1 的访问通常无冲突，步长增大可能让多个线程命中同一 bank。<strong>编程含义：</strong>分析 shared-memory kernel 时，除了总数据量，还要检查一个 warp 的地址到 bank 的映射。</p></div>
  <div class="figure-card" id="图-18-带-padding-的共享内存"><h3>图 18：32 x 33 共享内存数组</h3><img src="/figures/page-084.png" alt="PDF 第 84 页，图 18" /><p><strong>PDF p.84。</strong>在 32 x 32 转置缓冲区旁增加一列 padding，让相邻行落到不同 bank 的模式发生变化。<strong>编程含义：</strong>一个常见的消除 bank conflict 技巧是改变 tile 的 leading dimension，而不是修改算法结果。</p></div>
  <div class="figure-card" id="图-22-warp-的划分"><h3>图 22：线程块被划分为 warp</h3><img src="/figures/page-165.png" alt="PDF p.165，图 22" /><p><strong>PDF p.165。</strong>线程块按每组 32 个线程划分为 warp。<strong>编程含义：</strong>线程块总数不是必须为 32 的倍数，但这样可以避免最后一个 warp 中出现长期闲置的 lane，并有助于获得更稳定的执行效率。</p></div>
</div>

## 第四部分：新功能图版

<div class="figure-grid">
  <div class="figure-card" id="图-24-子图"><h3>图 24：Child Graph 示例</h3><img src="/figures/page-226.png" alt="PDF 第 226 页，图 24" /><p><strong>PDF p.226。</strong>图中把一个 CUDA Graph 作为另一个图的子图，展示图节点之间的层次关系。<strong>编程含义：</strong>图不是“把所有 kernel 粘成一条字符串”，而是可实例化、可复用的执行图；依赖关系要显式建模。</p></div>
  <div class="figure-card" id="图-26-流捕获与图"><h3>图 26：Stream Capture 与 CUDA Graph</h3><img src="/figures/page-231.png" alt="PDF 第 231 页，图 26" /><p><strong>PDF p.231。</strong>图版对比通过 Graph API 构建图和用 stream capture 记录既有异步调用。<strong>编程含义：</strong>前者适合直接描述依赖，后者适合把已经存在的 stream 工作流转成可重复执行的图。</p></div>
  <div class="figure-card" id="图-45-green-context-资源分区"><h3>图 45：Green Context 的资源分区动机</h3><img src="/figures/page-296.png" alt="PDF 第 296 页，图 45" /><p><strong>PDF p.296。</strong>静态划分一部分 SM 资源后，延迟敏感的工作可以更早开始，不必等待另一个大任务释放全部 SM。<strong>编程含义：</strong>资源隔离可能改善尾延迟，但会牺牲共享资源的弹性，应通过实际 timeline 测量决定分区。</p></div>
  <div class="figure-card" id="图-51-tma-无-swizzle"><h3>图 51：没有 swizzle 时的共享内存布局</h3><img src="/figures/page-383.png" alt="PDF 第 383 页，图 51" /><p><strong>PDF p.383。</strong>图示说明矩阵转置缓冲区中的共享内存索引与全局内存索引对应，某些载入方向会集中访问同一 bank。<strong>编程含义：</strong>TMA swizzle 的目标是改变共享内存布局，降低转置和矩阵搬运中的 bank conflict。</p></div>
  <div class="figure-card" id="图-52-tma-swizzle"><h3>图 52：使用 TMA swizzle 的布局</h3><img src="/figures/page-384.png" alt="PDF 第 384 页，图 52" /><p><strong>PDF p.384。</strong>swizzle 让矩阵元素在访问方向上分散到不同 bank。<strong>编程含义：</strong>它是数据布局层面的优化，要求 tile 尺寸、对齐、TMA 描述符和目标 GPU 能力相互匹配。</p></div>
  <div class="figure-card" id="图-55-vmm-使用概览"><h3>图 55：虚拟内存管理使用概览</h3><img src="/figures/page-412.png" alt="PDF 第 412 页，图 55" /><p><strong>PDF p.412。</strong>VMM 流程从环境检查开始，再创建物理分配、保留虚拟地址、映射并设置访问权限。<strong>编程含义：</strong>虚拟地址保留和物理内存分配是两个阶段，映射权限也需要显式管理，不能把 VMM 当成普通 `cudaMalloc` 的别名。</p></div>
</div>

## 图版核对方法

1. 先看标题和 PDF 页码，确认图属于哪个版本。
2. 再看结构：线程层次、数据流、资源边界还是时间线。
3. 最后把图映射回代码：索引、启动配置、内存布局、同步点或 API 生命周期。

如果图中出现硬件数量、最大 cluster size、compute capability 等数字，数字只对本地 PDF 的 Release 13.3 负责；请用官方在线文档核对目标 GPU 和当前驱动。
