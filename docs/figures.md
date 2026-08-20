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
  <div class="figure-card" id="图-12-合并内存访问"><h3>图 12：合并内存访问</h3><img src="/figures/page-073.png" alt="PDF 第 73 页，图 12" /><p><strong>PDF p.73。</strong>一个 warp 中的连续线程访问连续的 4 字节数据，四个 32 字节事务即可覆盖 128 字节请求。<strong>编程含义：</strong>让传输事务中的大多数字节都被实际使用，是提高 global memory 吞吐量的核心。</p></div>
  <div class="figure-card" id="图-13-未合并内存访问"><h3>图 13：未合并内存访问</h3><img src="/figures/page-074.png" alt="PDF 第 74 页，图 13" /><p><strong>PDF p.74。</strong>当连续线程访问相距至少 32 字节的元素时，一个 warp 可能为 32 个线程发出 32 个事务，实际利用率只有 12.5%。<strong>编程含义：</strong>跨步访问会把带宽浪费在没有被线程使用的事务字节上。</p></div>
  <div class="figure-card" id="图-14-全局内存中的矩阵转置"><h3>图 14：全局内存中的矩阵转置</h3><img src="/figures/page-075.png" alt="PDF 第 75 页，图 14" /><p><strong>PDF p.75。</strong>矩阵转置把读写方向交换，线程在一侧访问连续元素，另一侧可能产生跨步访问。<strong>编程含义：</strong>仅仅让线程索引“看起来连续”还不够，需要分别分析 load 和 store 是否合并，必要时用 shared memory 重排。</p></div>
  <div class="figure-card" id="图-15-共享内存中的步长访问"><h3>图 15：共享内存中的步长访问</h3><img src="/figures/page-077.png" alt="PDF 第 77 页，图 15" /><p><strong>PDF p.77。</strong>共享内存被划分为 bank，步长为 1 的访问通常无冲突，步长增大可能让多个线程命中同一 bank。<strong>编程含义：</strong>分析 shared-memory kernel 时，除了总数据量，还要检查一个 warp 的地址到 bank 的映射。</p></div>
  <div class="figure-card" id="图-16-不规则共享内存访问"><h3>图 16：不规则 Shared Memory 访问</h3><img src="/figures/page-078.png" alt="PDF 第 78 页，图 16" /><p><strong>PDF p.78。</strong>多个线程访问同一 bank 中的同一个字时，可以通过 broadcast 共享读取；随机置换也可能保持无冲突。<strong>编程含义：</strong>bank conflict 的判断不能只看地址是否不连续，还要区分同一位置的广播访问。</p></div>
  <div class="figure-card" id="图-17-32x32共享内存-bank结构"><h3>图 17：32 x 32 Shared Memory 的 Bank 结构</h3><img src="/figures/page-082.png" alt="PDF 第 82 页，图 17" /><p><strong>PDF p.82。</strong>warp 沿 32 x 32 数组的一列访问时，连续线程以 32 个元素为步长，集中命中同一 bank；横跨一行访问则落到不同 bank。<strong>编程含义：</strong>转置 kernel 即使 global memory 访问合并，也可能在 shared memory 中产生 32 路冲突。</p></div>
  <div class="figure-card" id="图-18-带-padding-的共享内存"><h3>图 18：32 x 33 共享内存数组</h3><img src="/figures/page-084.png" alt="PDF 第 84 页，图 18" /><p><strong>PDF p.84。</strong>在 32 x 32 转置缓冲区旁增加一列 padding，让相邻行落到不同 bank 的模式发生变化。<strong>编程含义：</strong>一个常见的消除 bank conflict 技巧是改变 tile 的 leading dimension，而不是修改算法结果。</p></div>
  <div class="figure-card" id="图-19-tile-space索引"><h3>图 19：Partition View 的 Tile-Space 索引</h3><img src="/figures/page-095.png" alt="PDF 第 95 页，图 19" /><p><strong>PDF p.95。</strong>一个 10 x 16 array 被划分为 2 x 4 tile，形成 5 x 4 tile grid；每个 tile-space 坐标对应 array 中的一个规则区域。<strong>编程含义：</strong>view 将 block 索引转换为可预测的区域 load/store，边界 tile 需要额外的 padding 或 mask 规则。</p></div>
  <div class="figure-card" id="图-20-cuda-stream异步并发"><h3>图 20：使用 CUDA Stream 的异步并发执行</h3><img src="/figures/page-118.png" alt="PDF 第 118 页，图 20" /><p><strong>PDF p.118。</strong>不同 stream 中的 kernel、内存复制和 host 工作可以在时间线上重叠，而同一 stream 内仍保持顺序。<strong>编程含义：</strong>异步并发依赖独立 stream、硬件 copy engine 和正确的同步点；看到 API 返回并不等于 GPU 工作已完成。</p></div>
</div>

## 第三部分：高级 CUDA

<div class="figure-grid">
  <div class="figure-card" id="图-22-warp-的划分"><h3>图 22：线程块被划分为 warp</h3><img src="/figures/page-165.png" alt="PDF 第 165 页，图 22" /><p><strong>PDF p.165。</strong>线程块按每组 32 个线程划分为 warp。<strong>编程含义：</strong>线程块总数不是必须为 32 的倍数，但这样可以避免最后一个 warp 中出现长期闲置的 lane，并有助于获得更稳定的执行效率。</p></div>
  <div class="figure-card" id="图-23-库上下文管理"><h3>图 23：库上下文管理</h3><img src="/figures/page-183.png" alt="PDF 第 183 页，图 23" /><p><strong>PDF p.183。</strong>库初始化时创建 context，完成初始化后将 context 从 host thread 弹出；后续库调用再 push 该 context，完成工作后再次 pop。<strong>编程含义：</strong>Driver API 的 context 是 host thread 的当前状态栈，库应正确管理 push/pop 和 usage count，避免覆盖调用方的 context 或过早销毁共享资源。</p></div>
  <div class="figure-card" id="pdl-程序化依赖-kernel-launch"><h3>程序化依赖 Kernel Launch（PDL）示意图</h3><img src="/figures/page-156.png" alt="PDF 第 156 页，3.1.4 PDL 示意图" /><p><strong>PDF p.156。</strong>图中对比不使用 PDL 时 primary kernel 与 dependent kernel 的串行时间线，以及使用 PDL 后 primary kernel 发出完成通知、dependent kernel 在需要数据的位置等待的时间线。<strong>编程含义：</strong>PDL 允许 dependent kernel 的独立前置工作与 primary kernel 的后续工作重叠，但仍需正确设置 `cudaLaunchAttributeProgrammaticStreamSerialization` 和依赖同步函数。</p></div>
</div>

## 第四部分：CUDA 功能图版

<div class="figure-grid">
  <div class="figure-card" id="图-24-child-graph"><h3>图 24：Child Graph 示例</h3><img src="/figures/page-226.png" alt="PDF 第 226 页，图 24" /><p><strong>PDF p.226。</strong>一个 CUDA Graph 被作为另一个图的子图，展示图节点之间的层次关系。<strong>编程含义：</strong>child graph 让可复用的执行结构嵌入父图，依赖关系仍需显式建模。</p></div>
  <div class="figure-card" id="图-25-使用-graph-api-创建图"><h3>图 25：使用 Graph API 创建图</h3><img src="/figures/page-227.png" alt="PDF 第 227 页，图 25" /><p><strong>PDF p.227。</strong>图中把 kernel、memory copy 和依赖边组合成一个简单 CUDA Graph。<strong>编程含义：</strong>Graph API 直接描述执行拓扑，适合需要显式控制节点和边的工作流。</p></div>
  <div class="figure-card" id="图-26-cuda-graph-归约"><h3>图 26：两阶段归约 CUDA Graph</h3><img src="/figures/page-231.png" alt="PDF 第 231 页，图 26" /><p><strong>PDF p.231。</strong>图示展示用 Graph API 或 stream capture 表达两阶段归约。<strong>编程含义：</strong>捕获已有 stream 工作流可以减少重写成本，但捕获限制和图更新策略仍需检查。</p></div>
  <div class="figure-card" id="图-27-条件-if-节点"><h3>图 27：Conditional IF 节点</h3><img src="/figures/page-240.png" alt="PDF 第 240 页，图 27" /><p><strong>PDF p.240。</strong>图执行时根据条件选择 IF 节点中的分支。<strong>编程含义：</strong>条件节点把运行时控制流纳入可实例化图，但分支结构和节点生命周期必须符合 CUDA Graph 规则。</p></div>
  <div class="figure-card" id="图-28-条件-while-节点"><h3>图 28：Conditional WHILE 节点</h3><img src="/figures/page-243.png" alt="PDF p.243，图 28" /><p><strong>PDF p.243。</strong>WHILE 节点反复执行子图，直到条件不再满足。<strong>编程含义：</strong>设备端循环可以减少 host 重新提交，但要控制迭代次数和图内同步成本。</p></div>
  <div class="figure-card" id="图-29-条件-switch-节点"><h3>图 29：Conditional SWITCH 节点</h3><img src="/figures/page-245.png" alt="PDF p.245，图 29" /><p><strong>PDF p.245。</strong>SWITCH 节点依据选择值执行多个子图中的一个。<strong>编程含义：</strong>可把有限状态工作流保留在 GPU 端，分支子图仍要提前构建和管理。</p></div>
  <div class="figure-card" id="图-30-kernel-节点"><h3>图 30：Kernel 节点</h3><img src="/figures/page-249.png" alt="PDF p.249，图 30" /><p><strong>PDF p.249。</strong>图中展示 kernel node 的参数、执行配置和依赖关系。<strong>编程含义：</strong>图节点参数更新与图实例化是独立步骤，不能把 host 端普通 launch 语义直接套用到图更新。</p></div>
  <div class="figure-card" id="图-31-添加分配节点-1"><h3>图 31：添加新的分配节点</h3><img src="/figures/page-256.png" alt="PDF p.256，图 31" /><p><strong>PDF p.256。</strong>图展示把 allocation node 加入已有执行图。<strong>编程含义：</strong>图内分配和释放需要与后续节点的访问依赖匹配，避免内存生命周期悬空。</p></div>
  <div class="figure-card" id="图-32-添加分配节点-2"><h3>图 32：添加新的分配节点（另一种布局）</h3><img src="/figures/page-257.png" alt="PDF p.257，图 32" /><p><strong>PDF p.257。</strong>新增分配节点后，图拓扑显式包含资源创建和使用阶段。<strong>编程含义：</strong>图内内存节点可复用执行结构，但必须确认 allocator、访问顺序和回收节点。</p></div>
  <div class="figure-card" id="图-33-顺序启动的图"><h3>图 33：顺序启动的 Graph</h3><img src="/figures/page-259.png" alt="PDF p.259，图 33" /><p><strong>PDF p.259。</strong>多个图按顺序启动，前一个图完成后才进入后一个图。<strong>编程含义：</strong>图之间的顺序和依赖仍属于应用调度的一部分，不能只依靠图内部节点。</p></div>
  <div class="figure-card" id="图-34-fire-and-forget"><h3>图 34：Fire-and-forget launch</h3><img src="/figures/page-264.png" alt="PDF p.264，图 34" /><p><strong>PDF p.264。</strong>device 端启动的子图不需要等待 host 重新提交即可执行。<strong>编程含义：</strong>适合设备端快速生成后续工作，但要明确启动资源和父子图生命周期。</p></div>
  <div class="figure-card" id="图-35-fire-and-forget-执行环境"><h3>图 35：带执行环境的 Fire-and-forget launch</h3><img src="/figures/page-265.png" alt="PDF p.265，图 35" /><p><strong>PDF p.265。</strong>图示展示 fire-and-forget graph 在执行环境中的提交路径。<strong>编程含义：</strong>执行环境会影响设备端图启动的资源和并发关系。</p></div>
  <div class="figure-card" id="图-36-嵌套-fire-and-forget"><h3>图 36：嵌套 Fire-and-forget 环境</h3><img src="/figures/page-266.png" alt="PDF p.266，图 36" /><p><strong>PDF p.266。</strong>一个设备端图启动另一个图，形成嵌套执行环境。<strong>编程含义：</strong>嵌套启动扩大了设备端调度能力，也增加了依赖、资源和调试复杂度。</p></div>
  <div class="figure-card" id="图-37-stream-环境"><h3>图 37：可视化的 Stream Environment</h3><img src="/figures/page-267.png" alt="PDF p.267，图 37" /><p><strong>PDF p.267。</strong>图示把 stream environment 与设备端图节点的关系画出。<strong>编程含义：</strong>设备端 graph launch 使用特定 stream 语义，必须区分 host stream 与 device launch stream。</p></div>
  <div class="figure-card" id="图-38-简单-tail-launch"><h3>图 38：简单 Tail Launch</h3><img src="/figures/page-268.png" alt="PDF p.268，图 38" /><p><strong>PDF p.268。</strong>当前图完成后在 tail 位置排队启动后续图。<strong>编程含义：</strong>Tail launch 适合设备端连续推进工作，但启动顺序和资源预算需要显式设计。</p></div>
  <div class="figure-card" id="图-39-tail-launch-排序"><h3>图 39：Tail Launch 的排序</h3><img src="/figures/page-268.png" alt="PDF p.268，图 39" /><p><strong>PDF p.268。</strong>图展示 tail-launched graph 在执行队列中的顺序。<strong>编程含义：</strong>不能把设备端排队误认为任意节点都能并行，仍需依据 graph stream 的排序规则分析。</p></div>
  <div class="figure-card" id="图-40-多个图的-tail-launch"><h3>图 40：多个 Graph 入队时的 Tail Launch 排序</h3><img src="/figures/page-268.png" alt="PDF p.268，图 40" /><p><strong>PDF p.268。</strong>来自多个图的 tail launch 会形成更复杂的入队顺序。<strong>编程含义：</strong>复杂设备端调度更需要把依赖、优先级和可见性写进图结构。</p></div>
  <div class="figure-card" id="图-41-简单-sibling-launch"><h3>图 41：简单 Sibling Launch</h3><img src="/figures/page-269.png" alt="PDF p.269，图 41" /><p><strong>PDF p.269。</strong>同一执行环境中的 sibling graph 共享父级启动关系。<strong>编程含义：</strong>Sibling launch 可表达并行的子工作，但不自动建立子图之间的数据依赖。</p></div>
  <div class="figure-card" id="图-42-gpu-活动时间线"><h3>图 42：GPU 活动时间线</h3><img src="/figures/page-292.png" alt="PDF p.292，图 42" /><p><strong>PDF p.292。</strong>时间线展示多个 kernel 在 GPU 上按 stream 顺序或重叠执行。<strong>编程含义：</strong>PDL 的收益来自重叠独立工作和隐藏 launch latency，而不是取消真实的数据依赖。</p></div>
  <div class="figure-card" id="图-43-secondary-kernel-前置阶段"><h3>图 43：Secondary Kernel 的前置阶段</h3><img src="/figures/page-292.png" alt="PDF p.292，图 43" /><p><strong>PDF p.292。</strong>secondary kernel 在等待 primary kernel 数据前先执行初始化和独立工作。<strong>编程含义：</strong>只有明确划分 independent 与 dependent 部分，PDL 才能安全重叠。</p></div>
  <div class="figure-card" id="图-44-primary-secondary-重叠"><h3>图 44：Primary 与 Secondary Kernel 的并发执行</h3><img src="/figures/page-293.png" alt="PDF p.293，图 44" /><p><strong>PDF p.293。</strong>两个 kernel 的独立阶段发生重叠，secondary 在依赖数据的位置等待。<strong>编程含义：</strong>需要使用 programmatic stream serialization 属性和对应的 device-side synchronization。</p></div>
  <div class="figure-card" id="图-45-green-context-资源分区"><h3>图 45：Green Context 的资源分区动机</h3><img src="/figures/page-296.png" alt="PDF p.296，图 45" /><p><strong>PDF p.296。</strong>静态划分一部分 SM 资源后，延迟敏感的工作可以更早开始，不必等待另一个大任务释放全部 SM。<strong>编程含义：</strong>资源隔离可能改善尾延迟，但会牺牲共享资源的弹性，应通过实际 timeline 测量决定分区。</p></div>
  <div class="figure-card" id="图-46-按数量划分-sm-资源"><h3>图 46：使用 `cudaDevSmResourceSplitByCount` 划分 SM 资源</h3><img src="/figures/page-303.png" alt="PDF p.303，图 46" /><p><strong>PDF p.303。</strong>请求把输入 SM 资源切分为若干个大小相同的组，硬件可能根据粒度和对齐要求调整最终组数。<strong>编程含义：</strong>请求值不一定等于结果，应用必须检查返回的资源描述。</p></div>
  <div class="figure-card" id="图-47-按参数划分-sm-资源"><h3>图 47：使用 `cudaDevSmResourceSplit` 划分 SM 资源</h3><img src="/figures/page-305.png" alt="PDF p.305，图 47" /><p><strong>PDF p.305。</strong>每个结果资源可以拥有不同数量的 SM，但不会得到零 SM 的资源。<strong>编程含义：</strong>按参数分区适合不同优先级任务，但应验证每个 green context 的实际资源量。</p></div>
  <div class="figure-card" id="图-48-nsight-compute-green-context"><h3>图 48：Nsight Compute 中的 Green Context 资源</h3><img src="/figures/page-314.png" alt="PDF p.314，图 48" /><p><strong>PDF p.314。</strong>图示用 Nsight Compute 检查多个 green context 的 SM 资源覆盖范围。<strong>编程含义：</strong>性能调试不仅要看 API 返回成功，还要确认资源没有重叠或超额订阅。</p></div>
  <div class="figure-card" id="图-49-没有-green-context 的时间线"><h3>图 49：没有 Green Context 的 Nsight Systems 时间线</h3><img src="/figures/page-315.png" alt="PDF p.315，图 49" /><p><strong>PDF p.315。</strong>长 kernel 占用全部 SM，后启动的高优先级短 kernel 仍要等待资源释放。<strong>编程含义：</strong>stream priority 不能在没有可用 SM 时强行抢占已运行工作。</p></div>
  <div class="figure-card" id="图-50-使用-green-context 的时间线"><h3>图 50：使用 Green Context 的 Nsight Systems 时间线</h3><img src="/figures/page-316.png" alt="PDF p.316，图 50" /><p><strong>PDF p.316。</strong>为长、短 kernel 分配不重叠的 green context 后，短 kernel 可以近乎立即开始。<strong>编程含义：</strong>资源隔离改善了可预测性，但长期任务的持续时间可能增加。</p></div>
  <div class="figure-card" id="图-51-tma-无-swizzle"><h3>图 51：没有 Swizzle 时的共享内存布局</h3><img src="/figures/page-383.png" alt="PDF p.383，图 51" /><p><strong>PDF p.383。</strong>转置缓冲区的一列元素集中落入同一 bank，store 被串行化并形成八路冲突。<strong>编程含义：</strong>TMA swizzle 的目标是改变共享内存布局，降低转置和矩阵搬运中的 bank conflict。</p></div>
  <div class="figure-card" id="图-52-tma-swizzle"><h3>图 52：使用 TMA Swizzle 的布局</h3><img src="/figures/page-384.png" alt="PDF p.384，图 52" /><p><strong>PDF p.384。</strong>`CU_TENSOR_MAP_SWIZZLE_128B` 让行、列元素分散到不同 bank。<strong>编程含义：</strong>这是数据布局层面的优化，要求 tile 尺寸、对齐、TMA 描述符和目标 GPU 能力相互匹配。</p></div>
  <div class="figure-card" id="图-53-tma-swizzle-模式"><h3>图 53：TMA Swizzle 模式概览</h3><img src="/figures/page-387.png" alt="PDF p.387，图 53" /><p><strong>PDF p.387。</strong>图示对比不同 swizzle 宽度和索引变换方式。<strong>编程含义：</strong>选择 swizzle 需要同时满足内维、对齐、共享内存容量和访问模式要求。</p></div>
  <div class="figure-card" id="图-54-cluster-launch-control"><h3>图 54：Cluster Launch Control 流程</h3><img src="/figures/page-392.png" alt="PDF p.392，图 54" /><p><strong>PDF p.392。</strong>cluster 中的 block 动态获取尚未处理的工作，并在没有剩余工作时结束。<strong>编程含义：</strong>工作窃取能改善不均匀负载，但必须正确处理工作计数、cluster 协作和边界。</p></div>
  <div class="figure-card" id="图-55-vmm-使用概览"><h3>图 55：虚拟内存管理使用概览</h3><img src="/figures/page-412.png" alt="PDF p.412，图 55" /><p><strong>PDF p.412。</strong>VMM 流程从环境检查开始，再创建物理分配、保留虚拟地址、映射并设置访问权限。<strong>编程含义：</strong>虚拟地址保留和物理内存分配是两个阶段，映射权限也需要显式管理，不能把 VMM 当成普通 `cudaMalloc` 的别名。</p></div>
  <div class="figure-card" id="图-56-unicast-memory-sharing"><h3>图 56：单播内存共享示例</h3><img src="/figures/page-413.png" alt="PDF p.413，图 56" /><p><strong>PDF p.413。</strong>一段物理分配被映射到多个设备或地址空间中的不同虚拟地址。<strong>编程含义：</strong>VMM 共享要求显式管理句柄、映射关系和访问权限，地址相同与否不是共享语义的依据。</p></div>
  <div class="figure-card" id="图-57-parent-child-launch"><h3>图 57：Parent-Child Launch Nesting</h3><img src="/figures/page-432.png" alt="PDF p.432，图 57" /><p><strong>PDF p.432。</strong>parent grid 在 device 端启动 child grid，形成嵌套的动态并行执行关系。<strong>编程含义：</strong>parent/child grid 的内存可见性、同步和资源生命周期必须明确，不能把 child launch 当作普通函数调用。</p></div>
</div>

## 图版核对方法

1. 先看标题和 PDF 页码，确认图属于哪个版本。
2. 再看结构：线程层次、数据流、资源边界还是时间线。
3. 最后把图映射回代码：索引、启动配置、内存布局、同步点或 API 生命周期。

如果图中出现硬件数量、最大 cluster size、compute capability 等数字，数字只对本地 PDF 的 Release 13.3 负责；请用官方在线文档核对目标 GPU 和当前驱动。
