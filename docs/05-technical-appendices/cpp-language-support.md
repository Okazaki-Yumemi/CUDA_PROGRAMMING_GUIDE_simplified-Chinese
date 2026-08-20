---
title: 5.3 C++ 语言支持（C++ Language Support）
description: CUDA 对 C++ 标准、标准库、Lambda、函数包装器及语言限制的支持。
---



<span id="c-cplusplus-language-support"></span>

# 5.3. C++ 语言支持（C++ Language Support）

`nvcc` 根据以下规范处理 CUDA 和设备代码：

- **C++03** (ISO/IEC 14882:2003), `--std=c++03` flag.

- **C++11** (ISO/IEC 14882:2011), `--std=c++11` flag.

- **C++14** (ISO/IEC 14882:2014), `--std=c++14` flag.

- **C++17** (ISO/IEC 14882:2017), `--std=c++17` flag.

- **C++20** (ISO/IEC 14882:2020), `--std=c++20` flag.

- **C++23** (ISO/IEC 14882:2024), `--std=c++23` flag.

Passing `nvcc` `-std=c++<version>` 标志打开与指定版本相关的所有 C++ 功能，并使用相应的 C++ 方言选项调用主机预处理器、编译器和链接器。

编译器支持受支持标准的所有语言功能，但受以下部分中报告的限制的约束。



<span id="cpp11-language-features"></span>

## 5.3.1. C++11 语言功能（C++11 Language Features）



<table id="id27" class="table-no-stripes table">
<caption>表 34 C++11 NVCC 支持的设备代码语言功能<a href="#id27" class="headerlink" title="Link to this table">#</a></caption>
<colgroup>
<col style="width: 50%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<thead>
<tr class="row-odd">
<th class="head">语言功能</th>
<th class="head">C++11 提案</th>
<th class="head">NVCC/CUDA 工具包 7.x</th>
</tr>
</thead>
<tbody>
<tr class="row-even">
<td><a href="#rvalue-references" class="reference internal">R 值参考</a></td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2118.html" class="reference external">N2118</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td> 的 R 值参考<code class="docutils literal notranslate">*this</code></td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2439.htm" class="reference external">N2439</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>通过右值初始化类对象</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1610.html" class="reference external">N1610</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>非静态数据成员初始值设定项</td>
<td><a href="http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2008/n2756.htm" class="reference external">N2756</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>可变参数模板</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2242.pdf" class="reference external">N2242</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td> 扩展可变参数模板模板参数</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2555.pdf" class="reference external">N2555</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td><a href="#initializer-list" class="reference internal">初始化程序列表</a></td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2672.htm" class="reference external">N2672</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>静态断言</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1720.html" class="reference external">N1720</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td><code class="docutils literal notranslate">auto</code> 类型变量</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1984.pdf" class="reference external">N1984</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>多重声明符 <code class="docutils literal notranslate">auto</code></td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1737.pdf" class="reference external">N1737</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td> 删除 auto 作为存储类说明符</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2546.htm" class="reference external">N2546</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td> 新函数声明符语法</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2541.htm" class="reference external">N2541</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td><a href="#lambda-expressions" class="reference internal">Lambda 表达式</a></td>
<td><a href="http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2927.pdf" class="reference external">N2927</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td> 表达式的声明类型</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2343.pdf" class="reference external">N2343</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td> 不完整返回类型</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3276.pdf" class="reference external">N3276</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>直角括号</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1757.html" class="reference external">N1757</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>函数模板的默认模板参数</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#226" class="reference external">DR226</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>解决表达式的SFINAE问题</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2634.html" class="reference external">DR339</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>别名模板</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2258.pdf" class="reference external">N2258</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>外部模板</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1987.htm" class="reference external">N1987</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>空指针常量</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2431.pdf" class="reference external">N2431</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>强类型枚举</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2347.pdf" class="reference external">N2347</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>前向声明枚举</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2764.pdf" class="reference external">N2764</a><br />
<a href="http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1206" class="reference external">DR1206</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>标准化属性语法</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2761.pdf" class="reference external">N2761</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td><a href="#constexpr-functions" class="reference internal">通用常量表达式</a></td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2235.pdf" class="reference external">N2235</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>对齐支持</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2341.pdf" class="reference external">N2341</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>有条件支持的行为</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1627.pdf" class="reference external">N1627</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>将未定义的行为更改为可诊断错误</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1727.pdf" class="reference external">N1727</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>委派构造函数</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1986.pdf" class="reference external">N1986</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>继承构造函数</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2540.htm" class="reference external">N2540</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>显式转换运算符</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2437.pdf" class="reference external">N2437</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>新字符类型</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2249.html" class="reference external">N2249</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>Unicode 字符串文字</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm" class="reference external">N2442</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>原始字符串文字</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm" class="reference external">N2442</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>文字中的通用字符名称</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2170.html" class="reference external">N2170</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>用户定义文字</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2765.pdf" class="reference external">N2765</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>标准布局类型</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2342.htm" class="reference external">N2342</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><a href="#cpp11-defaulted-function" class="reference internal">默认函数</a></td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm" class="reference external">N2346</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>删除函数</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm" class="reference external">N2346</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>扩展友元声明</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1791.pdf" class="reference external">N1791</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>扩展<code class="docutils literal notranslate">sizeof</code></td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2253.html" class="reference external">N2253</a><br />
<a href="http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#850" class="reference external">DR850</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><a href="#inline-namespaces" class="reference internal">内联命名空间</a></td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2535.htm" class="reference external">N2535</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>无限制联合</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2544.pdf" class="reference external">N2544</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><a href="#templates" class="reference internal">作为模板的本地和未命名类型参数</a></td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2657.htm" class="reference external">N2657</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>基于范围的</td>
<td><a href="http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2930.html" class="reference external">N2930</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>显式<code class="docutils literal notranslate">虚拟</code>覆盖</td>
<td><a href="http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2928.htm" class="reference external">N2928</a><br />
<a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3206.htm" class="reference external">N3206</a><br />
<a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3272.htm" class="reference external">N3272</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>对垃圾收集和基于可达性的泄漏检测的最低支持</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2670.htm" class="reference external">N2670</a></td>
<td>❌</td>
</tr>
<tr class="row-odd">
<td>允许移动构造函数抛出 [noexcept]</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3050.html" class="reference external">N3050</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>定义移动特殊成员函数</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3053.html" class="reference external">N3053</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td colspan="3"><strong>并发</strong></td>
</tr>
<tr class="row-even">
<td>序列点</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2239.html" class="reference external">N2239</a></td>
<td>❌</td>
</tr>
<tr class="row-odd">
<td>原子操作</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2427.html" class="reference external">N2427</a></td>
<td>❌</td>
</tr>
<tr class="row-even">
<td>强比较和交换</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2748.html" class="reference external">N2748</a></td>
<td>❌</td>
</tr>
<tr class="row-odd">
<td>双向栅栏</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2752.htm" class="reference external">N2752</a></td>
<td>❌</td>
</tr>
<tr class="row-even">
<td>内存模型</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2429.htm" class="reference external">N2429</a></td>
<td>❌</td>
</tr>
<tr class="row-odd">
<td>数据依赖性排序：原子和内存模型</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2664.htm" class="reference external">N2664</a></td>
<td>❌</td>
</tr>
<tr class="row-even">
<td>传播异常</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2179.html" class="reference external">N2179</a></td>
<td>❌</td>
</tr>
<tr class="row-odd">
<td>允许在信号处理程序中使用原子</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2547.htm" class="reference external">N2547</a></td>
<td>❌</td>
</tr>
<tr class="row-even">
<td>线程本地存储</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2659.htm" class="reference external">N2659</a></td>
<td>❌</td>
</tr>
<tr class="row-odd">
<td>并发动态初始化和销毁</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2660.htm" class="reference external">N2660</a></td>
<td>❌</td>
</tr>
<tr class="row-even">
<td colspan="3"><strong>C99中的功能C++11</strong></td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">__func__</code> 预定义标识符</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2340.htm" class="reference external">N2340</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>C99 预处理器</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1653.htm" class="reference external">N1653</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">long</code><code class="docutils literal notranslate"> </code><code class="docutils literal notranslate">long</code></td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1811.pdf" class="reference external">N1811</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>扩展整数类型</td>
<td><a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1988.pdf" class="reference external">N1988</a></td>
<td>❌</td>
</tr>
</tbody>
</table>







<span id="cpp14-language-features"></span>

## 5.3.2. C++14 语言功能（C++14 Language Features）



|语言特征 | C++14 提案 | NVCC/CUDA 工具包 9.x |
|----|----|----|
|调整某些 C++ 上下文转换 |<a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3323.pdf" class="reference external">N3323</a> | ✅ |
|二进制文字 |<a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3472.pdf" class="reference external">N3472</a> | ✅ |
| <a href="#return-type-deduction" class="reference internal">具有推导返回类型的函数</a> | <a href="https://isocpp.org/files/papers/N3638.html" class="reference external">N3638</a> | ✅ |
|广义 lambda 捕获（init-capture）|<a href="https://isocpp.org/files/papers/N3648.html" class="reference external">N3648</a> | ✅ |
|通用（多态）lambda 表达式 |<a href="https://isocpp.org/files/papers/N3649.html" class="reference external">N3649</a> | ✅ |
| <a href="#variable-templates" class="reference internal">可变模板</a> | <a href="https://isocpp.org/files/papers/N3651.pdf" class="reference external">N3651</a> | ✅ |
|放宽对 constexpr 函数的要求 |<a href="https://isocpp.org/files/papers/N3652.html" class="reference external">N3652</a> | ✅ |
|成员初始值设定项和聚合 |<a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3653.html" class="reference external">N3653</a> | ✅ |
|澄清内存分配 |<a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3664.html" class="reference external">N3664</a> | ❌ |
|大小释放 |<a href="https://isocpp.org/files/papers/n3778.html" class="reference external">N3778</a> | ❌ |
| `[[deprecated]]`属性|<a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3760.html" class="reference external">N3760</a> | ✅ |
|单引号作为数字分隔符 |<a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3781.pdf" class="reference external">N3781</a> | ✅ |

表35NVCC 支持的设备代码 C++14 语言功能<a href="#id28" class="headerlink" title="Link to this table">#</a>{#id28 .table-no-stripes .table}







<span id="cpp17-language-features"></span>

## 5.3.3. C++17 语言特性（C++17 Language Features）



<table id="id29" class="table-no-stripes table">
<caption>表36NVCC 支持的设备代码 C++17 语言功能<a href="#id29" class="headerlink" title="Link to this table">#</a></caption>
<colgroup>
<col style="width: 55%" />
<col style="width: 22%" />
<col style="width: 22%" />
</colgroup>
<thead>
<tr class="row-odd">
<th class="head">语言特性</th>
<th class="head">C++17 提案</th>
<th class="head">NVCC/CUDA 工具包 11.x</th>
</tr>
</thead>
<tbody>
<tr class="row-even">
<td>删除三字母组</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4086.html" class="reference external">N4086</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">u8</code>字符文字</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4267.html" class="reference external">N4267</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>折叠表达式</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4295.html" class="reference external">N4295</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>命名空间和枚举器的属性</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4266.html" class="reference external">N4266</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>嵌套命名空间定义</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4230.html" class="reference external">N4230</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>允许对所有非类型模板参数进行常量求值</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4268.html" class="reference external">N4268</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>扩展<code class="docutils literal notranslate">static_assert</code></td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3928.pdf" class="reference external">N3928</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>的新规则<code class="docutils literal notranslate">auto</code> 从花括号初始化列表中推导</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3922.html" class="reference external">N3922</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>允许模板模板参数中的类型名称</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4051.html" class="reference external">N4051</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">[[fallthrough]]</code> 属性</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0188r1.pdf" class="reference external">P0188R1</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td><code class="docutils literal notranslate">[[nodiscard]]</code>属性</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0189r1.pdf" class="reference external">P0189R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">[[maybe_unused]]</code> 属性</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0212r1.pdf" class="reference external">P0212R1</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>聚合初始化的扩展</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0017r1.html" class="reference external">P0017R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">constexpr</code> lambda</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0170r1.pdf" class="reference external">P0170R1</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>一元折叠和空参数包</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0036r0.pdf" class="reference external">P0036R0</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>概括基于范围的 For 循环</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0184r0.html" class="reference external">P0184R0</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>按值捕获 <code class="docutils literal notranslate">*this</code></td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0018r3.html" class="reference external">P0018R3</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">enum 的构造规则</code><code class="docutils literal notranslate"> </code><code class="docutils literal notranslate">类</code>变量</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0138r2.pdf" class="reference external">P0138R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>HC++的十六进制浮点文字</td>
<td><a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0245r1.html" class="reference external">P0245R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>过度对齐数据的动态内存分配</td>
<td><a href="https://wg21.link/p0035" class="reference external">P0035R4</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>保证复制省略</td>
<td><a href="https://wg21.link/p0135" class="reference external">P0135R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>精炼惯用C++的表达式求值顺序</td>
<td><a href="https://wg21.link/p0145" class="reference external">P0145R3</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td><code class="docutils literal notranslate">constexpr</code><code class="docutils literal notranslate"> </code><code class="docutils literal notranslate">if</code></td>
<td><a href="https://wg21.link/p0292" class="reference external">P0292R2</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>Selection带初始值设定项的语句</td>
<td><a href="https://wg21.link/p0305" class="reference external">P0305R1</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>类模板的模板参数推导</td>
<td><a href="https://wg21.link/p0091" class="reference external">P0091R3</a><br />
<a href="https://wg21.link/p0512r0" class="reference external">P0512R0</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>使用 <code class="docutils literal notranslate">声明非类型模板参数自动</code></td>
<td><a href="https://wg21.link/p0127" class="reference external">P0127R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>不重复使用属性命名空间</td>
<td><a href="https://wg21.link/p0028" class="reference external">P0028R4</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>忽略不支持的非标准属性</td>
<td><a href="https://wg21.link/p0283" class="reference external">P0283R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td><a href="#structured-binding" class="reference internal">结构化绑定</a></td>
<td><a href="https://wg21.link/p0217" class="reference external">P0217R3</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>删除不推荐使用 <code class="docutils literal notranslate"> 寄存器 </code> 关键字</td>
<td><a href="https://wg21.link/p0001" class="reference external">P0001R1</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>删除 不推荐使用 <code class="docutils literal notranslate">operator++(bool)</code></td>
<td><a href="https://wg21.link/p0002" class="reference external">P0002R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>将异常规范作为类型系统的一部分</td>
<td><a href="https://wg21.link/p0012" class="reference external">P0012R1</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td><code class="docutils literal notranslate">__has_include</code> C++17</td>
<td><a href="https://wg21.link/p0061" class="reference external">P0061R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>重述继承构造函数（核心问题 1941 等）</td>
<td><a href="https://wg21.link/p0136" class="reference external">P0136R1</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td><a href="#inline-variables" class="reference internal">内联变量</a></td>
<td><a href="https://wg21.link/p0386r2" class="reference external">P0386R2</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>DR 150，模板模板参数的匹配</td>
<td><a href="https://wg21.link/p0522r0" class="reference external">P0522R0</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>删除动态异常规范</td>
<td><a href="https://wg21.link/p0003r5" class="reference external">P0003R5</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>使用声明中的包扩展</td>
<td><a href="https://wg21.link/p0195r2" class="reference external">P0195R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>A <code class="docutils literal notranslate">byte</code>类型定义</td>
<td><a href="https://wg21.link/p0298r0" class="reference external">P0298R0</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>DR 727，类内显式实例化</td>
<td><a href="https://cplusplus.github.io/CWG/issues/727.html" class="reference external">CWG727</a></td>
<td>✅</td>
</tr>
</tbody>
</table>







<span id="cpp20-language-features"></span>

## 5.3.4. C++20 语言功能（C++20 Language Features）

GCC 版本≤10.0、Clang 版本≤10.0、Microsoft Visual Studio ≤2022 和 nvc++ 版本≤20.7。



Note

前缀为“凄R:”的条目是缺陷报告决议。它们修正了标准并适用于早期的 C++ 标准模式（例如 C++17）；此处列出它们是为了完整性，并非特定于 C++20。





<table id="id30" class="table-no-stripes table">
<caption>表 37 C++20 NVCC 支持的设备代码语言功能<a href="#id30" class="headerlink" title="Link to this table">#</a></caption>
<colgroup>
<col style="width: 55%" />
<col style="width: 22%" />
<col style="width: 22%" />
</colgroup>
<thead>
<tr class="row-odd">
<th class="head">语言功能</th>
<th class="head">C++20 提案</th>
<th class="head">NVCC/CUDA 工具包12.x</th>
</tr>
</thead>
<tbody>
<tr class="row-even">
<td>位字段的默认成员初始值设定项</td>
<td><a href="https://wg21.link/p0683r1" class="reference external">P0683R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>修复 <code class="docutils literal notranslate">const</code> 限定的指向成员的指针</td>
<td><a href="https://wg21.link/p0704r1" class="reference external">P0704R1</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>允许 lambda 捕获 <code class="docutils literal notranslate">[=,</code><code class="docutils literal notranslate"> </code><code class="docutils literal notranslate">this]</code></td>
<td><a href="https://wg21.link/p0409r2" class="reference external">P0409R2</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">__VA_OPT__</code> 用于预处理器逗号elision</td>
<td><a href="https://wg21.link/p0306r4" class="reference external">P0306R4</a><br />
<a href="https://wg21.link/p1042r1" class="reference external">P1042R1</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>指定的初始化器</td>
<td><a href="https://wg21.link/p0329r4" class="reference external">P0329R4</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>通用 lambda 的熟悉模板语法</td>
<td><a href="https://wg21.link/p0428r2" class="reference external">P0428R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>向量的列表推导</td>
<td><a href="https://wg21.link/p0702r1" class="reference external">P0702R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>概念</td>
<td><a href="https://wg21.link/p0734r0" class="reference external">P0734R0</a><br />
<a href="https://wg21.link/p0857r0" class="reference external">P0857R0</a><br />
<a href="https://wg21.link/p1084r2" class="reference external">P1084R2</a><br />
<a href="https://wg21.link/p1141r2" class="reference external">P1141R2</a><br />
<a href="https://wg21.link/p0848r3" class="reference external">P0848R3</a><br />
<a href="https://wg21.link/p1616r1" class="reference external">P1616R1</a><br />
<a href="https://wg21.link/p1452r2" class="reference external">P1452R2</a><br />
<a href="https://wg21.link/p1972r0" class="reference external">P1972R0</a><br />
<a href="https://wg21.link/p1980r0" class="reference external">P1980R0</a><br />
<a href="https://wg21.link/p2092r0" class="reference external">P2092R0</a><br />
<a href="https://wg21.link/p2103r0" class="reference external">P2103R0</a><br />
<a href="https://wg21.link/p2113r0" class="reference external">P2113R0</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>带初始值设定项的基于范围的 for 语句</td>
<td><a href="https://wg21.link/p0614r1" class="reference external">P0614R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>简化隐式 lambda 捕获</td>
<td><a href="https://wg21.link/p0588r1" class="reference external">P0588R1</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>ADL 和不可见的函数模板</td>
<td><a href="https://wg21.link/p0846r0" class="reference external">P0846R0</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">const</code> 与默认复制构造函数不匹配</td>
<td><a href="https://wg21.link/p0641r2" class="reference external">P0641R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>较少的急切实例化<code class="docutils literal notranslate">constexpr</code> 函数</td>
<td><a href="https://wg21.link/p0859r0" class="reference external">P0859R0</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><a href="#cpp20-spaceship" class="reference internal">一致比较</a> (<code class="docutils literal notranslate">operator&lt;=&gt;</code>)</td>
<td><a href="https://wg21.link/p0515r3" class="reference external">P0515R3</a><br />
<a href="https://wg21.link/p0905r1" class="reference external">P0905R1</a><br />
<a href="https://wg21.link/p1120r0" class="reference external">P1120R0</a><br />
<a href="https://wg21.link/p1185r2" class="reference external">P1185R2</a><br />
<a href="https://wg21.link/p1186r3" class="reference external">P1186R3</a><br />
<a href="https://wg21.link/p1630r1" class="reference external">P1630R1</a><br />
<a href="https://wg21.link/p1946r0" class="reference external">P1946R0</a><br />
<a href="https://wg21.link/p1959r0" class="reference external">P1959R0</a><br />
<a href="https://wg21.link/p2002r1" class="reference external">P2002R1</a><br />
<a href="https://wg21.link/p2085r0" class="reference external">P2085R0</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>对专业化的访问检查</td>
<td><a href="https://wg21.link/p0692r1" class="reference external">P0692R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>默认可构造和可分配的无状态 lambda</td>
<td><a href="https://wg21.link/p0624r2" class="reference external">P0624R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>未计算的 Lambda contexts</td>
<td><a href="https://wg21.link/p0315r4" class="reference external">P0315R4</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>空对象的语言支持</td>
<td><a href="https://wg21.link/p0840r2" class="reference external">P0840R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>放宽范围 for 循环自定义点查找规则</td>
<td><a href="https://wg21.link/p0962r1" class="reference external">P0962R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><a href="#structured-binding" class="reference internal">允许对可访问成员进行结构化绑定</a></td>
<td><a href="https://wg21.link/p0969r0" class="reference external">P0969R0</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>放宽结构化绑定自定义点查找规则</td>
<td><a href="https://wg21.link/p0961r1" class="reference external">P0961R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>Down with typename！</td>
<td><a href="https://wg21.link/p0634r3" class="reference external">P0634R3</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>允许 lambda init-capture 中的包扩展</td>
<td><a href="https://wg21.link/p0780r2" class="reference external">P0780R2</a><br />
<a href="https://wg21.link/p2095r0" class="reference external">P2095R0</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>建议的措辞<code class="docutils literal notranslate">likely</code> 和 <code class="docutils literal notranslate">unlikely</code> 属性</td>
<td><a href="https://wg21.link/p0479r5" class="reference external">P0479R5</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>不赞成通过 <code class="docutils literal notranslate">[=]</code></td>
<td><a href="https://wg21.link/p0806r2" class="reference external">P0806R2</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td> 非类型模板参数中的类类型</td>
<td><a href="https://wg21.link/p0732r2" class="reference external">P0732R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>与非类型模板参数不一致</td>
<td><a href="https://wg21.link/p1907r1" class="reference external">P1907R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>Atomic使用填充位进行比较和交换</td>
<td><a href="https://wg21.link/p0528r3" class="reference external">P0528R3</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>对可变大小的类进行高效大小的删除</td>
<td><a href="https://wg21.link/p0722r3" class="reference external">P0722R3</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>允许常量表达式中的虚拟函数调用</td>
<td><a href="https://wg21.link/p1064r0" class="reference external">P1064R0</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>禁止与用户声明的构造函数进行聚合</td>
<td><a href="https://wg21.link/p1008r1" class="reference external">P1008R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">显式(bool)</code></td>
<td><a href="https://wg21.link/p0892r2" class="reference external">P0892R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>有符号整数是两个补码</td>
<td><a href="https://wg21.link/p1236r1" class="reference external">P1236R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">char8_t</code></td>
<td><a href="https://wg21.link/p0482r6" class="reference external">P0482R6</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td><a href="#cpp20-consteval" class="reference internal">立即函数</a> (<code class="docutils literal notranslate">consteval</code>)</td>
<td><a href="https://wg21.link/p1073r3" class="reference external">P1073R3</a><br />
<a href="https://wg21.link/p1937r2" class="reference external">P1937R2</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">std::is_constant_evaluated</code></td>
<td><a href="https://wg21.link/p0595r2" class="reference external">P0595R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>嵌套 <code class="docutils literal notranslate">inline</code> 命名空间</td>
<td><a href="https://wg21.link/p1094r2" class="reference external">P1094R2</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">constexpr</code> 限制的放宽</td>
<td><a href="https://wg21.link/p1002r1" class="reference external">P1002R1</a><br />
<a href="https://wg21.link/p1327r1" class="reference external">P1327R1</a><br />
<a href="https://wg21.link/p1330r0" class="reference external">P1330R0</a><br />
<a href="https://wg21.link/p1331r2" class="reference external">P1331R2</a><br />
<a href="https://wg21.link/p1668r1" class="reference external">P1668R1</a><br />
<a href="https://wg21.link/p0784r7" class="reference external">P0784R7</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>功能测试宏</td>
<td><a href="https://wg21.link/p0941r2" class="reference external">P0941R2</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>模块</td>
<td><a href="https://wg21.link/p1103r3" class="reference external">P1103R3</a><br />
<a href="https://wg21.link/p1766r1" class="reference external">P1766R1</a><br />
<a href="https://wg21.link/p1811r0" class="reference external">P1811R0</a><br />
<a href="https://wg21.link/p1703r1" class="reference external">P1703R1</a><br />
<a href="https://wg21.link/p1874r1" class="reference external">P1874R1</a><br />
<a href="https://wg21.link/p1979r0" class="reference external">P1979R0</a><br />
<a href="https://wg21.link/p1779r3" class="reference external">P1779R3</a><br />
<a href="https://wg21.link/p1857r3" class="reference external">P1857R3</a><br />
<a href="https://wg21.link/p2115r0" class="reference external">P2115R0</a><br />
<a href="https://wg21.link/p1815r2" class="reference external">P1815R2</a></td>
<td>❌</td>
</tr>
<tr class="row-even">
<td>协程</td>
<td><a href="https://wg21.link/p0912r5" class="reference external">P0912R5</a></td>
<td>❌</td>
</tr>
<tr class="row-odd">
<td>聚合的括号初始化</td>
<td><a href="https://wg21.link/p0960r3" class="reference external">P0960R3</a><br />
<a href="https://wg21.link/p1975r0" class="reference external">P1975R0</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>DR：新表达式中的数组大小扣除</td>
<td><a href="https://wg21.link/p1009r2" class="reference external">P1009R2</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>DR：从<code class="docutils literal notranslate">T*</code>转换为bool应被视为缩小</td>
<td><a href="https://wg21.link/p1957r2" class="reference external">P1957R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>更强的Unicode要求</td>
<td><a href="https://wg21.link/p1041r4" class="reference external">P1041R4</a><br />
<a href="https://wg21.link/p1139r2" class="reference external">P1139R2</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>结构化绑定扩展</td>
<td><a href="https://wg21.link/p1091r3" class="reference external">P1091R3</a><br />
<a href="https://wg21.link/p1381r1" class="reference external">P1381R1</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>弃用<code class="docutils literal notranslate">a[b,c]</code></td>
<td><a href="https://wg21.link/p1161r3" class="reference external">P1161R3</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>弃用 <code class="docutils literal notranslate">volatile</code></td>
<td><a href="https://wg21.link/p1152r4" class="reference external">P1152R4</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td><code class="docutils literal notranslate">[[nodiscard("with</code><code class="docutils literal notranslate"> </code><code class="docutils literal notranslate">reason")]]</code></td>
<td><a href="https://wg21.link/p1301r4" class="reference external">P1301R4</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">using</code><code class="docutils literal notranslate"> </code><code class="docutils literal notranslate">enum</code></td>
<td><a href="https://wg21.link/p1099r5" class="reference external">P1099R5</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td> 聚合的类模板参数推导 </td>
<td><a href="https://wg21.link/p1816r0" class="reference external">P1816R0</a><br />
<a href="https://wg21.link/p2082r1" class="reference external">P2082R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td> 别名的类模板参数推导templates</td>
<td><a href="https://wg21.link/p1814r0" class="reference external">P1814R0</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>允许转换为未知边界的数组</td>
<td><a href="https://wg21.link/p0388r4" class="reference external">P0388R4</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td><code class="docutils literal notranslate">constinit</code></td>
<td><a href="https://wg21.link/p1143r2" class="reference external">P1143R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>布局兼容性和指针互换性特征</td>
<td><a href="https://wg21.link/p0466r5" class="reference external">P0466R5</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>DR：检查抽象类类型</td>
<td><a href="https://wg21.link/p0929r2" class="reference external">P0929R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>DR：更多隐式移动</td>
<td><a href="https://wg21.link/p1825r0" class="reference external">P1825R0</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>DR：伪析构函数最终对象生命周期</td>
<td><a href="https://wg21.link/p0593r6" class="reference external">P0593R6</a></td>
<td>✅</td>
</tr>
</tbody>
</table>







<span id="cpp23-language-features"></span>

## 5.3.5. C++23 语言功能（C++23 Language Features）

GCC 版本≤14.0、Clang 版本≤18.0、Microsoft Visual Studio（不支持）和 nvc++ 版本≤24.3。



注意

前缀为“AdR:”的条目是缺陷报告分辨率。它们修正了标准并适用于早期的 C++ 标准模式（例如 C++17、C++20）；此处列出它们是为了完整性，并非特定于 C++23。





Note

**N/A** 中的 NVCC 列表示该功能不适用于设备代码（例如，删除未使用的标准措辞，如垃圾收集支持或主机定义的行为）。





<table id="id31" class="table-no-stripes table">
<caption>表 38 C++23 支持的语言功能设备代码的 NVCC<a href="#id31" class="headerlink" title="Link to this table">#</a></caption>
<colgroup>
<col style="width: 55%" />
<col style="width: 22%" />
<col style="width: 22%" />
</colgroup>
<thead>
<tr class="row-odd">
<th class="head"> 语言功能</th>
<th class="head">C++23 提案</th>
<th class="head">NVCC/CUDA 工具包</th>
</tr>
</thead>
<tbody>
<tr class="row-even">
<td>核心问题 411、1656 和 2333 的建议解决方案；字符和字符串文字中的数字和通用字符转义</td>
<td><a href="https://wg21.link/p2029r4" class="reference external">P2029R4</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>（有符号）size_t</td>
<td><a href="https://wg21.link/p0330r8" class="reference external">P0330R8</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>的文字后缀使 () 对于 lambda 来说更加可选（用 () 来代替！）</td>
<td><a href="https://wg21.link/p1102r2" class="reference external">P1102R2</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>if consteval</td>
<td><a href="https://wg21.link/p1938r3" class="reference external">P1938R3</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>删除垃圾收集支持</td>
<td><a href="https://wg21.link/p2186r2" class="reference external">P2186R2</a></td>
<td>N/A</td>
</tr>
<tr class="row-odd">
<td>DR：使用 Unicode 标准附件 31 的 C++ 标识符语法</td>
<td><a href="https://wg21.link/p1949r7" class="reference external">P1949R7</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>DR：允许重复属性</td>
<td><a href="https://wg21.link/p2156r1" class="reference external">P2156R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>将上下文转换范围缩小为 bool</td>
<td><a href="https://wg21.link/p1401r5" class="reference external">P1401R5</a></td>
<td>❌</td>
</tr>
<tr class="row-even">
<td>在行拼接之前修剪空格</td>
<td><a href="https://wg21.link/p2223r2" class="reference external">P2223R2</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>强制声明顺序布局</td>
<td><a href="https://wg21.link/p1847r4" class="reference external">P1847R4</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>混合字符串文字连接</td>
<td><a href="https://wg21.link/p2201r1" class="reference external">P2201R1</a></td>
<td>N/A</td>
</tr>
<tr class="row-odd">
<td> constexpr 中的非文字变量（以及标签和 goto）函数</td>
<td><a href="https://wg21.link/p2242r3" class="reference external">P2242R3</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>推导此</td>
<td><a href="https://wg21.link/p0847r7" class="reference external">P0847R7</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>一致的字符文字编码</td>
<td><a href="https://wg21.link/p2316r2" class="reference external">P2316R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>添加对预处理指令elifdef和elifndef的支持</td>
<td><a href="https://wg21.link/p2334r1" class="reference external">P2334R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>诊断文本的字符编码</td>
<td><a href="https://wg21.link/p2246r1" class="reference external">P2246R1</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>扩展init语句以允许别名声明</td>
<td><a href="https://wg21.link/p2360r0" class="reference external">P2360R0</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>更改lambda的范围Trailing-return-type</td>
<td><a href="https://wg21.link/p2036r3" class="reference external">P2036R3</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>多维下标运算符</td>
<td><a href="https://wg21.link/p2128r6" class="reference external">P2128R6</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>字符集和编码</td>
<td><a href="https://wg21.link/p2314r4" class="reference external">P2314R4</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>auto(x) 和 auto {x}</td>
<td><a href="https://wg21.link/p0849r8" class="reference external">P0849R8</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>C++20 核心论文缺少功能测试宏</td>
<td><a href="https://wg21.link/p2493r0" class="reference external">P2493R0</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td> lambda 表达式上的属性</td>
<td><a href="https://wg21.link/p2173r1" class="reference external">P2173R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>支持 #warning</td>
<td><a href="https://wg21.link/p2437r1" class="reference external">P2437R1</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>删除不可编码的宽字符文字和多字符宽字符文字</td>
<td><a href="https://wg21.link/p2362r3" class="reference external">P2362R3</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>复合语句末尾的标签（C 兼容性）</td>
<td><a href="https://wg21.link/p2324r2" class="reference external">P2324R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>定界转义序列</td>
<td><a href="https://wg21.link/p2290r3" class="reference external">P2290R3</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>放宽一些常量表达式限制</td>
<td><a href="https://wg21.link/p2448r2" class="reference external">P2448R2</a></td>
<td>❌</td>
</tr>
<tr class="row-even">
<td>更简单的隐式移动</td>
<td><a href="https://wg21.link/p2266r3" class="reference external">P2266R3</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>命名通用字符转义</td>
<td><a href="https://wg21.link/p2071r2" class="reference external">P2071R2</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>静态运算符()</td>
<td><a href="https://wg21.link/p1169r4" class="reference external">P1169R4</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>静态运算符[]</td>
<td><a href="https://wg21.link/p2589r1" class="reference external">P2589R1</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>扩展浮点类型和标准名称</td>
<td><a href="https://wg21.link/p1467r9" class="reference external">P1467R9</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>可移植假设<code class="docutils literal notranslate">[[假设]]</code></td>
<td><a href="https://wg21.link/p1774r8" class="reference external">P1774R8</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>支持 UTF-8 作为可移植源文件编码</td>
<td><a href="https://wg21.link/p2295r6" class="reference external">P2295R6</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>DR：char8_t 兼容性和可移植性修复</td>
<td><a href="https://wg21.link/p2513r4" class="reference external">P2513R4</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>DR：弃用易失性按位复合赋值操作</td>
<td><a href="https://wg21.link/p2327r1" class="reference external">P2327R1</a></td>
<td>❌</td>
</tr>
<tr class="row-odd">
<td>DR：放宽对 wchar_t 的要求以匹配现有的实践</td>
<td><a href="https://wg21.link/p2460r2" class="reference external">P2460R2</a></td>
<td>N/A</td>
</tr>
<tr class="row-even">
<td>DR：在常量表达式中使用未知指针和引用</td>
<td><a href="https://wg21.link/p2280r4" class="reference external">P2280R4</a></td>
<td>❌</td>
</tr>
<tr class="row-odd">
<td>DR：您正在寻找的相等运算符</td>
<td><a href="https://wg21.link/p2468r2" class="reference external">P2468R2</a></td>
<td>❌</td>
</tr>
<tr class="row-even">
<td>在 constexpr 函数中允许静态 constexpr 变量</td>
<td><a href="https://wg21.link/p2647r1" class="reference external">P2647R1</a></td>
<td>✅</td>
</tr>
<tr class="row-odd">
<td>延长基于范围的 for 循环初始值设定项中临时变量的生命周期</td>
<td><a href="https://wg21.link/p2644r1" class="reference external">P2644R1</a><br />
<a href="https://wg21.link/p2718r0" class="reference external">P2718R0</a></td>
<td>✅</td>
</tr>
<tr class="row-even">
<td>DR：保守需要向上传播</td>
<td><a href="https://wg21.link/p2564r3" class="reference external">P2564R3</a></td>
<td>✅</td>
</tr>
</tbody>
</table>







<span id="cpp-standard-library"></span>

## 5.3.6. CUDA C++ 标准库（CUDA C++ Standard Library）

CUDA 提供了 C++ 标准库 (STL) 的实现，称为<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/standard_api.html" class="reference external">库++</a>。该库具有以下优点：

- 这些功能在主机和设备上均可用。

- 兼容所有<a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#id59" class="reference external">Linux</a>和<a href="https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#id2" class="reference external">视窗</a>CUDA 工具包支持的平台。

- 兼容所有<a href="https://developer.nvidia.com/cuda-gpus" class="reference external">GPU架构</a>受 CUDA 工具包的最新两个主要版本支持。

- 兼容所有<a href="https://developer.nvidia.com/cuda-toolkit-archive" class="reference external">CUDA工具包</a>与当前和以前的主要版本。

- 提供最新标准版本中可用的 C++ 标准库功能的 C++17 向后移植，包括 C++20、C++23 和 C++26。

- 支持扩展数据类型，例如128位整数（`__int128`), 半精度浮点数 (`__half`), Bfloat16 (`__nv_bfloat16`）和四精度浮点数（`__float128`).

- 针对设备代码进行高度优化。

此外，`libcu++`提供<a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api.html" class="reference external">扩展功能</a>C++ 标准库中不提供这些功能，可提高生产力和应用程序性能。这些功能包括数学函数、内存操作、同步原语、容器扩展、CUDA 内在函数的高级抽象、C++ PTX 包装器等。

`libcu++`可作为<a href="https://developer.nvidia.com/cuda-downloads" class="reference external">CUDA工具包</a>，以及开源的一部分<a href="https://nvidia.github.io/cccl/unstable/" class="reference external">CCCL</a>存储库。





<span id="id1"></span>

## 5.3.7. C 标准库函数（C Standard Library Functions）



<span id="clock-function"></span>

### 5.3.7.1. `clock()`和`clock64()`（`clock()` and `clock64()`）

```cuda


    __host__ __device__ clock_t   clock();
    __device__          long long clock64();


```

当在设备代码中执行时，它返回每个多处理器计数器的值，该计数器在每个时钟周期递增。在内核的开始和结束处对该计数器进行采样，减去这两个值，并记录每个线程的结果，从而提供设备执行该线程所花费的时钟周期数的估计值。但是，该值并不代表设备执行线程指令所花费的实际时钟周期数。前一个数字大于后一个数字，因为线程是按时间分片的。



暗示

- 对应的<a href="https://en.cppreference.com/w/cpp/chrono/c/clock.html" class="reference external">CUDA C++ 函数</a> `cuda::std::clock()`中提供了`<cuda/std/ctime>`标头。

- 便携式<a href="https://en.cppreference.com/w/cpp/header/chrono" class="reference external">C++</a> `<chrono>`实施方案中还提供了`<cuda/std/chrono>` <a href="https://nvidia.github.io/cccl/unstable/libcudacxx/standard_api/time_library.html#libcudacxx-standard-api-time" class="reference external">标头</a>出于类似目的。







<span id="id2"></span>

### 5.3.7.2. `printf()`（`printf()`）

```cuda


    __host__ __device__ __tile__ int printf(const char* format[, arg, ...]);


```

该函数将格式化的输出从内核打印到主机端输出流。

内核内`printf()`函数的行为与标准 C 库类似`printf()`功能。用户应参阅其主机系统的手册页以获取完整的说明`printf()`行为。本质上，传入的字符串为`format`输出到主机上的流。

The `printf()`命令的执行方式与任何其他设备端函数类似：每个线程并在调用线程的上下文中执行。在多线程内核中，直接调用`printf()`将由每个线程使用该线程指定的数据执行。因此，输出字符串的多个版本将出现在主机流中，每个版本对应于遇到该字符串的线程。`printf()`.

与C标准不同`printf()`，返回打印的字符数，CUDA 的`printf()`返回已解析的参数数量。如果格式字符串后面没有参数，则返回 0。如果格式字符串是`NULL`, `-1`被返回。如果发生内部错误，则返回-2。

在内部，`printf()`使用共享数据结构，因此调用 `printf()` 可能会改变线程的执行顺序。特别是，调用 `printf()` 的线程可能比不调用 `printf()` 的线程占用更长的执行路径，并且该路径的长度取决于 `printf()` 的参数。但是，请注意，除了显式的 `__syncthreads()` 障碍之外，CUDA 不保证线程执行的顺序。因此，无法判断执行顺序是否已被 `printf()` 或硬件中的其他调度行为修改。

The `printf()` 函数在区块代码中的行为与在 SIMT 设备代码中的行为不同。在图块代码中，

-除了标量之外，参数还可以是图块。传递图块参数时，将根据相应的格式说明符打印图块的每个元素。

- 即使格式字符串为 `NULL` 或发生内部错误，返回值始终是提供的参数数量。

- 格式字符串必须是文字。

- 如果参数数量与格式说明符数量不匹配，则会发出错误。在 SIMT 设备代码中，这种情况会导致警告。

------------------------------------------------------------------------

**格式说明符**

 对于标准 `printf()`，格式说明符采用以下形式：`%[flags][width][.precision][size]type`

 支持以下字段。有关所有行为的完整说明，请参阅广泛可用的文档。

- 标志：`#`, `'`` ``'`, `0`, `+`, `-`

- 宽度：`*`, `0-9`

- 精度：`0-9`

- 大小：`h`, `l`, `ll`

- 类型：`%cdiouxXpeEfgGaAs`

------------------------------------------------------------------------

**限制**

 的最终格式`printf()` 输出发生在主机系统上。这意味着格式字符串必须被主机系统的编译器和 C 库理解。虽然已尽一切努力确保 CUDA 支持的格式说明符`printf()`函数是最常见主机编译器支持的通用子集，确切的行为将取决于主机操作系统。

`printf()` 接受标志和类型的所有有效组合。这是因为它无法确定什么在最终输出格式化的主机系统上有效，什么无效。因此，如果程序发出包含无效组合的格式字符串，则输出可能未定义。除了格式字符串之外，

The `printf()` 函数最多可接受 32 个参数。任何其他参数都将被忽略，格式说明符将按原样输出。

由于 Windows 平台（32 位）和 Linux 平台（64 位）上 `long` 类型的大小不同，在 Linux 计算机上编译然后在 Windows 计算机上运行的内核将为包含 `%ld` 的所有格式字符串生成损坏的输出。为了确保安全，建议编译和执行平台匹配。

------------------------------------------------------------------------

**主机端缓冲区**

`printf()`的输出缓冲区在内核启动之前设置为固定大小。缓冲区是循环的，因此如果内核执行期间产生的输出多于缓冲区所能容纳的输出，则旧的输出将被覆盖。仅当执行以下操作之一时才会刷新缓冲区：

-通过 `<<<`` ``>>>` or `cuLaunchKernel()` 启动内核：在启动开始时，如果 `CUDA_LAUNCH_BLOCKING` 环境变量设置为 1，则也在启动结束时，

-通过 `cudaDeviceSynchronize()`, `cuCtxSynchronize()`, `cudaStreamSynchronize()`, `cuStreamSynchronize()`, `cudaEventSynchronize()`, or `cuEventSynchronize()`,

 进行同步-通过任何阻塞版本进行内存复制`cudaMemcpy*()` or `cuMemcpy*()`,

- 通过 `cuModuleLoad()` or `cuModuleUnload()`,

 进行模块加载/卸载 - 通过 `cudaDeviceReset()` or `cuCtxDestroy()`.

 进行上下文销毁 - 在执行 `cudaLaunchHostFunc()` or `cuLaunchHostFunc()`.

 添加的流回调之前请注意，程序退出时缓冲区不会自动刷新。

以下 API 函数设置和检索用于将 `printf()` 参数和内部元数据传输到主机的缓冲区的大小。默认大小为 1 兆字节。

- `cudaDeviceGetLimit(size_t*`` ``size,cudaLimitPrintfFifoSize)`

- `cudaDeviceSetLimit(cudaLimitPrintfFifoSize,`` ``size_t`` ``size)`

------------------------------------------------------------------------

**示例**

 以下代码示例：

```cuda


    #include <stdio.h>

    __global__ void helloCUDA(float value) {
        printf("Hello thread %d, value=%f\n", threadIdx.x, value);
    }

    int main() {
        helloCUDA<<<1, 5>>>(1.2345f);
        cudaDeviceSynchronize();
        return 0;
    }


```

 将输出：

```text


    Hello thread 2, value=1.2345
    Hello thread 1, value=1.2345
    Hello thread 4, value=1.2345
    Hello thread 0, value=1.2345
    Hello thread 3, value=1.2345


```

 请注意，每个线程都会遇到 `printf()` 命令。因此，网格中有多少个线程，就有多少行输出。

请参阅 <a href="https://cuda.godbolt.org/z/d4MPj7qG8" class="reference external"> 编译器资源管理器上的示例</a>.

------------------------------------------------------------------------

，以下代码示例：

```cuda


    #include <stdio.h>

    __global__ void helloCUDA(float value) {
        if (threadIdx.x == 0)
            printf("Hello thread %d, value=%f\n", threadIdx.x, value);
    }

    int main() {
        helloCUDA<<<1, 5>>>(1.2345f);
        cudaDeviceSynchronize();
        return 0;
    }


```

 将输出：

```text


    Hello thread 0, value=1.2345


```

 显然，`if()` 语句限制了哪些线程调用 `printf()`，因此只有一个

参见<a href="https://cuda.godbolt.org/z/YqEss81sf" class="reference external">编译器资源管理器上的示例</a>.

------------------------------------------------------------------------

以下代码示例：

```cuda


    #include "cuda_tile.h"
    #include <cstdio>

    namespace ct = cuda::tiles;

    __tile_global__ void kernel() {
      auto ints = ct::iota<ct::tile<int, ct::shape<4, 4>>>();
      printf("%i\n", ints);
    }

    int main() {
      kernel<<<1,1>>>();
      cudaDeviceSynchronize();
      return 0;
    }


```

将输出：

```text


    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]


```





### 5.3.7.3. `memcpy()`和`memset()`（`memcpy()` and `memset()`）

```cuda


    __host__ __device__ __tile__ void* memcpy(void* dest, const void* src, size_t size);


```

该函数将`size`字节从`src`指向的内存位置复制到指向的内存位置通过`dest`.

```cuda


    __host__ __device__ __tile__ void* memset(void* ptr, int value, size_t size);


```

函数设置`size`指向的内存块的`ptr` to `value`字节，解释为`unsigned`` ``char`.



Hint

，建议使用`cuda::std::memcpy()`header`cuda::std::memset()`中提供的`<cuda/std/cstring>` <a href="https://nvidia.github.io/cccl/unstable/libcudacxx/standard_api/c_library/cstring.html#libcudacxx-standard-api-cstring" class="reference external">和</a>函数作为 `memcpy` 和 `memset`.







### 5.3.7.4. `malloc()` 和 `free()`（`malloc()` and `free()`）

```cuda


    __host__ __device__ void* malloc(size_t size);
    // or cuda::std::malloc(), cuda::std::calloc() in the <cuda/std/cstdlib> header


```

 的更安全版本，函数 `malloc()`（设备端）、`cuda::std::malloc()` 和 `cuda::std::calloc()` 从设备堆中分配至少 `size` 字节，并返回指向已分配内存的指针。如果内存不足以满足请求，则返回 `NULL`。返回的指针保证与 16 字节边界对齐。

```cuda


    __device__ void* __nv_aligned_device_malloc(size_t size, size_t align);
    // or cuda::std::aligned_alloc() in the <cuda/std/cstdlib> header


```

 函数 `__nv_aligned_device_malloc()` 和<a href="https://en.cppreference.com/w/cpp/memory/c/aligned_alloc" class="reference external">C++</a> `cuda::std::aligned_alloc()`从设备堆中分配至少 `size` 字节并返回指向已分配内存的指针。如果没有足够的内存来满足请求的大小或对齐方式，则返回 `NULL`。分配的内存的地址是`align`. `align`的倍数，必须是2的非零幂。

```cuda


    __host__ __device__ void free(void* ptr);
    // or cuda::std::free() in the <cuda/std/cstdlib> header


```

设备端函数`free()`和`cuda::std::free()`释放`ptr`指向的内存，该内存必须由先前调用`malloc()`, `cuda::std::malloc()`, `cuda::std::calloc()`, `__nv_aligned_device_malloc()`, or `cuda::std::aligned_alloc()`. If `ptr` is `NULL`返回，调用`free()` or `cuda::std::free()` 被忽略。使用相同的 `free()` or `cuda::std::free()` 重复调用 `ptr` 具有未定义的行为。

 给定 CUDA 线程通过 `malloc()`, `cuda::std::malloc()`, `cuda::std::calloc()`, `__nv_aligned_device_malloc()`, or `cuda::std::aligned_alloc()` 分配的内存在 CUDA 上下文的生命周期内保持分配状态，或者直到通过调用 `free()` or `cuda::std::free()` 显式释放为止。该内存可供其他 CUDA 线程使用，甚至是后续内核启动的线程。任何 CUDA 线程都可以释放另一个线程分配的内存；但是，应注意确保同一指针不会被释放多次。

------------------------------------------------------------------------

**堆内存 API**

 必须在设备代码中分配或释放内存的任何程序之前指定设备内存堆的大小，包括 `new` 和 `delete` 关键字。如果任何程序在没有显式指定堆大小的情况下使用设备内存堆，则会分配 8 MB 的默认堆。

 以下 API 函数获取和设置堆大小：

- `cudaDeviceGetLimit(size_t*`` ``size,`` ``cudaLimitMallocHeapSize)`

- `cudaDeviceSetLimit(cudaLimitMallocHeapSize,`` ``size_t`` ``size)`

 授予的堆大小至少为 `size` 字节。 <a href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8" class="reference external">cuCtxGetLimit()</a> 和 <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g720e159aeb125910c22aa20fe9611ec2" class="reference external">cudaDeviceGetLimit()</a> 返回当前请求的堆大小。

 当模块加载到上下文中时，会发生堆的实际内存分配，可以通过 CUDA 驱动程序 API 显式加载（请参阅<a href="../03-advanced-cuda/driver-api.html#driver-api-module" class="reference internal">Module</a>) 或通过 CUDA 运行时 API 隐式进行。如果内存分配失败，模块加载会生成 `CUDA_ERROR_SHARED_OBJECT_INIT_FAILED` 错误。

加载模块后，堆大小无法更改，并且不会根据需要动态调整大小。

 为设备堆保留的内存不包括通过主机端 CUDA API 调用分配的内存，例如 `cudaMalloc()`.

------------------------------------------------------------------------

**与主机内存 API 的互操作性**

 通过设备端函数 `malloc()`, `cuda::std::malloc()`, `cuda::std::calloc()`, `__nv_aligned_device_malloc()`, `cuda::std::aligned_alloc()` 分配的内存，或`new` 关键字不能与运行时或驱动程序 API 调用（例如 `cudaMalloc`, `cudaMemcpy`, or `cudaMemset`）一起使用或释放。同样，通过主机运行时 API 分配的内存不能使用设备端函数 `free()`, `cuda::std::free()` 或 `delete` 关键字释放。

------------------------------------------------------------------------

每线程分配示例：

```cuda


    #include <stdlib.h>
    #include <stdio.h>

    __global__ void single_thread_allocation_kernel() {
        size_t size = 123;
        char*  ptr  = (char*) malloc(size);
        memset(ptr, 0, size);
        printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
        free(ptr);
    }

    int main() {
        // Set a heap size of 128 megabytes.
        // Note that this must be done before any kernel is launched.
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
        single_thread_allocation_kernel<<<1, 5>>>();
        cudaDeviceSynchronize();
        return 0;
    }


```

 将输出：

```cuda


    Thread 0 got pointer: 0x20d5ffe20
    Thread 1 got pointer: 0x20d5ffec0
    Thread 2 got pointer: 0x20d5fff60
    Thread 3 got pointer: 0x20d5f97c0
    Thread 4 got pointer: 0x20d5f9720


```

注意每个线程如何遇到 `malloc()` 和 `memset()` 命令并因此接收并初始化其自己的分配。

请参阅 <a href="https://cuda.godbolt.org/z/z7K191z58" class="reference external"> 编译器资源管理器中的示例</a>.

------------------------------------------------------------------------

每线程块分配示例：

```cuda


    #include <stdlib.h>

    __global__ void block_level_allocation_kernel() {
        __shared__ int* data;
        // The first thread in the block performs the allocation and shares the pointer
        // with all other threads through shared memory, so that access can be coalesced.
        if (threadIdx.x == 0) {
            size_t size = blockDim.x * 64; // 64 bytes per thread are allocated.
            data = (int*) malloc(size);
        }
        __syncthreads();
        // Check for failure
        if (data == nullptr)
            return;

        // Threads index into the memory, ensuring coalescence
        for (int i = 0; i < 64; ++i)
            data[i * blockDim.x + threadIdx.x] = threadIdx.x;
        // Ensure all threads complete before freeing
        __syncthreads();

        // Only one thread may free the memory!
        if (threadIdx.x == 0)
            free(data);
    }

    int main() {
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
        block_level_allocation_kernel<<<10, 128>>>();
        cudaDeviceSynchronize();
        return 0;
    }


```

请参阅 <a href="https://cuda.godbolt.org/z/7s8x7oonz" class="reference external"> 编译器资源管理器</a>.

------------------------------------------------------------------------

内核启动之间保留的分配示例：

```cuda


    #include <stdlib.h>
    #include <stdio.h>

    const int NUM_BLOCKS = 20;

    __device__ int* data_ptrs[NUM_BLOCKS]; // Per-block pointer

    __global__ void allocate_memory_kernel() {
        // Only the first thread in the block performs the allocation
        // since we need only one allocation per block.
        if (threadIdx.x == 0)
            data_ptrs[blockIdx.x] = (int*) malloc(blockDim.x * 4);
        __syncthreads();
        // Check for failure
        if (data_ptrs[blockIdx.x] == nullptr)
            return;
        // Zero the data with all threads in parallel
        data_ptrs[blockIdx.x][threadIdx.x] = 0;
    }

    // Simple example: store the thread ID into each element
    __global__ void use_memory_kernel() {
        int* ptr = data_ptrs[blockIdx.x];
        if (ptr != nullptr)
            ptr[threadIdx.x] += threadIdx.x;
    }

    // Print the content of the buffer before freeing it
    __global__ void free_memory_kernel() {
        int* ptr = data_ptrs[blockIdx.x];
        if (ptr != nullptr)
            printf("Block %d, Thread %d: final value = %d\n",
                blockIdx.x, threadIdx.x, ptr[threadIdx.x]);
        // Only free from one thread!
        if (threadIdx.x == 0)
            free(ptr);
    }

    int main() {
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
        // Allocate memory
        allocate_memory_kernel<<<NUM_BLOCKS, 10>>>();

        // Use memory
        use_memory_kernel<<<NUM_BLOCKS, 10>>>();
        use_memory_kernel<<<NUM_BLOCKS, 10>>>();
        use_memory_kernel<<<NUM_BLOCKS, 10>>>();

        // Free memory
        free_memory_kernel<<<NUM_BLOCKS, 10>>>();
        cudaDeviceSynchronize();
        return 0;
    }


```

请参阅以下示例： <a href="https://cuda.godbolt.org/z/h7r6G3dGP" class="reference external">Compiler Explorer</a>.





<span id="alloca-function"></span>

### 5.3.7.5. `alloca()`（`alloca()`）

```cuda


    __host__ __device__ void* alloca(size_t size);


```

The `alloca()` 函数在调用方的堆栈帧内分配 `size` 字节的内存。返回值是指向已分配内存的指针。当从设备代码调用该函数时，内存的开头是 16 字节对齐的。当调用者从`alloca()`.



返回时自动释放内存。注意

在Windows平台上，使用`<malloc.h>`函数之前必须包含`alloca()`头文件。调用`alloca()`可能会导致堆栈溢出；用户需要相应地调整堆栈大小。



示例：

```cuda


    __device__ void device_function(int num_items) {
        int4* ptr = (int4*) alloca(num_items * sizeof(int4));
        // use of ptr
        ...
    }


```







<span id="id3"></span>

## 5.3.8. Lambda 表达式（Lambda Expressions）



编译器通过将 lambda 表达式或闭包类型 (C++11) 与最内层封闭函数作用域的执行空间相关联来确定其执行空间。如果没有封闭函数作用域，则执行空间指定为 `__host__`.





也可以使用 <a href="#extended-lambdas" class="reference internal"> 扩展 lambda 语法显式指定执行空间</a>.



E示例：

```cuda


    auto global_lambda = [](){ return 0; }; // __host__

    void host_function() {
      auto lambda1 = [](){ return 1; };   // __host__
      [](){ return 3; };                  // __host__, closure type (body of a lambda expression)
    }

    __device__ void device_function() {
      auto lambda2 = [](){ return 2; };   // __device__
    }

    __global__ void kernel_function(void) {
      auto lambda3 = [](){ return 3; };   // __device__
    }

    __host__ __device__ void host_device_function() {
      auto lambda4 = [](){ return 4; };   // __host__ __device__
    }

    __tile__ void tile_function() {
      auto lambda5 = [](){ return 5; };   // __tile__
    }

    __tile__ __device__ void tile_device_function() {
      auto lambda6 = [](){ return 6; };   // __tile__ __device__
    }

    using function_ptr_t = int (*)();

    __device__ void device_function(float          value,
                                    function_ptr_t ptr = [](){ return 4; } /* __host__ */) {}


```

请参阅 <a href="https://godbolt.org/z/scv4vcczr" class="reference external">Compiler Explorer</a>.



<span id="lambda-expressions-global"></span>

### 5.3.8.1. Lambda 表达式和 `__global__` 函数上的示例参数（Lambda Expressions and `__global__` Function Parameters）

A lambda 表达式或闭包类型只能用作 `__global__` 函数的参数（如果其执行空间为 `__device__` or `__host__`` ``__device__`）。全局或命名空间范围 lambda 表达式不能用作 `__global__` 函数中的参数。

示例：

```cuda


    template <typename T>
     __global__ void kernel(T input) {}

     __device__ void device_function() {
         // device kernel call requires separate compilation (-rdc=true flag)
         kernel<<<1, 1>>>([](){});
         kernel<<<1, 1>>>([] __device__() {});          // extended lambda
         kernel<<<1, 1>>>([] __host__ __device__() {}); // extended lambda
     }

     auto global_lambda = [] __host__ __device__() {};

     void host_function() {
         kernel<<<1, 1>>>([] __device__() {});          // CORRECT, extended lambda
         kernel<<<1, 1>>>([] __host__ __device__() {}); // CORRECT, extended lambda
     //  kernel<<<1, 1>>>([](){});                      // ERROR, closure type with host execution space
     //  kernel<<<1, 1>>>(global_lambda);               // ERROR, extended lambda, but at global scope
     }


```

请参阅 <a href="https://godbolt.org/z/ajrsn5z5Y" class="reference external">Compiler Explorer</a>.





<span id="id4"></span>

### 5.3.8.2. Extended Lambdas（Extended Lambdas）



The `nvcc` 标志 `--extended-lambda` 允许在 lambda 表达式中显式注释执行空间。这些注释应出现在 lambda 引入符之后和可选的 lambda 声明符之前。





`nvcc` 在指定 `__CUDACC_EXTENDED_LAMBDA__` 标志时定义宏 `--extended-lambda`。



- *扩展 lambda* 在 `__host__` or `__host__`` ``__device__` 函数的直接块或嵌套块的范围内定义。

- *扩展设备 lambda* 是用 `__device__` 关键字注释的 lambda 表达式。

 - *扩展主机设备 lambda* 是用 `__host__`` ``__device__` 关键字注释的 lambda 表达式。

 与标准 lambda 表达式不同，扩展 lambda 可以用作 `__global__` 中的类型参数函数。

示例：

```cuda


    void host_function() {
        auto lambda1 = [] {};                      // NOT an extended lambda: no explicit execution space annotations
        auto lambda2 = [] __device__ {};           // extended lambda
        auto lambda3 = [] __host__ __device__ {};  // extended lambda
        auto lambda4 = [] __host__ {};             // NOT an extended lambda
    }

    __host__ __device__ void host_device_function() {
        auto lambda1 = [] {};                      // NOT an extended lambda: no explicit execution space annotations
        auto lambda2 = [] __device__ {};           // extended lambda
        auto lambda3 = [] __host__ __device__ {};  // extended lambda
        auto lambda4 = [] __host__ {};             // NOT an extended lambda
    }

    __device__ void device_function() {
        // none of the lambdas within this function are extended lambdas,
        // because the enclosing function is not a __host__ or __host__ __device__  function.
        auto lambda1 = [] {};
        auto lambda2 = [] __device__ {};
        auto lambda3 = [] __host__ __device__ {};
        auto lambda4 = [] __host__ {};
    }

    auto global_lambda = [] __host__ __device__ { }; // NOT an extended lambda because it is not defined
                                                     // within a __host__ or __host__ __device__ function


```





<span id="extended-lambda-traits"></span>

### 5.3.8.3. 扩展 Lambda 类型特征（Extended Lambda Type Traits）

编译器提供类型特征来在编译时检测扩展 lambda 的闭包类型。

```cuda


    bool __nv_is_extended_device_lambda_closure_type(type);


```

 函数返回 `true` if `type` 是为扩展 `__device__` lambda 创建的闭包类，`false`否则，

```cuda


    bool __nv_is_extended_device_lambda_with_preserved_return_type(type);


```

函数返回`true` if `type`是为扩展创建的闭包类`__device__`lambda 且 lambda 是使用尾随返回类型定义的，否则为 `false`。如果尾随返回类型定义引用任何 lambda 参数名称，则不保留返回类型。

```cuda


    bool __nv_is_extended_host_device_lambda_closure_type(type);


```

 函数返回 `true` if `type` 是为扩展 `__host__`` ``__device__` lambda 创建的闭包类，否则为 `false`。

------------------------------------------------------------------------

 lambda 类型特征可以在所有编译模式下使用，无论是否启用 lambda 或扩展 lambda。如果扩展 lambda 模式处于非活动状态，这些特征将始终返回 `false`。

示例：

```cuda


    auto lambda0 = [] __host__ __device__ { };

    void host_function() {
        auto lambda1 = [] { };
        auto lambda2 = [] __device__ { };
        auto lambda3 = [] __host__ __device__ { };
        auto lambda4 = [] __device__ () -> double { return 3.14; }
        auto lambda5 = [] __device__ (int x) -> decltype(&x) { return 0; }

        using lambda0_t = decltype(lambda0);
        using lambda1_t = decltype(lambda1);
        using lambda2_t = decltype(lambda2);
        using lambda3_t = decltype(lambda3);
        using lambda4_t = decltype(lambda4);
        using lambda5_t = decltype(lambda5);

        // 'lambda0' is not an extended lambda because it is defined outside function scope
        static_assert(!__nv_is_extended_device_lambda_closure_type(lambda0_t));
        static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda0_t));
        static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda0_t));

        // 'lambda1' is not an extended lambda because it has no execution space annotations
        static_assert(!__nv_is_extended_device_lambda_closure_type(lambda1_t));
        static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda1_t));
        static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda1_t));

        // 'lambda2' is an extended device-only lambda
        static_assert(__nv_is_extended_device_lambda_closure_type(lambda2_t));
        static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda2_t));
        static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda2_t));

        // 'lambda3' is an extended host-device lambda
        static_assert(!__nv_is_extended_device_lambda_closure_type(lambda3_t));
        static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda3_t));
        static_assert(__nv_is_extended_host_device_lambda_closure_type(lambda3_t));

        // 'lambda4' is an extended device-only lambda with preserved return type
        static_assert(__nv_is_extended_device_lambda_closure_type(lambda4_t));
        static_assert(__nv_is_extended_device_lambda_with_preserved_return_type(lambda4_t));
        static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda4_t));

        // 'lambda5' is not an extended device-only lambda with preserved return type
        // because it references the operator()'s parameter types in the trailing return type.
        static_assert(__nv_is_extended_device_lambda_closure_type(lambda5_t));
        static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda5_t));
        static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda5_t));
    }


```





<span id="id5"></span>

### 5.3.8.4. 扩展 Lambda 限制（Extended Lambda Restrictions）

在调用主机编译器之前，CUDA 编译器会将扩展 lambda 表达式替换为命名空间范围中定义的占位符类型的实例。占位符类型的模板参数需要获取包含原始扩展 lambda 表达式的函数的地址。这对于正确执行任何 `__global__` 函数模板（其模板参数涉及扩展 lambda 的闭包类型）是必要的。封闭函数的计算方式如下。

 根据定义，扩展 lambda 存在于 `__host__` or `__host__`` ``__device__` 函数的直接或嵌套块作用域内。

 - 如果函数不是 lambda 表达式的 `operator()`，则将其视为扩展 lambda 的封闭函数。

 - 否则，扩展 lambda 在一个或多个封闭 lambda 表达式的 `operator()` 的直接或嵌套块作用域。

 - 如果最外层 lambda 表达式是在函数 `F` 的直接或嵌套块作用域内定义的，则 `F` 是计算的封闭函数。

 - 否则，封闭函数不会

示例：

```cuda


    void host_function() {
        auto lambda1 = [] __device__ { }; // enclosing function for lambda1 is "host_function()"
        auto lambda2 = [] {
            auto lambda3 = [] {
                auto lambda4 = [] __host__ __device__ { }; // enclosing function for lambda4 is "host_function"
            };
        };
    }

    auto global_lambda = [] {
        auto lambda5 = [] __host__ __device__ { }; // enclosing function for lambda5 does not exist
    };


```

------------------------------------------------------------------------

扩展 Lambda 限制

1. 扩展 lambda 不能在另一个扩展 lambda 表达式内定义。例子：

```cuda


    void host_function() {
        auto lambda1 = [] __host__ __device__  {
             // ERROR, extended lambda defined within another extended lambda
            auto lambda2 = [] __host__ __device__ { };
        };
    }


```

2. 扩展 lambda 不能在通用 lambda 表达式内定义。示例：

```cuda


    void host_function() {
        auto lambda1 = [] (auto) {
             // ERROR, extended lambda defined within a generic lambda
            auto lambda2 = [] __host__ __device__ { };
        };
    }


```

3. 如果扩展 lambda 是在一个或多个嵌套 lambda 表达式的直接或嵌套块作用域内定义的，则最外层 lambda 表达式必须在函数的直接或嵌套块作用域内定义。示例：

```cuda


    auto lambda1 = []  {
        // ERROR, outer enclosing lambda is not defined within a non-lambda-operator() function
        auto lambda2 = [] __host__ __device__ { };
    };


```

4. 扩展 lambda 的封闭函数必须命名，并且其地址必须可访问。如果封闭函数是类成员，则必须满足以下条件:

 - 所有封闭该成员函数的类都必须有名称。

 - 成员函数在其父类中不得具有私有或受保护的访问权限。

 - 所有封闭类在其各自的父类中不得有私有或受保护的访问权限。

示例：

```cuda


    void host_function() {
        auto lambda1 = [] __device__ { return 0; }; // OK
        {
            auto lambda2 = [] __device__          { return 0; }; // OK
            auto lambda3 = [] __device__ __host__ { return 0; }; // OK
        }
    }

    struct MyStruct1 {
        MyStruct1() {
            auto lambda4 = [] __device__ { return 0; }; // ERROR, address of the enclosing function is not accessible
        }
    };

    class MyStruct2 {
        void foo() {
            auto temp1 = [] __device__ { return 10; }; // ERROR, enclosing function has private access in parent class
        }

        struct MyStruct3 {
            void foo() {
                auto temp1 = [] __device__ { return 10; };  // ERROR, enclosing class MyStruct3 has private access in its parent class
            }
        };
    };


```

5. 在定义扩展 lambda 时，必须能够明确地获取封闭例程的地址。但是，这可能并不总是可行，例如，当别名声明隐藏同名的模板类型参数时。示例：

```cuda


    template <typename T>
    struct A {
        using Bar = void;
        void test();
    };

    template<>
    struct A<void> { };

    template <typename Bar>
    void A<Bar>::test() {
        // In code sent to host compiler, nvcc will inject an address expression here, of the form:
        //   (void (A< Bar> ::*)(void))(&A::test))
        //  However, the class typedef 'Bar' (to void) shadows the template argument 'Bar',
        //  causing the address expression in A<int>::test to actually refer to:
        //    (void (A< void> ::*)(void))(&A::test))
        //  which doesn't take the address of the enclosing routine 'A<int>::test' correctly.
        auto lambda1 = [] __host__ __device__ { return 4; };
    }

    int main() {
        A<int> var;
        var.test();
    }


```

6. 扩展 lambda 不能在函数本地的类中定义。示例：

```cuda


    void host_function() {
        struct MyStruct {
            void bar() {
                // ERROR, bar() is member of a class that is local to a function
                auto lambda2 = [] __host__ __device__ { return 0; };
            }
        };
    }


```

7. 扩展 lambda 的封闭函数不能具有推导的返回类型。示例：

```cuda


    auto host_function() {
        // ERROR, the return type of host_function() is deduced
        auto lambda3 = [] __host__ __device__ { return 0; };
    }


```

8. 主机设备扩展 lambda 不能是通用 lambda，即具有 `auto` 参数类型的 lambda。示例：

```cuda


    void host_function() {
        // ERROR, __host__ __device__ extended lambdas cannot be a generic lambda
        auto lambda1 = [] __host__ __device__ (auto i) { return i; };

        // ERROR, a host-device extended lambda cannot be a generic lambda
        auto lambda2 = [] __host__ __device__ (auto... i) {
            return sizeof...(i);
        };
    }


```

9. 如果封闭函数是函数或成员模板的实例化，或者函数是类模板的成员，则模板必须满足以下约束：

 - 模板最多只能有一个可变参数，并且它必须列在模板参数列表的最后。

 - 模板参数必须命名。

- 模板实例化参数类型不能涉及函数本地类型（扩展 lambda 的闭包类型除外）或 `private` or `protected` 类成员。

 示例 1：

```cuda


    template <template <typename...> class T,
              typename... P1,
              typename... P2>
    void bar1(const T<P1...>, const T<P2...>) {
        // ERROR, enclosing function has multiple parameter packs
        auto lambda = [] __device__ { return 10; };
    }

    template <template <typename...> class T,
              typename... P1,
              typename    T2>
    void bar2(const T<P1...>, T2) {
        // ERROR, for enclosing function, the parameter pack is not last in the template parameter list
        auto lambda = [] __device__ { return 10; };
    }

    template <typename T, T>
    void bar3() {
        // ERROR, for enclosing function, the second template parameter is not named
        auto lambda = [] __device__ { return 10; };
    }


```

 示例 2：

```cuda


    template <typename T>
    void bar4() {
        auto lambda1 = [] __device__ { return 10; };
    }

    class MyStruct {
        struct MyNestedStruct {};

        friend int main();
    };

    int main() {
        struct MyLocalStruct {};
        // ERROR, enclosing function for device lambda in bar4() is instantiated with a type local to main
        bar4<MyLocalStruct>();

        // ERROR, enclosing function for device lambda in bar4 is instantiated with a type
        //        that is a private member of a class
        bar4<MyStruct::MyNestedStruct>();
    }


```

10。对于 Microsoft Visual Studio 主机编译器，封闭函数必须具有外部链接。存在此限制是因为主机编译器不支持使用非外部链接函数的地址作为模板参数。 CUDA 编译器转换要求这些地址支持扩展 lambda。

11。对于 Microsoft Visual Studio 主机编译器，不应在 `if`` ``constexpr` 块的主体内定义扩展 lambda。

12。扩展 lambda 对捕获的变量有以下限制：

 - 在用于直接初始化表示扩展 lambda 的闭包类型的类类型的字段之前，该变量可以按值传递到发送到主机编译器的代码中的一系列辅助函数。但是，C++ 标准规定捕获的变量应该用于直接初始化闭包类型的字段。

 - 只能通过值捕获变量。

 - 如果数组维数大于 7，则无法捕获数组类型的变量。

 - 对于数组类型变量，首先默认初始化闭包类型的数组字段，然后再默认初始化每个数组元素从发送到主机编译器的代码中捕获的数组变量的相应元素进行复制分配。因此，数组元素类型在主机代码中必须既可默认构造又可复制分配。

 - 无法捕获作为可变参数包元素的函数参数。

- 捕获的变量类型不能是函数的本地变量，扩展 lambda 闭包类型或 `private` or `protected` 类成员除外。

 - 主机设备扩展 lambda 不支持初始化捕获。但是，它支持设备扩展 lambda，除非初始化程序是数组或 `std::initializer_list`.

 类型 - 扩展 lambda 的函数调用运算符不是 `constexpr`。扩展 lambda 的闭包类型不是文字类型。声明扩展 lambda 时，不能使用 `constexpr` 和 `consteval` 说明符。

 - 不能在词法嵌套在扩展 lambda 内的 `if-constexpr` 块内隐式捕获变量，除非该变量已在 `if-constexpr` 块外部隐式捕获或出现在扩展 lambda 的显式捕获中list.

 示例：

```cuda


    void host_function() {
        // CORRECT, an init-capture is allowed for an extended device-only lambda
        auto lambda1 = [x = 1] __device__ () { return x; };

        // ERROR, an init-capture is not allowed for an extended host-device lambda
        auto lambda2 = [x = 1] __host__ __device__ () { return x; };

        int a = 1;
        // ERROR, an extended __device__ lambda cannot capture variables by reference
        auto lambda3 = [&a] __device__ () { return a; };

        // ERROR, by-reference capture is not allowed for an extended device-only lambda
        auto lambda4 = [&x = a] __device__ () { return x; };

        struct MyStruct {};
        MyStruct s1;
        // ERROR, a type local to a function cannot be used in the type of a captured variable
        auto lambda6 = [s1] __device__ () { };

        // ERROR, an init-capture cannot be of type std::initializer_list
        auto lambda7 = [x = {11}] __device__ () { };

        std::initializer_list<int> b = {11,22,33};
        // ERROR, an init-capture cannot be of type std::initializer_list
        auto lambda8 = [x = b] __device__ () { };

        int  var     = 4;
        auto lambda9 = [=] __device__ {
            int result = 0;
            if constexpr(false) {
                //ERROR, An extended device-only lambda cannot first-capture 'var' in if-constexpr context
                result += var;
            }
            return result;
        };

        auto lambda10 = [var] __device__ {
            int result = 0;
            if constexpr(false) {
                // CORRECT, 'var' already listed in explicit capture list for the extended lambda
                result += var;
            }
            return result;
        };

        auto lambda11 = [=] __device__ {
            int result = var;
            if constexpr(false) {
                // CORRECT, 'var' already implicit captured outside the 'if-constexpr' block
                result += var;
            }
            return result;
        };
    }


```

13。解析函数时，CUDA 编译器会为函数中的每个扩展 lambda 分配一个计数器值。此计数器值用在传递给主机编译器的替换命名类型中。因此，函数中是否存在扩展 lambda 不应取决于 `__CUDA_ARCH__` 的特定值，也不应取决于 `__CUDA_ARCH__` 是否未定义。示例：

```cuda


    template <typename T>
    __global__ void kernel(T in) { in(); }

    __host__ __device__ void host_device_function() {
        // ERROR, the number and relative declaration order of
        //        extended lambdas depend on __CUDA_ARCH__
    #if defined(__CUDA_ARCH__)
        auto lambda1 = [] __device__ { return 0; };
        auto lambda2 = [] __host__ __device__ { return 10; };
    #endif
        auto lambda3 = [] __device__ { return 4; };
        kernel<<<1, 1>>>(lambda3);
    }


```

14。如上所述，CUDA 编译器将主机函数中定义的设备扩展 lambda 替换为命名空间范围中定义的占位符类型。占位符类型不会定义与原始 lambda 声明等效的 `operator()` 函数，除非特征 `__nv_is_extended_device_lambda_with_preserved_return_type()` 返回扩展 lambda 的闭包类型的 `true`。因此，尝试确定此类 lambda 的 `operator()` 函数的返回类型或参数类型可能会在主机代码中无法正常工作，因为主机编译器处理的代码在语义上与 CUDA 编译器处理的输入代码不同。但是，内省返回类型或参数类型`operator()`设备代码中的函数是可以接受的。请注意，此限制不适用于特征 `__nv_is_extended_device_lambda_with_preserved_return_type()` 返回 `true` 的主机或设备扩展 lambda。示例：

```cuda


    #include <cuda/std/type_traits>

    const char& getRef(const char* p) { return *p; }

    void foo() {
        auto lambda1 = [] __device__ { return "10"; };

        // ERROR, attempt to extract the return type of a device lambda in host code
        cuda::std::result_of<decltype(lambda1)()>::type xx1 = "abc";

        auto lambda2 = [] __host__ __device__ { return "10"; };

        // CORRECT, lambda2 represents a host-device extended lambda
        cuda::std::result_of<decltype(lambda2)()>::type xx2 = "abc";

        auto lambda3 = [] __device__ () -> const char* { return "10"; };

        // CORRECT, lambda3 represents a device extended lambda with preserved return type
        cuda::std::result_of<decltype(lambda3)()>::type xx2 = "abc";
        static_assert(cuda::std::is_same_v<cuda::std::result_of<decltype(lambda3)()>::type, const char*>);

        auto lambda4 = [] __device__ (char x) -> decltype(getRef(&x)) { return 0; };
        // lambda4's return type is not preserved because it references the operator()'s
        // parameter types in the trailing return type.
        static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(decltype(lambda4)));
    }


```

15。对于仅扩展设备的 lambda：

 - 仅在设备代码中支持 `operator()` 参数类型的自省。

 - 仅在设备代码中支持 `operator()` 返回类型的自省，除非特征函数 `__nv_is_extended_device_lambda_with_preserved_return_type()` 返回 `true`.

16。例如，如果将扩展 lambda 作为 `__global__` 函数的参数从主机传递到设备代码，则 lambda 主体中捕获变量的任何表达式都必须保持不变，无论是否定义了 `__CUDA_ARCH__` 宏以及它具有什么值。出现此限制是因为 lambda 的闭包类布局取决于编译器在处理 lambda 表达式时遇到捕获的变量的顺序。如果设备和主机编译之间的闭包类布局不同，程序可能会错误执行。示例：

```cuda


    __device__ int result;

    template <typename T>
    __global__ void kernel(T in) { result = in(); }

    void foo(void) {
        int x1 = 1;
        // ERROR, "x1" is only captured when __CUDA_ARCH__ is defined.
        auto lambda1 = [=] __host__ __device__ {
    #ifdef __CUDA_ARCH__
            return x1 + 1;
    #else
            return 10;
    #endif
        };
        kernel<<<1, 1>>>(lambda1);
    }


```

17。如前所述，CUDA 编译器将发送到主机编译器的代码中的仅扩展设备 lambda 表达式替换为占位符类型实例。占位符类型在主机代码中没有定义指针到函数的转换运算符；但是，设备代码中提供了转换运算符。请注意，此限制不适用于主机设备扩展 lambda。例子：

```cuda


    template <typename T>
    __global__ void kernel(T in) {
        int (*fp)(double) = in;
        fp(0); // CORRECT, conversion in device code is supported
        auto lambda1 = [](double) { return 1; };
    }

    void foo() {
        auto lambda_device      = [] __device__ (double) { return 1; };
        auto lambda_host_device = [] __host__ __device__ (double) { return 1; };
        kernel<<<1, 1>>>(lambda_device);
        kernel<<<1, 1>>>(lambda_host_device);

        // CORRECT, conversion for a __host__ __device__ lambda is supported in host code
        int (*fp1)(double) = lambda_host_device;

        // ERROR, conversion for a device lambda is not supported in host code
        int (*fp2)(double) = lambda_device;
    }


```

18. 如前所述，CUDA 编译器在发送到主机编译器的代码中用占位符类型实例替换扩展的仅设备或主机设备 lambda 表达式。该占位符类型可以定义 C++ 特殊成员函数，例如构造函数和析构函数。因此，对于 CUDA 前端编译器中的扩展 lambda 的闭包类型，某些标准 C++ 类型特征可能会与主机编译器中产生不同的结果。以下类型特征会受到影响： : `std::is_trivially_copyable`, `std::is_trivially_constructible`, `std::is_trivially_copy_constructible`, `std::is_trivially_move_constructible`, `std::is_trivially_destructible`。必须小心确保这些特征的结果不会在 `__global__`, `__device__`, `__constant__`, or `__managed__` 函数或变量模板的实例化中使用。示例：

```cuda


    #include <cstdio>
    #include <type_traits>

    template <bool b>
    void __global__ kernel() { printf("hi"); }

    template <typename T>
    void kernel_launch() {
        // ERROR, this kernel launch may fail, because CUDA frontend compiler and host compiler
        //        may disagree on the result of std::is_trivially_copyable_v trait on the
        //        closure type of the extended lambda
        kernel<std::is_trivially_copyable_v<T>><<<1,1>>>();
        cudaDeviceSynchronize();
    }

    int main() {
        int  x       = 0;
        auto lambda1 = [=] __host__ __device__ () { return x; };
        kernel_launch<decltype(lambda1)>();
    }


```

 CUDA 编译器将为 `1-12` 中描述的部分情况生成编译器诊断信息；对于 `13-17` 情况，不会生成任何诊断信息，但主机编译器可能无法编译生成的代码。





<span id="host-device-lambda-notes"></span>

### 5.3.8.5. 主机设备 Lambda 优化说明（Host-Device Lambda Optimization Notes）

 与仅限设备的 lambda 不同，可以从主机代码调用主机设备 lambda。如前所述，CUDA 编译器将主机代码中定义的扩展 lambda 表达式替换为命名占位符类型的实例。扩展主机设备 lambda 的占位符类型通过间接函数调用来调用原始 lambda 的 `operator()`。如果扩展 lambda 模式未激活，这些特征将始终返回 false。

间接函数调用的存在可能会导致主机编译器优化的扩展主机设备 lambda 小于隐式或显式的 lambda`__host__`仅有的。在后一种情况下，主机编译器可以轻松地将 lambda 主体内联到调用上下文中。但是，当遇到扩展的主机设备 lambda 时，主机编译器可能无法轻松内联原始 lambda 主体。





<span id="star-this-capture"></span>

### 5.3.8.6. `*this`按值捕获（`*this` Capture By-Value）

根据 C++11/C++14 规则，当 lambda 定义在非`static`类成员函数和 lambda 体指的是类成员变量，`this`类的指针必须通过值而不是引用的成员变量来捕获。如果 lambda 是在主机函数中定义的仅扩展设备或主机设备 lambda 并在 GPU 上执行，则访问 GPU 上引用的成员变量将导致运行时错误，如果`this`指针指向主机内存。

例子：

```cuda


    #include <cstdio>

    template <typename T>
    __global__ void foo(T in) { printf("value = %d\n", in()); }

    struct MyStruct {
        int var;

        __host__ __device__ MyStruct() : var(10) {};

        void run() {
            auto lambda1 = [=] __device__ {
                // reference to "var" causes the 'this' pointer (MyStruct*) to be captured by value
                return var + 1;
            };
            // Kernel launch fails at run time because 'this->var' is not accessible from the GPU
            foo<<<1, 1>>>(lambda1);
            cudaDeviceSynchronize();
        }
    };

    int main() {
        MyStruct s1;
        s1.run();
    }


```

C++17 通过引入新的方法解决了这个问题`*this`捕捉模式。在这种模式下，编译器复制由`*this`而不是捕获`this`按值指针。这`*this`捕获模式更详细地描述在<a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0018r3.html" class="reference external">P0018R3</a>.

CUDA编译器支持`*this`内定义的 lambda 捕获模式`__device__`和`__global__`函数和主机代码中定义的扩展的仅设备 lambda，当`--extended-lambda`使用标志。

这里是上面的例子修改后使用`*this`捕捉模式：

```cuda


    #include <cstdio>

    template <typename T>
    __global__ void foo(T in) { printf("\n value = %d", in()); }

    struct MyStruct {
        int var;
        __host__ __device__ MyStruct() : var(10) { };

        void run() {
            // note the "*this" capture specification
            auto lambda1 = [=, *this] __device__ {
                // reference to "var" causes the object denoted by '*this' to be captured by
                // value, and the GPU code will access 'copy_of_star_this->var'
                return var + 1;
            };
            // Kernel launch succeeds
            foo<<<1, 1>>>(lambda1);
            cudaDeviceSynchronize();
        }
    };

    int main() {
        MyStruct s1;
        s1.run();
    }


```

`*this`主机代码中定义的无注释 lambda 或扩展主机设备 lambda 不允许使用捕获模式，除非`*this`捕获由所选语言方言启用。以下是支持和不支持的用法示例：

```cuda


    struct MyStruct {
        int var;
        __host__ __device__ MyStruct() : var(10) { };

        void host_function() {
            // CORRECT, use in an extended device-only lambda
            auto lambda1 = [=, *this] __device__ { return var; };

            // Use in an extended host-device lambda
            // Error if *this capture not enabled by language dialect
            auto lambda2 = [=, *this] __host__ __device__ { return var; };

            // Use in an non-annotated lambda in host function
            // Error if *this capture not enabled by language dialect
            auto lambda3 = [=, *this]  { return var; };
        }

        __device__ void device_function() {
            // CORRECT, use in a lambda defined in a device-only function
            auto lambda1 = [=, *this] __device__ { return var; };

            // CORRECT, use in a lambda defined in a device-only function
            auto lambda2 = [=, *this] __host__ __device__ { return var; };

            // CORRECT, use in a lambda defined in a device-only function
            auto lambda3 = [=, *this]  { return var; };
        }

        __host__ __device__ void host_device_function() {
            // CORRECT, use in an extended device-only lambda
            auto lambda1 = [=, *this] __device__ { return var; };

            // Use in an extended host-device lambda
            // Error if *this capture not enabled by language dialect
            auto lambda2 = [=, *this] __host__ __device__ { return var; };

            // Use in an unannotated lambda in a host-device function
            // Error if *this capture not enabled by language dialect
            auto lambda3 = [=, *this]  { return var; };
        }
    };


```





<span id="lambda-argument-dependent-lookup"></span>

### 5.3.8.7. 参数相关查找 (ADL)（Argument Dependent Lookup (ADL)）

如前所述，CUDA 编译器在调用主机编译器之前将扩展 lambda 表达式替换为占位符类型。占位符类型的一个模板参数使用包含原始 lambda 表达式的函数的地址。这可能会导致额外的命名空间参与<a href="https://en.cppreference.com/w/cpp/language/adl.html" class="reference external">参数相关查找 (ADL)</a>对于其参数类型涉及扩展 lambda 表达式的闭包类型的任何主机函数调用。因此，主机编译器可能会选择不正确的函数。

例子：

```cuda


    namespace N1 {

    struct MyStruct {};

    template <typename T>
    void my_function(T);

    }; // namespace N1

    namespace N2 {

    template <typename T>
    int my_function(T);

    template <typename T>
    void run(T in) { my_function(in); }

    } // namespace N2

    void bar(N1::MyStruct in) {
        // For extended device-only lambda, the code sent to the host compiler is replaced with
        // the placeholder type instantiation expression
        //    ' __nv_dl_wrapper_t< __nv_dl_tag<void (*)(N1::MyStruct in),(&bar),1> > { }'
        //
        // As a result, the namespace 'N1' participates in ADL lookup of the
        // call to "my_function()" in the body of N2::run, causing ambiguity.
        auto lambda1 = [=] __device__ { };
        N2::run(lambda1);
    }


```

在上面的示例中，CUDA 编译器将扩​​展 lambda 替换为涉及以下内容的占位符类型`N1`命名空间。因此，`N1`命名空间参与 ADL 查找`my_function(in)`在体内`N2::run()`，由于发现多个重载候选项而导致主机编译失败：`N1::my_function`和`N2::my_function`.







<span id="id6"></span>

## 5.3.9. 多态函数包装器（Polymorphic Function Wrappers）

The `nvfunctional`header 提供了多态函数包装类模板，`nvstd::function`。此类模板的实例可以存储、复制和调用任何可调用目标，例如 lambda 表达式。`nvstd::function`可以在主机和设备代码中使用。

例子：

```cuda


    #include <nvfunctional>

    __host__            int host_function()        { return 1; }
    __device__          int device_function()      { return 2; }
    __host__ __device__ int host_device_function() { return 3; }

    __global__ void kernel(int* result) {
        nvstd::function<int()> fn1 = device_function;
        nvstd::function<int()> fn2 = host_device_function;
        nvstd::function<int()> fn3 = [](){ return 10; };
        *result                    = fn1() + fn2() + fn3();
    }

    __host__ __device__ void host_device_test(int* result) {
        nvstd::function<int()> fn1 = host_device_function;
        nvstd::function<int()> fn2 = [](){ return 10; };
        *result                    = fn1() + fn2();
    }

    __host__ void host_test(int* result) {
        nvstd::function<int()> fn1 = host_function;
        nvstd::function<int()> fn2 = host_device_function;
        nvstd::function<int()> fn3 = [](){ return 10; };
        *result                    = fn1() + fn2() + fn3();
    }


```

------------------------------------------------------------------------

无效情况：

- 实例`nvstd::function`主机代码中不能使用 a 的地址进行初始化`__device__`函数或函子，其`operator()` is a `__device__`功能。

- 同样，实例`nvstd::function`设备代码中不能使用a的地址进行初始化`__host__`函数或函子，其`operator()` is a `__host__`功能。

- `nvstd::function`实例无法在运行时从主机代码传递到设备代码（反之亦然）。

- `nvstd::function`不能用在 a 的参数类型中`__global__`函数如果`__global__`函数是从主机代码启动的。

无效案例举例：

```cuda


    #include <nvfunctional>

    __device__ int device_function() { return 1; }
    __host__   int host_function() { return 3; }
    auto       lambda_host  = [] { return 0; };

    __global__ void k() {
        nvstd::function<int()> fn1 = host_function; // ERROR, initialized with address of __host__ function
        nvstd::function<int()> fn2 = lambda_host;   // ERROR, initialized with address of functor with
                                                    //        __host__ operator() function
    }

    __global__ void kernel(nvstd::function<int()> f1) {}

    void foo(void) {
        auto lambda_device = [=] __device__ { return 1; };

        nvstd::function<int()> fn1 = device_function; // ERROR, initialized with address of __device__ function
        nvstd::function<int()> fn2 = lambda_device;   // ERROR, initialized with address of functor with
                                                      //        __device__ operator() function
        kernel<<<1, 1>>>(fn2);                        // ERROR, passing nvstd::function from host to device
    }


```

------------------------------------------------------------------------

`nvstd::function`定义在`nvfunctional`标头如下：

```cuda


    namespace nvstd {

    template <typename RetType, typename ...ArgTypes>
    class function<RetType(ArgTypes...)> {
    public:
        // constructors
        __device__ __host__ function() noexcept;
        __device__ __host__ function(nullptr_t) noexcept;
        __device__ __host__ function(const function&);
        __device__ __host__ function(function&&);

        template<typename F>
        __device__ __host__ function(F);

        // destructor
        __device__ __host__ ~function();

        // assignment operators
        __device__ __host__ function& operator=(const function&);
        __device__ __host__ function& operator=(function&&);
        __device__ __host__ function& operator=(nullptr_t);
        template<typename F>
        __device__ __host__ function& operator=(F&&);

        // swap
        __device__ __host__ void swap(function&) noexcept;

        // function capacity
        __device__ __host__ explicit operator bool() const noexcept;

        // function invocation
        __device__ RetType operator()(ArgTypes...) const;
    };

    // null pointer comparisons
    template <typename R, typename... ArgTypes>
    __device__ __host__
    bool operator==(const function<R(ArgTypes...)>&, nullptr_t) noexcept;

    template <typename R, typename... ArgTypes>
    __device__ __host__
    bool operator==(nullptr_t, const function<R(ArgTypes...)>&) noexcept;

    template <typename R, typename... ArgTypes>
    __device__ __host__
    bool operator!=(const function<R(ArgTypes...)>&, nullptr_t) noexcept;

    template <typename R, typename... ArgTypes>
    __device__ __host__
    bool operator!=(nullptr_t, const function<R(ArgTypes...)>&) noexcept;

    // specialized algorithms
    template <typename R, typename... ArgTypes>
    __device__ __host__
    void swap(function<R(ArgTypes...)>&, function<R(ArgTypes...)>&);

    } // namespace nvstd


```





<span id="language-restrictions"></span>

## 5.3.10. C/C++ 语言限制（C/C++ Language Restrictions）



<span id="id7"></span>

### 5.3.10.1. 不支持的功能（Unsupported Features）

- 设备代码中不支持运行时类型信息 (RTTI) 和异常：

  - `typeid` 关键字

  - `dynamic_cast` 关键字

  - `try/catch/throw` 设备代码中不支持关键字

- `long`` ``double`。

- 任何平台都不支持三字符组。 Windows 上不支持二合字母。

 - 用户定义的 `operator`` ``new`, `operator`` ``new[]`, `operator`` ``delete`, or `operator`` ``delete[]` 不能用于替换编译器提供的相应内置函数，并且在主机和设备上都被视为未定义行为。





### 5.3.10.2. 命名空间保留（Namespace Reservations）

除非另有说明，否则将定义添加到顶级命名空间 `cuda::`, `nv::`, or `cooperative_groups::` 或其中的任何嵌套命名空间他们，是未定义的行为。我们允许 `cuda::` 作为子命名空间，如下所示：

 示例：

```cuda


    namespace cuda {   // same for "nv" and "cooperative_groups" namespaces

    struct foo;        // ERROR, class declaration in the "cuda" namespace

    void bar();        // ERROR, function declaration in the "cuda" namespace

    namespace utils {} // ERROR, namespace declaration in the "cuda" namespace

    } // namespace cuda


```

```cuda


    namespace utils {
    namespace cuda {

    // CORRECT, namespace "cuda" may be used nested within a non-reserved namespace
    void bar();

    } // namespace cuda
    } // namespace utils

    // ERROR, Equivalent to adding symbols to namespace "cuda" at global scope
    using namespace utils;


```





<span id="pointers"></span>

### 5.3.10.3. 指针和内存地址（Pointers and Memory Addresses）

 指针取消引用 (`*pointer`, `pointer->member`, `pointer[0]`) 仅允许在关联内存所在的同一执行空间中。以下情况会导致未定义的行为，最常见的是分段错误和应用程序终止。

- 取消引用指向 <a href="../02-programming-gpus/writing-simt-kernels.html#_2-3-3-1-global-memory" class="reference internal"> 全局内存的指针</a>, <a href="../02-programming-gpus/writing-simt-kernels.html#_2-3-3-2-shared-memory" class="reference internal">共享内存</a>, or <a href="../02-programming-gpus/writing-simt-kernels.html#_2-3-3-5-constant-memory" class="reference internal"> 主机上的常量内存</a>。

- 取消引用指向设备中主机内存的指针代码。

 以下限制适用于函数：

 - 不允许在主机代码中获取 `__device__` 函数的地址。

 - 在主机代码中获取的 `__global__` 函数的地址不能在设备代码中使用。同样，在设备代码中获取的 `__global__` 函数的地址不能在主机代码中使用。

 通过 `__device__` or `__constant__` 获得的 `cudaGetSymbolAddress()` 变量的地址（如 <a href="cpp-language-extensions.html#memory-space-specifiers" class="reference internal"> 内存空间说明符</a> 部分中所述）只能在主机中使用代码.





### 5.3.10.4. Variables（Variables）



<span id="id8"></span>

#### 5.3.10.4.1. 局部变量（Local Variables）

The `__device__`, `__tile__`, `__shared__`, `__managed__`， 和`__constant__`内存空间说明符不允许在非`extern`在主机上执行的函数内的变量声明。

示例：

```cuda


    __host__ void host_function() {
        int x;                   // CORRECT, __host__ variable
        __device__   int y;      // ERROR,   __device__ variable declaration within a host function
        __tile__     int z;      // ERROR,   __tile__ variable declaration within a host function
        __shared__   int w;      // ERROR,   __shared__ variable declaration within a host function
        __managed__  int h;      // ERROR,   __managed__ variable  declaration within a host function
        __constant__ int i;      // ERROR,   __constant__ variable declaration within a host function
        extern __device__ int j; // CORRECT, extern __device__ variable
    }


```

The `__device__`, `__tile__`, `__constant__`， 和`__managed__`变量声明中不允许使用内存空间说明符`extern`也不`static`在设备上执行的函数内。

```cuda


    __device__ void device_function() {
        int x;                   // CORRECT, __device__ variable
        __constant__      int y; // ERROR,   __constant__ variable declaration within a device function
        __managed__       int z; // ERROR,   __managed__ variable  declaration within a device function
        extern __device__ int k; // CORRECT, extern __device__ variable
    }


```

另请参阅<a href="#static-variables" class="reference internal">静态变量</a>部分。





<span id="const-variables"></span>

#### 5.3.10.4.2. `const`-限定变量（`const`-qualified Variables）

A `const`- 没有内存空间注释的限定变量（`__device__`, `__tile__`, or `__constant__`) 在全局、命名空间或类范围声明的被视为主变量。设备代码不能包含引用或获取变量的地址。

该变量可以直接在设备代码中使用，如果

- 在使用前已用常量表达式对其进行了初始化，

- 类型不是`volatile`- 合格，并且

- 它具有以下类型之一：

- 内置积分型，或

- 内置浮点类型，除非主机编译器是 Microsoft Visual Studio。

从C++14开始，推荐使用`constexpr` or `inline`` ``constexpr`(C++17) 变量而不是`const`- 合格的。`constexpr`变量不受相同类型的限制，可以直接在设备代码中使用。

`__managed__`变量不支持`const`- 合格类型。

示例：

```cuda


    const            int   ConstVar          = 10;
    const            float ConstFloatVar     = 5.0f;
    inline constexpr float ConstexprFloatVar = 5.0f; // C++17

    struct MyStruct {
        static const            int   ConstVar          = 20;
    //  static const             float ConstFloatVar     = 5.0f; // ERROR, static const variables cannot be float
        static inline constexpr float ConstexprFloatVar = 5.0f; // CORRECT
    };

    extern const int ExternVar;

    __device__ void foo() {
        int array1[ConstVar];                     // CORRECT
        int array2[MyStruct::ConstVar];           // CORRECT

        const     float var1 = ConstFloatVar;     // CORRECT, except when the host compiler is Microsoft Visual Studio.
        constexpr float var2 = ConstexprFloatVar; // CORRECT
    //  int             var3 = ExternVar;          // ERROR, "ExternVar" is not initialized with a constant expression
    //  int&            var4 = ConstVar;           // ERROR, reference to host variable
    //  int*            var5 = &ConstVar;          // ERROR, address of host variable
    }


```

请参阅示例<a href="https://godbolt.org/z/eWG8KxK94" class="reference external">编译器资源管理器</a>.





<span id="volatile-qualifier"></span>

#### 5.3.10.4.3. `volatile`-限定变量（`volatile`-qualified Variables）



笔记

The `volatile`支持关键字以保持与 ISO C++ 的兼容性。然而，即使有的话，也很少有<a href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1152r0.html#prop" class="reference external">剩余的未弃用用途</a>适用于 GPU。



读取和写入`volatile`- 限定对象不是原子的，并且被编译成一个或多个<a href="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#volatile-operation" class="reference external">易失指令</a>不保证：

- 内存操作的顺序，或

- 硬件执行的内存操作数与 PTX 指令数相匹配。

 在平铺代码中，`volatile` 关键字对内存访问的行为没有影响。

CUDA C++ `volatile` 不适合：

- **线程间同步**：通过 <a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html" class="reference external">cuda::atomic_ref</a>, <a href="https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic.html" class="reference external">cuda::atomic</a>, or <a href="cpp-language-extensions.html#atomic-functions" class="reference internal"> 原子操作代替 </a>。

 原子内存操作提供线程间同步保证，并提供比 `volatile` 操作更好的性能。但是，CUDA C++ `volatile` 操作不提供任何线程间同步保证，因此不适合此目的。以下示例显示如何使用原子操作在两个线程之间传递消息。



 cuda::atomic_ref




  <table id="table-cuda-atomic-ref" class="table">
  <colgroup>
  <col style="width: 100%" />
  </colgroup>
  <tbody>
  <tr class="row-odd">
  <td>
  <pre><code>#include &lt;cuda/atomic&gt;
  &#10;__global__ void kernel(int* flag, int* data) {
      cuda::atomic_ref&lt;int, cuda::thread_scope_device&gt; atomic_ref{*flag};
      if (threadIdx.x == 0) {
          // Consumer: blocks until flag is set by producer, then reads data
          while(atomic_ref.load(cuda::memory_order_acquire) == 0)
              ;
          if (*data != 42)
              __trap(); // Errors if wrong data read
      }
      else if (threadIdx.x == 1) {
          // Producer: writes data then sets flag
          *data = 42;
          atomic_ref.store(1, cuda::memory_order_release);
      }
  }</code></pre>
  </td>
  </tr>
  </tbody>
  </table>





 cuda::atomic




  <table id="table-cuda-atomic" class="table">
  <colgroup>
  <col style="width: 100%" />
  </colgroup>
  <tbody>
  <tr class="row-odd">
  <td>
  <pre><code>#include &lt;cuda/atomic&gt;
  &#10;__global__ void kernel(cuda::atomic&lt;int, cuda::thread_scope_device&gt;* flag, int* data) {
      if (threadIdx.x == 0) {
          // Consumer: blocks until flag is set by producer, then reads data
          while(flag-&gt;load(cuda::memory_order_acquire) == 0)
              ;
          if (*data != 42)
              __trap(); // Errors if wrong data read
      }
      else if (threadIdx.x == 1) {
          // Producer: writes data then sets flag
          *data = 42;
          flag-&gt;store(1, cuda::memory_order_release);
      }
  }</code></pre>
  </td>
  </tr>
  </tbody>
  </table>





 原子函数（`atomicAdd` 和`atomicExch`)




  <table id="table-atomic-functions" class="table">
  <colgroup>
  <col style="width: 100%" />
  </colgroup>
  <tbody>
  <tr class="row-odd">
  <td>
  <pre><code>__global__ void kernel(int* flag, int* data) {
      if (threadIdx.x == 0) {
          // Consumer: blocks until flag is set by producer, then reads data
          while(atomicAdd(flag, 0) == 0)
              ;                // Load with Relaxed Read-Modify-Write
          __threadfence();     // SequentiallyConsistent fence
          if (*data != 42)
              __trap();        // Errors if wrong data read
      } else if (threadIdx.x == 1) {
          // Producer: writes data then sets flag
          *data = 42;
          __threadfence();     // SequentiallyConsistent fence
          atomicExch(flag, 1); // Store with Relaxed Read-Modify-Write
      }
  }</code></pre>
  </td>
  </tr>
  </tbody>
  </table>







- **内存映射 IO** (MMIO)：通过内联 PTX 使用 <a href="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mmio-operation" class="reference external">PTX MMIO 操作</a>。

 PTX MMIO 操作严格保留执行的内存访问次数。但是，CUDA C++ `volatile` 操作不会保留所执行的内存访问次数，并且可能会以不确定的方式执行比请求的更多或更少的访问。这使得它们不适合 MMIO。以下示例显示如何使用 PTX MMIO 操作读取和写入寄存器。

  ```cuda


      __global__ void kernel(int* mmio_reg0, int* mmio_reg1) {
          // Write to MMIO register:
          int value = 13;
          asm volatile("st.relaxed.mmio.sys.u32 [%0], %1;"
              :
              : "l"(mmio_reg0), "r"(value) : "memory");

          // Read MMIO register:
          asm volatile("ld.relaxed.mmio.sys.u32 %0, [%1];"
              : "=r"(value)
              : "l"(mmio_reg1) : "memory");

          if (value != 42)
              __trap(); // Errors if wrong data read
      }


  ```





<span id="id9"></span>

#### 5.3.10.4.4. `static` 变量在设备函数中允许使用（`static` Variables）

`static` 局部变量。

A 不同的静态变量用于封闭函数的每个执行空间。例如，

- A `__host__`` ``__device__` 函数具有一份用于主机执行的静态变量副本和一份用于设备执行的静态变量副本。

- A `__tile__`` ``__device__` 函数具有一份用于平铺执行的副本和一份用于 SIMT 执行的副本。

 如果函数具有 `__host__` 执行空间说明符，则具有显式内存空间的 `static` 变量（例如 `static`` ``__device__/__tile__/__constant__/__shared__/__managed__`）为仅当定义了 `__CUDA_ARCH__` 时才允许。

函数作用域 `static` 变量的合法和非法使用示例如下所示。

```cuda


    struct TrivialStruct {
        int x;
    };

    struct NonTrivialStruct {
        __device__ NonTrivialStruct(int x) {}
    };

    __device__ void device_function(int x) {
        static int v1;              // CORRECT, implicit __device__ memory space specifier
        static int v2 = 11;         // CORRECT, implicit __device__ memory space specifier
    //  static int v3 = x;           // ERROR, dynamic initialization is not allowed

        static __managed__  int v4; // CORRECT, explicit
        static __device__   int v5; // CORRECT, explicit
        static __constant__ int v6; // CORRECT, explicit
        static __shared__   int v7; // CORRECT, explicit

        static TrivialStruct    s1;     // CORRECT, implicit __device__ memory space specifier
        static TrivialStruct    s2{22}; // CORRECT, implicit __device__ memory space specifier
    //  static TrivialStruct    s3{x};   // ERROR, dynamic initialization is not allowed
    //  static NonTrivialStruct s4{3};   // ERROR, dynamic initialization is not allowed
    }


```

请参阅 <a href="https://godbolt.org/z/TdYKaTq3f" class="reference external"> 编译器资源管理器上的示例</a>.

------------------------------------------------------------------------

```cuda


    __host__ __device__ void host_device_function() {
        static            int v1; // CORRECT, implicit __device__ memory space specifier
    //  static __device__ int v2;  // ERROR, __device__-only variable inside a host-device function
    #ifdef __CUDA_ARCH__
        static __device__ int v3; // CORRECT, declaration is only visible during device compilation
    #else
        static int v4;            // CORRECT, declaration is only visible during host compilation
    #endif
    }


```

请参阅 <a href="https://godbolt.org/z/18qhjn8P1" class="reference external"> 编译器上的示例Explorer</a>.

------------------------------------------------------------------------

```cuda


    #include <cassert>

    __host__ __device__ int host_device_function() {
        static int v = 0;
        v++;
        return v;
    }

    __global__ void kernel() {
        int ret = host_device_function(); // v = 1
        assert(ret == 4);                 // FAIL
    }

    int main() {
        host_device_function();           // v = 1
        host_device_function();           // v = 2
        int ret = host_device_function(); // v = 3
        assert(ret == 3);                 // OK
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }


```

参见<a href="https://godbolt.org/z/Wqo9WjvYY" class="reference external">上的示例编译器资源管理器</a>.





<span id="id10"></span>

#### 5.3.10.4.5. `extern`变量（`extern` Variables）

在<a href="../02-programming-gpus/nvcc.html#nvcc-separate-compilation" class="reference internal">整个程序编译模式下编译时</a>, `__device__`, `__tile__`, `__shared__`, `__managed__`，并且不能使用`__constant__`关键字通过外部链接定义`extern`变量。此限制也适用于 <a href="../02-programming-gpus/nvcc.html#nvcc-separate-compilation" class="reference internal"> 单独编译模式 </a> 对于 `__tile__` 变量。

 唯一的例外是动态分配的 `__shared__` 变量，如 <a href="../02-programming-gpus/writing-simt-kernels.html#_2-3-3-2-2-动态分配-shared-memory-dynamic-allocation-of-shared-memory" class="reference internal"> 共享内存的动态分配</a> 中所述节.

```cuda


    __device__        int x; // OK
    extern __device__ int y; // ERROR in whole program compilation mode
    extern __shared__ int z; // OK


```







<span id="id11"></span>

### 5.3.10.5. 函数（Functions）



<span id="function-recursion"></span>

#### 5.3.10.5.1. 递归（Recursion）

`__global__`, `__tile_global__`和`__tile__`函数不支持递归，而`__device__`和`__host__`` ``__device__`函数没有这样的限制。





<span id="id12"></span>

#### 5.3.10.5.2. 外部链接（External Linkage）

具有外部链接的设备变量或函数需要<a href="../02-programming-gpus/nvcc.html#nvcc-separate-compilation" class="reference internal">单独编译模式</a>跨多个翻译单元。

在单独编译模式下，如果要求`__device__` or `__global__`函数定义存在于特定的翻译单元中，那么该函数的参数和返回类型必须在该翻译单元中完整。该概念也称为单一定义规则使用或 ODR 使用。

 示例：

```cuda


    //first.cu:
    struct S;                   // forward declaration
    __device__ void foo(S);     // ERROR, type 'S' is an incomplete type
    __device__ auto* ptr = foo; // ODR-use, address taken

    int main() {}


```

```cuda


    //second.cu:
    struct S {};               // struct definition
    __device__ void foo(S) {}  // function definition


```

```console


    # compiler invocation
    $ nvcc -std=c++14 -rdc=true first.cu second.cu -o prog
    nvlink error   : Prototype doesn't match for '_Z3foo1S' in '/tmp/tmpxft_00005c8c_00000000-18_second.o',
                     first defined in '/tmp/tmpxft_00005c8c_00000000-18_second.o'
    nvlink fatal   : merge_elf failed


```





#### 5.3.10.5.3. 形式参数（Formal Parameters）

The `__device__`, `__tile__`, `__shared__`, `__managed__` 和 `__constant__` 内存空间说明符不允许用于形式参数。

```cuda


    void device_function1(__device__ int x) { } // ERROR, __device__ parameter
    void device_function2(__shared__ int x) { } // ERROR, __shared__ parameter


```





<span id="id13"></span>

#### 5.3.10.5.4. `__global__` 函数参数（`__global__` Function Parameters）

A `__global__` or `__tile_global__` 函数具有以下内容限制：

 - 它不能有可变数量的参数，即 C 省略语法 `...` 和 `va_list` 类型。允许使用 C++11 可变参数模板，但须遵守 <a href="#cpp11-variadic-template" class="reference internal">__global__ 可变参数模板</a> 部分中所述的限制。

- 函数参数通过 <a href="device-callable-apis.html#constant-memory" class="reference internal"> 常量内存 </a> 传递到设备，其总大小限制为 32,764 字节。

-函数参数不能是 `std::initializer_list`.

 类型 - 多态类参数 (`virtual`) 被视为未定义行为。

 - 允许使用 Lambda 表达式和闭包类型，但要遵守 <a href="#lambda-expressions-global" class="reference internal">Lambda 表达式和 __global__ 函数参数 </a> 部分中描述的限制。在

- For `__tile_global__` 函数中，函数参数不能是按值传递类、结构或联合。





<span id="global-function-arguments"></span>

#### 5.3.10.5.5. `__global__` 函数参数传递（`__global__` Function Arguments Passing）

启动 `__global__` 函数时<a href="../02-programming-gpus/intro-to-cuda-cpp.html#intro-cpp-launching-kernels" class="reference internal">从设备代码</a>，每个参数必须是普通可复制和普通可破坏的。

当从主机代码启动 `__global__` 函数时，每个参数类型可以是不可普通复制或不可普通破坏的。然而，这些类型的处理并不遵循标准 C++ 模型，如下所述。用户代码必须保证这个工作流程不会影响程序的正确性。该工作流程在两个方面与标准 C++ 不同：

1. **原始内存复制而不是复制构造函数调用**

 CUDA 运行时通过复制原始内存内容（最终使用 `__global__`）将内核参数传递给 `memcpy` 函数。如果参数不可简单复制并提供用户定义的复制构造函数，则在主机到设备复制中会跳过调用的操作和副作用。

 示例：

```cuda


    #include <cassert>

    struct MyStruct {
        int  value = 1;
        int* ptr;

        MyStruct() = default;

        __host__ __device__ MyStruct(const MyStruct&) { ptr = &value; }
    };

    __global__ void device_function(MyStruct my_struct) {
        // this assert fails because "my_struct" is obtained by copying
        // the raw memory content and the copy constructor is skipped.
        assert(my_struct.ptr == &my_struct.value); // FAIL
    }

    void host_function(MyStruct my_struct) {
        assert(my_struct.ptr == &my_struct.value); // CORRECT
    }

    int main() {
        MyStruct my_struct;
        host_function(my_struct);
        device_function<<<1, 1>>>(my_struct); // copy constructor invoked in the host-side only
        cudaDeviceSynchronize();
    }


```

 请参阅 <a href="https://godbolt.org/z/xhqe16dec" class="reference external">Compiler Explorer</a>.

2 上的示例。  **可以在** `__global__` **函数完成之前调用析构函数**

 内核启动与主机执行异步。因此，如果 `__global__` 函数参数具有重要的析构函数，则析构函数甚至可以在 `__global__` 函数完成执行之前在主机代码中执行。这可能会破坏析构函数具有副作用的程序。

 示例：

```cuda


    #include <cassert>

    __managed__ int var = 0;

    struct MyStruct {
        __host__ __device__ ~MyStruct() { var = 3; }
    };

    __global__ void device_function(MyStruct my_struct) {
        assert(var == 0); // FAIL, MyStruct::~MyStruct() sets the value to 3
    }

    int main() {
        MyStruct my_struct;
        // GPU kernel execution is asynchronous with host execution.
        // As a result, MyStruct::~MyStruct() could be executed before
        // the kernel finishes executing.
        device_function<<<1, 1>>>(my_struct);
        cudaDeviceSynchronize();
    }


```

 请参阅 <a href="https://godbolt.org/z/cn6Y5W6zs" class="reference external"> 编译器资源管理器中的示例</a>.







<span id="id14"></span>

### 5.3.10.6. Classes（Classes）



<span id="id15"></span>

#### 5.3.10.6.1. 类类型变量（Class-type Variables）

A 具有 `__device__`, `__tile__`, `__constant__`, `__managed__` or `__shared__` 内存空间的变量定义不能具有非空的类类型构造函数或非空析构函数。如果类类型的构造函数很简单或者在翻译单元中的某个点满足以下所有条件，则该构造函数被视为空：

 - 构造函数已定义。

- 构造函数没有参数、空的初始值设定项列表和空的复合语句函数体。

- 它的类没有`virtual`功能，`virtual`基类，或非`static`数据成员初始值设定项。

- 所有基类的默认构造函数都可以被认为是空的。

- 对于所有非`static`如果类的数据成员属于类类型（或其数组），则默认构造函数可以被视为空。

如果类的析构函数很简单或者在翻译单元中的某个点满足以下所有条件，则该类的析构函数被认为是空的：

- 析构函数已定义。

- 析构函数体是一个空的复合语句。

- 它的类没有`virtual`函数或`virtual`基类。

- 所有基类的析构函数都可以被认为是空的。

- 对于所有非`static`如果类的数据成员属于类类型（或其数组），则析构函数可以被认为是空的。





<span id="id16"></span>

#### 5.3.10.6.2. 数据成员（Data Members）

The `__device__`, `__tile__`, `__shared__`, `__managed__`和`__constant__`不允许使用内存空间说明符`class`, `struct`， 和`union`数据成员。

仅有的`static`支持在编译时评估的数据成员，例如<a href="#const-variables" class="reference internal">const 限定的</a>和`constexpr`变量。

```cuda


    struct MyStruct {
       static inline constexpr int value1 = 10; // C++17
       static constexpr        int value2 = 10; // C++11
       static const            int value3 = 10;
    // static                  int value4; // ERROR
    };


```





<span id="id17"></span>

#### 5.3.10.6.3. 职能成员（Function Members）

`__global__`和`__tile_global__`函数不能是 a 的成员`struct`, `class`, or `union`.

A `__global__` or `__tile_global__`函数允许在`friend`声明，但不能定义。

例子：

```cuda


    struct MyStruct {
        friend __global__ void f();   // CORRECT, friend declaration only

    //  friend __global__ void g() {} // ERROR, friend definition
    };


```

请参阅示例<a href="https://godbolt.org/z/rv6cP3b9j" class="reference external">编译器资源管理器</a>.





<span id="compiler-generated-functions"></span>

#### 5.3.10.6.4. 隐式声明和非虚拟显式默认函数（Implicitly-Declared and Non-Virtual Explicitly-Defaulted functions）

隐式声明的特殊成员函数是编译器在用户未声明时为类声明的函数；显式默认函数是用户声明但用其标记的函数`=`` ``default`。隐式声明或显式默认的特殊成员函数是默认构造函数、复制构造函数、移动构造函数、复制赋值运算符、移动赋值运算符和析构函数。

Let `F`表示一个非`virtual`在其第一个声明中隐式声明或显式默认的函数。执行空间说明符`F`是调用它的所有函数的执行空间说明符的并集。请注意，对于此分析，`__global__`调用者将被视为`__device__`呼叫者。例如：

```cuda


    class Base {
        int x;
    public:
        __host__ __device__ Base() : x(10) {}
    };

    class Derived : public Base {
        int y;
    };

    class Other: public Base {
        int z;
    };

    __device__ void foo() {
        Derived D1;
        Other D2;
    }

    __host__ void bar() {
        Other D3;
    }


```

在这种情况下，隐式声明的构造函数`Derived::Derived()`将被视为`__device__`函数，因为它仅从`__device__`功能`foo()`。隐式声明的构造函数`Other::Other()`将被视为`__host__`` ``__device__`函数，因为它是从两个 a 调用的`__device__`功能`foo()`和一个`__host__`功能`bar()`.

另外，如果`F`是一个隐式声明的`virtual`函数（例如，`virtual`析构函数），每个虚函数的执行空间`D`被覆盖`F`被添加到执行空间集中`F` if `D`没有隐式声明。

例如：

```cuda


    struct Base1 {
        virtual __host__ __device__ ~Base1() {}
    };

    struct Derived1 : Base1 {}; // implicitly-declared virtual destructor
                                // ~Derived1() has __host__ __device__  execution space specifiers

    struct Base2 {
        virtual __device__ ~Base2() = default;
    };

    struct Derived2 : Base2 {}; // implicitly-declared virtual destructor
                                // ~Derived2() has __device__ execution space specifiers


```





<span id="id18"></span>

#### 5.3.10.6.5. 多态类（Polymorphic Classes）

多态类，即具有`virtual`从其他多态类派生的函数或具有多态数据成员的函数受到以下限制：

- 将多态对象从设备复制到主机或从主机复制到设备，包括`__global__`函数参数是未定义的行为。

- 被覆盖的执行空间`virtual`函数必须与基类中函数的执行空间匹配。

例子：

```cuda


    struct MyClass {
        virtual __host__ __device__ void f() {}
    };

    __global__ void kernel(MyClass my_class) {
        my_class.f(); // undefined behavior
    }

    int main() {
        MyClass my_class;
        kernel<<<1, 1>>>(my_class);
        cudaDeviceSynchronize();
    }


```

请参阅示例<a href="https://godbolt.org/z/To39sGTrW" class="reference external">编译器资源管理器</a>.

------------------------------------------------------------------------

```cuda


    struct BaseClass {
        virtual __host__ __device__ void f() {}
    };

    struct DerivedClass : BaseClass {
        __device__ void f() override {} // ERROR
    };


```

请参阅示例<a href="https://godbolt.org/z/xfKhEGfdG" class="reference external">编译器资源管理器</a>.





<span id="windows-specific"></span>

#### 5.3.10.6.6. Windows 特定的类布局（Windows-Specific Class Layout）

CUDA 编译器遵循 IA64 ABI 进行类布局，而 Microsoft Visual Studio 则不然。这可以防止主机和设备代码之间按位复制特殊对象，如下所述。

Let `T` 表示指向成员类型的指针，或满足以下任一条件的类类型：

- `T` is a <a href="#polymorphic-classes" class="reference internal"> 多态类</a>

- `T` 具有多个继承，具有多个直接或间接 <a href="#class-type-variables" class="reference internal"> 空基类</a>.

 - 所有直接和间接基类`B` 是 <a href="#class-type-variables" class="reference internal">empty</a>，第一个字段 `F` of `T` 的类型在其定义中使用 `B`，这样 `B` 在 `F`.

 定义中的偏移量 0 处布局 `T` 类型的类，其基类为使用 Microsoft Visual Studio 编译时，类型 `T` 或具有 `T` 类型的数据成员可能在主机和设备之间具有不同的类布局和大小。

 将此类对象从设备复制到主机或从主机复制到设备（包括 `__global__` 函数参数）是未定义的行为。







<span id="id19"></span>

### 5.3.10.7. Templates（Templates）

A 类型不能是用作 `__global__` 函数或 `__device__/__constant__` 变量 (C++14) 的模板参数，如果满足以下任一条件：

 - 该类型在 `__host__` or `__host__`` ``__device__` 函数范围内定义。

 - 该类型未命名，例如匿名结构或 lambda 表达式，除非该类型是 `__device__` or `__global__` 的本地类型函数。

- 该类型是具有 `private` or `protected` 的类成员，除非该类是 `__device__` or `__global__` 函数的本地类。

- 该类型是由上述任何类型组合而成。

 示例：

```cuda


    template <typename T>
    __global__ void kernel() {}

    template <typename T>
    __device__ int device_var; // C++14

    struct {
        int v;
    } unnamed_struct;

    void host_function() {
        struct LocalStruct {};
    //  kernel<LocalStruct><<<1, 1>>>(); // ERROR, LocalStruct is defined within a host function
        int data = 4;
    //  cudaMemcpyToSymbol(device_var<LocalStruct>, &data, sizeof(data)); // ERROR, same as above

        auto lambda = [](){};
    //  kernel<decltype(lambda)><<<1, 1>>>();         // ERROR, unnamed type
    //  kernel<decltype(unnamed_struct)><<<1, 1>>>(); // ERROR, unnamed type
    }

    class MyClass {
    private:
        struct PrivateStruct {};
    public:
        static void launch() {
    //      kernel<PrivateStruct><<<1, 1>>>(); // ERROR, private type
        }
    };


```

 请参阅 <a href="https://godbolt.org/z/EhTn3GT3z" class="reference external"> 编译器上的示例Explorer</a>.





### 5.3.10.8. 平铺代码中的限制（Restrictions in Tile Code）

 使用 `__tile__` or `__tile_global__` 注释的函数具有以下附加限制：

 - 平铺代码中不支持以下语言构造：

 - 返回 `do`, `while` or `for` 循环内的语句。

- 虚函数调用。

 - Goto 语句。

 - Switch 语句。

 - 生成函数指针、函数引用、指向成员变量的指针或指向成员函数的指针的表达式。

 - 函数指针和指向成员函数调用的指针。

 - 指向成员变量的指针访问。

 - 128 位整数或浮点类型。

 - 包含位域的类型。

 - 大小超过 16 MB 的类型。

 - 具有虚拟基类或虚拟函数的类型。

 - 使用非放置运算符进行动态内存分配或释放`new` or `delete`.

- A `__tile__` or `__tile_global__` 函数必须在声明它的同一翻译单元中具有函数体。

 - 不支持使用 `__tile__` 注释虚函数。

- A `__tile_global__` or `__tile__` 函数不得使用 C 省略语法来具有可变数量的参数。 `...`.

- A `__tile_global__` or `__tile__` 函数不得直接或间接recursive.

- For `__tile_global__` 函数，函数参数不能是按值传递的类、结构或联合。

- Tile 代码可能无法执行设备端内核启动，并且 Tile 内核可能无法从设备端内核调用启动。

- 直接访问 `__x` 的 `__half`, `__nv_bfloat16` 成员变量及相关扩展浮动磁贴代码中不支持点类型。







<span id="cpp11"></span>

## 5.3.11. C++11 限制（C++11 Restrictions）



<span id="id20"></span>

### 5.3.11.1. `inline` 命名空间（`inline` Namespaces）

 当封闭命名空间中定义了具有相同名称和类型签名的另一个实体时，不允许在 `inline` 命名空间中定义以下实体之一：

- `__global__` or `__tile_global__` 函数。

- `__device__`, `__tile__`, `__constant__`, `__managed__`, `__shared__` Variables.

 - 具有表面或纹理类型的变量，例如 `cudaSurfaceObject_t` or `cudaTextureObject_t`.

 示例：

```cuda


    __device__ int my_var; // global scope

    inline namespace NS {

    __device__ int my_var; // namespace scope

    } // namespace NS


```





<span id="id21"></span>

### 5.3.11.2. `inline` 未命名命名空间（`inline` Unnamed Namespaces）

不能在 `inline` 未命名命名空间内的命名空间范围内声明以下实体：

- `__global__` or `__tile_global__` 函数。

- `__device__`, `__tile__`, `__constant__`, `__managed__`, `__shared__` 变量。

- 具有表面或纹理类型的变量，例如 `cudaSurfaceObject_t` or `cudaTextureObject_t`.





<span id="id22"></span>

### 5.3.11.3. `constexpr` 函数（`constexpr` Functions）

A `__global__` 函数不能声明为 `constexpr`。默认情况下，无法从具有不兼容执行空间的函数调用 `constexpr` 函数，这与标准函数相同。在本节列出的示例代码中，`UB` 代表“未定义行为”。

 - 在主机编译阶段（当 `constexpr` 宏未定义时）从主机函数调用没有显式或隐式 `__host__` 注释的 `__CUDA_ARCH__` 函数具有未定义的行为。示例：

  >
  >
> ```cuda
>
>
> constexpr __device__             int  device_func() { return 0; }
> constexpr __tile__               int  tile_func()   { return 0; }
>
> constexpr __device__ __host___   int host_device_func() { return 0; }
>
> int main() {
> constexpr int x1 = device_func(); // UB: calling a __device__-only constexpr function from host code
> constexpr int x2 = tile_func();   // UB: calling a __tile__-only constexpr function from host code
> constexpr int x3 = host_device_func(); // OK
>
> }
>
>
> ```
  >
  >

 - 在设备编译阶段（定义了 `constexpr` 宏时）从 `__device__` 函数调用没有显式或隐式 `__device__` or `__global__` 注释的 `__CUDA_ARCH__` 函数具有未定义的行为。示例：

  >
  >
> ```cuda
>
>
> constexpr  int host_func() { return 0; }
>
> __device__ void dmain()
> {
> int x = host_func();  // UB: calling a host-only constexpr function from device code
> }
>
>
> ```
  >
  >

 - 在设备编译阶段（定义了 `constexpr` 宏时）从 `__tile__` 函数调用没有显式或隐式 `__tile__` or `__tile_global__` 注释的 `__CUDA_ARCH__` 函数具有未定义的行为。示例：

  >
  >
> ```cuda
>
>
> constexpr  int host_func() { return 0; }
>
> __tile__ void dmain()
> {
> int x = host_func();  // UB: calling a host-only constexpr function from tile code
> }
>
>
> ```
  >
  >

 请注意，函数模板特化可能不是 `constexpr` 函数，即使相应的模板函数标有关键字 `constexpr`.

**放宽 constexpr-函数支持**

 实验性 `nvcc` 标志 `--expt-relaxed-constexpr` 可用于放宽此约束，如下所述。 `nvcc` 还将定义宏 `__CUDACC_RELAXED_CONSTEXPR__`。在本节列出的示例代码中`UB`代表“未定义行为”？

当`--expt-relaxed-constexpr`指定flag后，编译器将支持跨执行空间调用，如下：

1. 跨执行空间调用`constexpr`如果函数出现在需要常量计算的上下文中（例如在 constexpr 变量的初始值设定项中），则支持该函数。例子：

    >
    >
> ```cuda
>
>
> constexpr __host__ int host_func(int x) { return x + 1; };
>
> __global__ void doit() {
> constexpr int val = host_func(1); // OK: call is in a context that
> // requires constant evaluation.
> }
>
> __tile_global__ void tile_doit() {
> constexpr int val = host_func(1); // OK: call is in a context that
> // requires constant evaluation.
> }
>
>
> constexpr __device__ int device_func(int x) { return x + 1; }
>
> constexpr __tile__ int tile_func(int x) { return x + 2; }
>
> int main() {
> constexpr int val = device_func(1) + tile_func(1); // OK: call is in a context that
> // requires constant evaluation.
> }
>
>
> ```
    >
    >

2.否则：

    >
    >
> 1.从Tile代码到a的交叉执行空间调用`constexpr`没有显式或隐式的函数`__tile__`在语言规则需要常量折叠的上下文之外，不支持注释。例子：
    >
    >     >
    >     >
> ```cuda
>
>
> constexpr __host__ int host_func(int x) { return x + 1; }
>
> __tile__ int doit(int in) {
> in = host_func(in); // UB: call occurs outside of a context that requires
> // constant evaluation.
> constexpr int other = host_func(10); // OK with -expt-relaxed-constexpr:
> // call is  required to be evaluated at compile time
> }
>
>
> ```
    >     >
    >     >
    >
> 2. SIMT设备代码生成过程中，为仅主机的主体生成设备代码`constexpr`功能`host_func`， 除非`host_func`不使用或仅在恒定的评估上下文中调用。例子：
    >
    >     >
    >     >
> ```cuda
>
>
> // NOTE: "host_func" is emitted in generated device code because it is
> // called from device code in a non-constexpr context
> constexpr __host__ int host_func(int x) { return x + 1; }
>
> __device__ int doit(int in) {
> in = host_func(in);  // OK, even though argument is not a constant expression
> return in;
> }
>
>
> ```
    >     >
    >     >
    >
> 3. 适用于的所有代码限制`__device__`函数也适用于`constexpr`仅主机功能`H`这是从 SIMT 设备代码调用的。但是，编译器可能不会发出任何构建时诊断`H`对于这些限制。原因是诊断通常是在解析期间生成的，但是`H`在调用之前可能已经被解析`H`稍后在翻译单元中遇到来自设备代码。
    >
> 例如，以下代码模式在主体中不受支持`H`（与任何`__device__`函数），但可能不会生成编译器诊断信息：
    >
    >     >
    >     >
>> - ODR-使用主机变量或仅主机非`constexpr`功能。例子：
    >     >
    >     >   >
    >     >   >
> ```cuda
>
>
> int host_var1, host_var2;
>
> constexpr __host__ int* host_func(bool b) { return b ? &host_var1 : &host_var2; };
>
> __device__ int doit(bool flag) {
> int *ptr;
> ptr = host_func(flag); // UB: host_func() attempts to refer to host variables 'host_var1' and 'host_var2'.
> // code will compile, but will NOT execute correctly.
> return *ptr;
> }
>
>
> ```
    >     >   >
    >     >   >
    >     >
>> - 异常的使用 (`throw/catch`) 和 RTTI (`typeid,`` ``dynamic_cast`）。例子：
    >     >
    >     >   >
    >     >   >
> ```cuda
>
>
> struct Base { };
> struct Derived : public Base { };
>
> // NOTE: "host_func" is emitted in generated device code
> constexpr int host_func(bool b, Base *ptr) {
> if (b) {
> return 1;
> } else if (typeid(ptr) == typeid(Derived)) { // UB: use of typeid in code executing on the GPU
> return 2;
> } else {
> throw int{4}; // UB: use of throw in code executing on the GPU
> }
> }
>
> __device__ void doit(bool flag) {
> int val;
> Derived d;
> val = host_func(flag, &d); //UB: host_func() attempts use typeid and throw(), which are not allowed in code that executes on the GPU
> }
>
>
> ```
    >     >   >
    >     >   >
    >     >
    >     >
    >
    > >
    > >
>> 4. 在主机代码生成期间，`constexpr` 非主机函数 `F` 的主体保留在发送到主机编译器的代码中。如果 `F` 的主体尝试 ODR 使用命名空间作用域设备或 `__tile__` 变量或非主机非 `constexpr` 函数，则不支持从主机代码调用 `F`（代码可能在没有编译器诊断的情况下构建，但在运行时可能表现不正确）。示例：
    > >
> ```cuda
>
>
> __device__ int device_var1, device_var2;
>
> constexpr __device__ int* device_func(bool b) { return b ? &device_var1 : &device_var2; };
>
> __tile__ int tile_var1, tile_var2;
>
> constexpr __tile__ int* tile_func(bool b) { return b ? &tile_var1 : &tile_var2; };
>
> int doit1(bool flag) {
> int *ptr;
> ptr = device_func(flag); // UB: device_func() attempts to refer to device variables 'device_var1' and 'device_var2'
> // code will compile, but will NOT execute correctly.
> return *ptr;
> }
>
> int doit2(bool flag) {
> int *ptr;
> ptr = tile_func(flag); // UB: tile_func() attempts to refer to __tile__ variables 'tile_var1' and 'tile_var2'
> // code will compile, but will NOT execute correctly.
> return *ptr;
> }
>
>
> ```
    > >
    > >
    >
    >



Warning

 由于上述限制以及缺乏对错误使用的编译器诊断，建议避免从设备代码调用标准 C++ 头文件 `std::` 中的函数。这些功能的实现根据主机平台的不同而有所不同。相反，强烈建议在 <a href="#cpp-standard-library" class="reference internal"> 命名空间中调用 CUDA C++ 标准库 </a>libcu++`cuda::std::` 中的等效功能。







<span id="id23"></span>

### 5.3.11.4. `constexpr` 变量（`constexpr` Variables）

 默认情况下，`constexpr` 变量不能在执行空间不兼容的函数中使用，与标准的方式相同

A `constexpr` 变量在以下情况下可以直接在设备代码中使用：

 - C++ 标量类型，不包括指针和指向成员的指针类型：

  - `nullptr_t`.

  - `bool`.

 - 整型：`char`, `signed`` ``char`, `unsigned`, `long`` ``long` 等。

 - 浮点类型：`float`, `double`.

 - 枚举器： `enum` 和 `enum`` ``class`.

 - 类类型：`class`, `struct` 和带有 `union` 构造函数的 `constexpr`。

 - 上述类型的原始数组，例如 `int[]`，仅当它们在 `constexpr` `__device__` or `__host__`` ``__device__` 内使用时function.

`constexpr`` ``__managed__` 和 `constexpr`` ``__shared__` 变量是不允许的。

示例：

```cuda


    constexpr int ConstexprVar = 4; // scalar type

    struct MyStruct {
        static constexpr int ConstexprVar = 100;
    };

    constexpr MyStruct my_struct = MyStruct{}; // class type

    constexpr int array[] = {1, 2, 3};

    __device__ constexpr int get_value(int idx) {
        return array[idx];                      // CORRECT
    }

    __device__ void foo(int idx) {
        int        v1 = ConstexprVar;           // CORRECT
        int        v2 = MyStruct::ConstexprVar; // CORRECT
    //  const int &v3 = ConstexprVar1;          // ERROR, reference to host constexpr variable
    //  const int *v4 = &ConstexprVar1;         // ERROR, address of host constexpr variable
        int        v5 = get_value(2);           // CORRECT, 'get_value(2)' is a constant expression.
    //  int        v6 = get_value(idx);         // ERROR, 'get_value(idx)' is not a constant expression
    //  int        v7 = array[2];               // ERROR, 'array' is not scalar type.
        MyStruct   v8 = my_struct;              // CORRECT
    }


```

请参阅示例<a href="https://godbolt.org/z/MWa1o3c9z" class="reference external">编译器资源管理器</a>.





<span id="cpp11-variadic-template"></span>

### 5.3.11.5. `__global__`可变参数模板（`__global__` Variadic Template）

可变参数`__global__` or `__tile_global__`函数模板有以下限制：

- 只允许使用单个包参数。

- pack 参数必须列在模板参数列表的最后。

示例：

```cuda


    template <typename... Pack>
    __global__ void kernel1(); // CORRECT

    // template <typename... Pack, template T>
    // __global__ void kernel2(); // ERROR, parameter pack is not the last parameter

    template <typename... TArgs>
    struct MyStruct {};

    // template <typename... Pack1, typename... Pack2>
    // __global__ void kernel3(MyStruct<Pack1...>, MyStruct<Pack2...>); // ERROR, more than one parameter pack


```

请参阅示例<a href="https://godbolt.org/z/x48KnPbbY" class="reference external">编译器资源管理器</a>.





<span id="cpp11-defaulted-function"></span>

### 5.3.11.6. 默认功能`=`` ``default`（Defaulted Functions `=`` ``default`）

CUDA 编译器推断显式默认成员函数的执行空间，如中所述<a href="#compiler-generated-functions" class="reference internal">隐式声明和显式默认函数</a>.

显式默认函数上的执行空间说明符会被编译器忽略，除非该函数是外联定义的或者是一个`virtual`功能。

示例：

```cuda


    struct MyStruct1 {
        MyStruct1() = default;
    };

    void host_function() {
        MyStruct1 my_struct; // __host__ __device__ constructor
    }

    __device__ void device_function() {
        MyStruct1 my_struct; // __host__ __device__ constructor
    }

    struct MyStruct2 {
        __device__ MyStruct2() = default; // WARNING: __device__ annotation is ignored
    };

    struct MyStruct3 {
        __host__ MyStruct3();
    };
    MyStruct3::MyStruct3() = default; // out-of-line definition, not ignored

    __device__ void device_function2() {
    //  MyStruct3 my_struct; // ERROR, __host__ constructor
    }

    struct MyStruct4 {
        //  MyStruct4::~MyStruct4 has host execution space, not ignored because virtual
        virtual __host__ ~MyStruct4() = default;
    };

    __device__ void device_function3() {
        MyStruct4 my_struct4;
        // implicit destructor call for 'my_struct4':
        //    ERROR: call from a __device__ function 'device_function3' to a
        //    __host__ function 'MyStruct4::~MyStruct4'
    }


```

请参阅示例<a href="https://godbolt.org/z/q1M4j8YYf" class="reference external">编译器资源管理器</a>.





<span id="initializer-list"></span>

### 5.3.11.7. `[cuda::]std::initializer_list`（`[cuda::]std::initializer_list`）



默认情况下，CUDA 编译器隐式考虑以下成员函数`[cuda::]std::initializer_list`拥有`__host__`` ``__device__`` ``__tile__`执行空间说明符，因此可以直接从设备代码调用它们。





The `nvcc`旗帜`--no-host-device-initializer-list`禁用此行为；的成员函数`[cuda::]std::initializer_list`然后将被视为`__host__`函数，并且不能直接从设备代码调用。



A `__global__` or `__tile_global__`函数不能有类型的参数`[cuda::]std::initializer_list`.

例子：

```cuda


    #include <initializer_list>

    __device__ void foo(std::initializer_list<int> in) {}

    __device__ void bar() {
        foo({4,5,6}); // (a) initializer list containing only constant expressions.
        int i = 4;
        foo({i,5,6}); // (b) initializer list with at least one  non-constant element.
                      // This form may have better performance than (a).
    }


```

请参阅示例<a href="https://godbolt.org/z/xeah7r44T" class="reference external">编译器资源管理器</a>.





<span id="rvalue-references"></span>

### 5.3.11.8. `[cuda::]std::move`, `[cuda::]std::forward`（`[cuda::]std::move`, `[cuda::]std::forward`）

默认情况下，CUDA 编译器隐式考虑`std::move`和`std::forward`具有的函数模板`__host__`` ``__device__`` ``__tile__`执行空间说明符，因此可以直接从设备代码调用它们。这`nvcc`旗帜`--no-host-device-move-forward`禁用此行为；`std::move`和`std::forward`然后将被视为`__host__`函数，并且不能直接从设备代码调用。



暗示

`cuda::std::move`和`cuda::std::forward`相反总是有`__host__`` ``__device__`执行空间。









<span id="cpp14"></span>

## 5.3.12. C++14 限制（C++14 Restrictions）



<span id="return-type-deduction"></span>

### 5.3.12.1. 具有推导返回类型的函数（Functions with Deduced Return Type）

A `__global__` or `__tile_global__`函数不能有推导的返回类型`auto`.

自省 a 的返回类型`__device__`主机代码中不允许使用具有推导返回类型的函数。



笔记

CUDA 前端编译器更改函数声明以具有`void`返回类型，在调用主机编译器之前。这可能会破坏对推导的返回类型的自省`__device__`主机代码中的函数。因此，CUDA 编译器将在设备函数体外部引用此类推导的返回类型时发出编译时错误。



示例：

```cuda


     __device__ auto device_function(int x) { // deduced return type
         return x;                            // decltype(auto) has the same behavior
     }

     __global__ void kernel() {
         int x = sizeof(device_function(2));         // CORRECT, device code scope
     }

     // const int size = sizeof(device_function(2)); // ERROR, return type deduction on host

     void host_function() {
     //  using T = decltype(device_function(2));     // ERROR, return type deduction on host
     }

    void host_fn1() {
      // ERROR, referenced outside device function bodies
      int (*p1)(int) = fn1;

      struct S_local_t {
        // ERROR, referenced outside device function bodies
        decltype(fn2(10)) m1;

        S_local_t() : m1(10) { }
      };
    }

    // ERROR, referenced outside device function bodies
    template <typename T = decltype(fn2)>
    void host_fn2() { }

    template<typename T> struct MyStruct { };

    // ERROR, referenced outside device function bodies
    struct S1_derived_t : MyStruct<decltype(fn1)> { };


```





<span id="id24"></span>

### 5.3.12.2. 变量模板（Variable Templates）

A `__device__`, `__tile__` or `__constant__`变量模板不能`const`- 使用 Microsoft 编译器时限定。

示例：

```cuda


    // ERROR on Windows (non-portable), const-qualified
    template <typename T>
    __device__ const T var = 0;

     // CORRECT, ptr1 is not const-qualified
    template <typename T>
    __device__ const T* ptr1 = nullptr;

    // ERROR on Windows (non-portable), ptr2 is const-qualified
    template <typename T>
    __device__ const T* const ptr2 = nullptr;


```

请参阅示例<a href="https://godbolt.org/z/8hM5Yh7db" class="reference external">编译器资源管理器</a>.







<span id="cpp17"></span>

## 5.3.13. C++17 限制（C++17 Restrictions）



<span id="id25"></span>

### 5.3.13.1. `inline`变量（`inline` Variables）



在单个翻译单元中，使用`inline`变量不提供超出常规变量的附加功能，并且不提供任何实际优势。





`nvcc`允许`inline`变量与`__device__`, `__tile__`, `__constant__`, or `__managed__`内存空间仅在<a href="../02-programming-gpus/nvcc.html#nvcc-separate-compilation" class="reference internal">单独编译</a>模式或具有内部链接的变量。





笔记

使用时`gcc/g++`主机编译器，一个`inline`声明的变量`__managed__`内存空间说明符可能对调试器不可见。



示例：

```cuda


    inline        __device__ int device_var1;  // CORRECT, when compiled in Separate Compilation mode (-rdc=true or -dc)
                                               // ERROR, when compiled in Whole Program Compilation mode

    static inline __device__ int device_var2;  // CORRECT, internal linkage

    namespace {

    inline __device__ int device_var3;         // CORRECT, internal linkage

    inline __shared__ int shared_var;          // CORRECT, internal linkage

    static inline __device__ int device_var4;  // CORRECT, internal linkage

    inline __device__ int device_var5;         // CORRECT, internal linkage

    } // namespace


```

请参阅 <a href="https://godbolt.org/z/oraqeGTzY" class="reference external">Compiler Explorer</a>.





<span id="id26"></span>

### 5.3.13.2. Structured Binding（Structured Binding）

A 结构化绑定无法使用内存空间说明符声明，例如`__device__`, `__tile__`, `__shared__`, `__constant__`, or `__managed__`.

示例：

```cuda


    struct S {
        int x, y;
    };
    // __device__ auto [a, b] = S{4, 5}; // ERROR


```







<span id="cpp20"></span>

## 5.3.14. C++20 限制（C++20 Restrictions）



<span id="cpp20-spaceship"></span>

### 5.3.14.1. 三向比较运算符（Three-way Comparison Operator）

`<=>` 和 `__device__` 函数支持三向比较运算符 (`__global__`)，但某些使用隐式依赖于 C++ 标准库中的功能，其中由主机实现提供。使用这些运算符可能需要指定标志 `--expt-relaxed-constexpr` 来消除警告，并且该功能要求主机实现满足设备代码的要求。

 示例：

```cuda


    #include <compare> // std::strong_ordering implementation

    struct S {
        int x, y;

        auto operator<=>(const S&) const = default; // (a)

        __host__ __device__ bool operator<=>(int rhs) const { return false; } // (b)
    };

    __host__ __device__ bool host_device_function(S a, S b) {
        if (a <=> 1)  // CORRECT, calls a user-defined host-device overload (b)
            return true;
        return a < b; // CORRECT, call to an implicitly-declared function (a)
                      // Note: it requires a device-compatible std::strong_ordering
                      //       implementation provided in the header <compare>
                      //       and the flag --expt-relaxed-constexpr
    }


```

 请参阅 <a href="https://godbolt.org/z/qzs5arfx4" class="reference external"> 编译器资源管理器中的示例 </a>.





<span id="cpp20-consteval"></span>

### 5.3.14.2. `consteval` 函数（`consteval` Functions）

`consteval` 函数可以从主机和设备代码中调用，与主机和设备代码无关。

示例：

```cuda


    consteval int host_consteval() {
        return 10;
    }

    __device__ consteval int device_consteval() {
        return 10;
    }

    __device__ int device_function() {
        return host_consteval();   // CORRECT, even if called from device code
    }

    __host__ __device__ int host_device_function() {
        return device_function();  // CORRECT, even if called from host-device code
    }


```







<span id="cpp23-restrictions"></span>

## 5.3.15. C++23 限制（C++23 Restrictions）

 除了上面的 <a href="#cpp23-language-features" class="reference internal">C++23 语言功能表 </a> 中指示的不支持或不适用的功能外，没有已知的特定于 C++23 的限制。该表中标记为“或 N/A”的条目（包括缺陷报告解决方案）反映了缺失或不适用的功能，而不是其他行为限制。



### 5.3.15.1. Equality 运算符 (P2468R2)（Equality Operator (P2468R2)）

 尽管 NVCC 未完全实现 **DR：您正在寻找的 Equality 运算符** (P2468R2)，但它会近似行为，并且这种近似不会导致已知故障在用户代码中。
