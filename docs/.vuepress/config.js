import { viteBundler } from '@vuepress/bundler-vite'
import { searchPlugin } from '@vuepress/plugin-search'
import { defaultTheme } from '@vuepress/theme-default'
import { defineUserConfig } from 'vuepress'

const sectionSidebar = {
  '/01-introduction/': [
    {
      text: '1. Introduction to CUDA',
      children: [
        { text: '1.1 Introduction', link: '/01-introduction/introduction.html' },
        { text: '1.2 Programming Model', link: '/01-introduction/programming-model.html' },
        { text: '1.3 The CUDA platform', link: '/01-introduction/cuda-platform.html' },
      ],
    },
  ],
  '/02-programming-gpus/': [
    {
      text: '2. Programming GPUs in CUDA',
      children: [
        { text: '2.1 Intro to CUDA C++', link: '/02-programming-gpus/intro-to-cuda-cpp.html' },
        { text: '2.2 Intro to CUDA Python', link: '/02-programming-gpus/intro-to-cuda-python.html' },
        { text: '2.3 Writing SIMT Kernels', link: '/02-programming-gpus/writing-simt-kernels.html' },
        { text: '2.4 Writing Tile Kernels', link: '/02-programming-gpus/writing-tile-kernels.html' },
        { text: '2.5 Asynchronous Execution', link: '/02-programming-gpus/asynchronous-execution.html' },
        { text: '2.6 Unified and System Memory', link: '/02-programming-gpus/unified-and-system-memory.html' },
        { text: '2.7 NVCC', link: '/02-programming-gpus/nvcc.html' },
      ],
    },
  ],
  '/03-advanced-cuda/': [
    {
      text: '3. Advanced CUDA',
      children: [
        { text: '3.1 Advanced CUDA APIs and Features', link: '/03-advanced-cuda/advanced-apis-and-features.html' },
        { text: '3.2 Advanced Kernel Programming', link: '/03-advanced-cuda/advanced-kernel-programming.html' },
        { text: '3.3 The CUDA Driver API', link: '/03-advanced-cuda/driver-api.html' },
        { text: '3.4 Multiple GPUs', link: '/03-advanced-cuda/multiple-gpus.html' },
        { text: '3.5 A Tour of CUDA Features', link: '/03-advanced-cuda/tour-of-cuda-features.html' },
      ],
    },
  ],
  '/04-cuda-features/': [
    {
      text: '4. CUDA Features',
      children: [
        { text: '4.1 Unified Memory', link: '/04-cuda-features/unified-memory.html' },
        { text: '4.2 CUDA Graphs', link: '/04-cuda-features/cuda-graphs.html' },
        { text: '4.3 Stream-Ordered Memory Allocator', link: '/04-cuda-features/stream-ordered-memory-allocator.html' },
        { text: '4.4 Cooperative Groups', link: '/04-cuda-features/cooperative-groups.html' },
        { text: '4.5 Programmatic Dependent Launch', link: '/04-cuda-features/programmatic-dependent-launch.html' },
        { text: '4.6 Green Contexts', link: '/04-cuda-features/green-contexts.html' },
        { text: '4.7 Lazy Loading', link: '/04-cuda-features/lazy-loading.html' },
        { text: '4.8 Error Log Management', link: '/04-cuda-features/error-log-management.html' },
        { text: '4.9 Asynchronous Barriers', link: '/04-cuda-features/asynchronous-barriers.html' },
        { text: '4.10 Pipelines', link: '/04-cuda-features/pipelines.html' },
        { text: '4.11 Asynchronous Data Copies', link: '/04-cuda-features/asynchronous-data-copies.html' },
        { text: '4.12 Work Stealing', link: '/04-cuda-features/work-stealing.html' },
        { text: '4.13 L2 Cache Control', link: '/04-cuda-features/l2-cache-control.html' },
        { text: '4.14 Memory Synchronization Domains', link: '/04-cuda-features/memory-synchronization-domains.html' },
        { text: '4.15 Interprocess Communication', link: '/04-cuda-features/interprocess-communication.html' },
        { text: '4.16 Virtual Memory Management', link: '/04-cuda-features/virtual-memory-management.html' },
        { text: '4.17 Extended GPU Memory', link: '/04-cuda-features/extended-gpu-memory.html' },
        { text: '4.18 CUDA Dynamic Parallelism', link: '/04-cuda-features/cuda-dynamic-parallelism.html' },
        { text: '4.19 CUDA Interoperability', link: '/04-cuda-features/cuda-interoperability.html' },
        { text: '4.20 Driver Entry Point Access', link: '/04-cuda-features/driver-entry-point-access.html' },
      ],
    },
  ],
  '/05-technical-appendices/': [
    {
      text: '5. Technical Appendices',
      children: [
        { text: '5.1 Compute Capabilities', link: '/05-technical-appendices/compute-capabilities.html' },
        { text: '5.2 CUDA Environment Variables', link: '/05-technical-appendices/environment-variables.html' },
        { text: '5.3 C++ Language Support', link: '/05-technical-appendices/cpp-language-support.html' },
        { text: '5.4 C/C++ Language Extensions', link: '/05-technical-appendices/cpp-language-extensions.html' },
        { text: '5.5 Floating-Point Computation', link: '/05-technical-appendices/floating-point-computation.html' },
        { text: '5.6 Device-Callable APIs and Intrinsics', link: '/05-technical-appendices/device-callable-apis.html' },
        { text: '5.7 CUDA C++ Memory Model', link: '/05-technical-appendices/cuda-memory-model.html' },
        { text: '5.8 CUDA C++ Execution model', link: '/05-technical-appendices/cuda-execution-model.html' },
      ],
    },
  ],
  '/06-notices/': [
    {
      text: '6. Notices',
      children: [
        { text: '6.1 Notice', link: '/06-notices/notice.html' },
        { text: '6.2 OpenCL', link: '/06-notices/opencl.html' },
        { text: '6.3 Trademarks', link: '/06-notices/trademarks.html' },
      ],
    },
  ],
  '/': [
    {
      text: '开始阅读',
      children: ['/guide-index.html', '/figures.html', '/source.html'],
    },
  ],
}

const navbar = [
  { text: '首页', link: '/' },
  { text: '阅读地图', link: '/guide-index.html' },
  { text: '1.1 Introduction', link: '/01-introduction/introduction.html' },
  { text: 'PDF 图版', link: '/figures.html' },
  {
    text: '官方资料',
    children: [
      {
        text: 'NVIDIA 官方在线目录',
        link: 'https://docs.nvidia.com/cuda/cuda-programming-guide/contents.html',
      },
      { text: '本地原始 PDF', link: '/source.html' },
    ],
  },
]

export default defineUserConfig({
  lang: 'zh-CN',
  title: 'CUDA 编程指南中文图解',
  description: '基于 NVIDIA CUDA Programming Guide Release 13.3 的中文翻译、概念导读与 PDF 图版说明',
  head: [
    ['meta', { name: 'theme-color', content: '#76b900' }],
    ['meta', { name: 'viewport', content: 'width=device-width, initial-scale=1' }],
  ],
  bundler: viteBundler(),
  theme: defaultTheme({
    logo: '/figures/page-001.png',
    logoAlt: 'CUDA Programming Guide',
    colorMode: 'auto',
    colorModeSwitch: true,
    navbar,
    sidebar: sectionSidebar,
    editLink: false,
    contributors: false,
    lastUpdated: false,
    repo: false,
    locales: {
      '/': {
        selectLanguageName: '简体中文',
        toggleColorMode: '切换颜色模式',
      },
    },
  }),
  plugins: [
    searchPlugin({
      maxSuggestions: 12,
    }),
  ],
})
