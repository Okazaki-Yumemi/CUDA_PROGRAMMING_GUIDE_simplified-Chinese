import { viteBundler } from '@vuepress/bundler-vite'
import { searchPlugin } from '@vuepress/plugin-search'
import { defaultTheme } from '@vuepress/theme-default'
import { defineUserConfig } from 'vuepress'

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
    navbar: [
      { text: '首页', link: '/' },
      { text: '阅读地图', link: '/guide-index.html' },
      { text: '六大部分', link: '/chapters/01-introduction.html' },
      { text: 'PDF 图版', link: '/figures.html' },
      {
        text: '官方资料',
        children: [
          {
            text: 'NVIDIA 官方在线目录',
            link: 'https://docs.nvidia.com/cuda/cuda-programming-guide/contents.html',
          },
          {
            text: '本地原始 PDF',
            link: '/source.html',
          },
        ],
      },
    ],
    sidebar: {
      '/chapters/': [
        {
          text: 'CUDA 编程指南',
          children: [
            '/chapters/01-introduction.html',
            '/chapters/02-programming-gpus.html',
            '/chapters/03-advanced-cuda.html',
            '/chapters/04-cuda-features.html',
            '/chapters/05-technical-appendices.html',
            '/chapters/06-notices.html',
          ],
        },
      ],
      '/': [
        {
          text: '开始阅读',
          children: ['/guide-index.html', '/figures.html', '/source.html'],
        },
      ],
    },
    editLink: false,
    contributors: false,
    lastUpdated: false,
    repo: false,
    locales: {
      '/': {
        selectLanguageName: '简体中文',
        navbar: [
          { text: '首页', link: '/' },
          { text: '阅读地图', link: '/guide-index.html' },
          { text: '六大部分', link: '/chapters/01-introduction.html' },
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
        ],
        sidebar: {
          '/chapters/': [
            {
              text: 'CUDA 编程指南',
              children: [
                '/chapters/01-introduction.html',
                '/chapters/02-programming-gpus.html',
                '/chapters/03-advanced-cuda.html',
                '/chapters/04-cuda-features.html',
                '/chapters/05-technical-appendices.html',
                '/chapters/06-notices.html',
              ],
            },
          ],
          '/': [
            {
              text: '开始阅读',
              children: ['/guide-index.html', '/figures.html', '/source.html'],
            },
          ],
        },
        toggleColorMode: '切换颜色模式',
        selectLanguageName: '简体中文',
      },
    },
  }),
  plugins: [
    searchPlugin({
      maxSuggestions: 12,
    }),
  ],
})
