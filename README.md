# CUDA 编程指南中文图解

这是 NVIDIA **CUDA Programming Guide Release 13.3** 的简体中文翻译与图解 VuePress 站点。内容按官方在线文档目录分节整理，并保留原 PDF 图版及对应说明。

## 在线阅读

- [GitHub Pages 在线站点](https://okazaki-yumemi.github.io/CUDA_PROGRAMMING_GUIDE_simplified-Chinese/)
- [NVIDIA 官方在线文档](https://docs.nvidia.com/cuda/cuda-programming-guide/contents.html)

站点支持全文搜索，并提供白天/夜间颜色模式切换。

## 本地预览

需要 Node.js 22 或更高版本，以及 pnpm 9.15.0：

```bash
pnpm install
pnpm run docs:dev
```

构建静态文件：

```bash
pnpm run docs:build
```

生成目录为 `docs/.vuepress/dist`。

## 自动部署

仓库中的 `.github/workflows/deploy-pages.yml` 会在 `phase-2/full-translation` 分支有新提交时自动构建并发布到 GitHub Pages，也可以在 GitHub Actions 页面手动运行。

首次启用时，请在仓库的 **Settings → Pages → Build and deployment → Source** 中选择 **GitHub Actions**。部署完成后，站点地址为：

```text
https://okazaki-yumemi.github.io/CUDA_PROGRAMMING_GUIDE_simplified-Chinese/
```

由于这是项目站点，VuePress 配置中的 `base` 必须保持为 `/CUDA_PROGRAMMING_GUIDE_simplified-Chinese/`。

## 内容范围

- 第 1 至第 6 节的中文翻译与导航
- 原 PDF 图片图版及图片说明
- CUDA C++、CUDA Python、内存、执行模型、CUDA Graphs 等主题
- NVIDIA 官方原文与本地 PDF 入口

本项目是学习用途的非官方中文翻译，CUDA、NVIDIA 及相关产品名称的权利归其各自所有者所有。
