---
title: 4.8 错误日志管理
description: CUDA Error Log Management 机制、输出格式与 Driver API
---

# 4.8 错误日志管理（Error Log Management）

*Error Log Management* 机制允许 CUDA API 错误以通俗英文报告给开发者，并描述问题的原因。

## 4.8.1 背景（Background）

传统上，CUDA API 调用失败的唯一迹象是返回非零错误码。截至 CUDA Toolkit 12.9，CUDA Runtime 针对错误条件定义了 100 多种不同的返回码，但其中许多返回码过于通用，无法帮助开发者调试问题根因。

## 4.8.2 激活（Activation）

设置 *CUDA_LOG_FILE* 环境变量。可接受的值为 *stdout*、*stderr*，或者系统中用于写入文件的有效路径。即使程序执行前没有设置 *CUDA_LOG_FILE*，也可以通过 API 导出日志缓冲区。注意：无错误的执行可能不会打印任何日志。

## 4.8.3 输出（Output）

日志按以下格式输出：

```text
[Time][TID][Source][Severity][API Entry Point] Message
```

下面是一条开发者尝试将 Error Log Management 日志导出到未分配缓冲区时生成的实际错误消息：

```text
[22:21:32.099][25642][CUDA][E][cuLogsDumpToMemory] buffer cannot be NULL
```

此前，开发者只能从返回码中得到 *CUDA_ERROR_INVALID_VALUE*，并且在调用 *cuGetErrorString* 时可能得到“invalid argument”。

## 4.8.4 API 描述（API Description）

CUDA Driver 提供两类 API，用于与 Error Log Management 功能交互。

该功能允许开发者注册回调函数，在生成错误日志时调用。回调函数签名如下：

```cpp
void callbackFunc(void *data, CUlogLevel logLevel, char *message, size_t length)
```

使用以下 API 注册回调：

```cpp
CUresult cuLogsRegisterCallback(CUlogsCallback callbackFunc, void *userData, CUlogsCallbackHandle *callback_out)
```

其中，*userData* 会原样传递给回调函数。调用者应保存 *callback_out*，以便之后传递给 *cuLogsUnregisterCallback*。

```cpp
CUresult cuLogsUnregisterCallback(CUlogsCallbackHandle callback)
```

另一组 API 用于管理日志输出。一个重要概念是日志迭代器，它指向缓冲区当前末尾：

```cpp
CUresult cuLogsCurrent(CUlogIterator *iterator_out, unsigned int flags)
```

当调用方不希望每次都导出整个日志缓冲区时，可以保存迭代器位置。目前 `flags` 参数必须为 0；其他选项为未来 CUDA 版本预留。

任何时候，都可以使用以下函数将错误日志缓冲区导出到文件或内存：

```cpp
CUresult cuLogsDumpToFile(CUlogIterator *iterator, const char *pathToFile, unsigned int flags)
CUresult cuLogsDumpToMemory(CUlogIterator *iterator, char *buffer, size_t *size, unsigned int flags)
```

如果 *iterator* 为 NULL，则会导出整个缓冲区，最多 100 条。如果 *iterator* 不为 NULL，则从该条目开始导出日志，并将 *iterator* 更新到当前日志末尾，效果等同于调用 *cuLogsCurrent*。如果缓冲区中日志条目超过 100 条，导出开头会附加一条说明。

`flags` 参数必须为 0；其他选项为未来 CUDA 版本预留。

`cuLogsDumpToMemory` 函数还有以下注意事项：

1. 缓冲区本身会以 null 结尾，但每个日志条目之间只用换行字符 `\n` 分隔。
2. 缓冲区最大大小为 25600 字节。
3. 如果 `size` 提供的值不足以存储所有希望导出的日志，会将一条说明作为第一条记录加入，并且不会导出放不下的最旧记录。
4. 返回后，`size` 包含写入所提供缓冲区的实际字节数。

## 4.8.5 限制与已知问题（Limitations and Known Issues）

1. 日志缓冲区最多保存 100 条记录。达到上限后，最旧的记录会被替换，日志导出中会包含一行说明发生了回卷。
2. 当前尚未覆盖所有 CUDA API。这是一个持续进行中的项目，目标是为所有 API 提供更好的使用错误报告。
3. 只有在生成日志时才会检查 Error Log Management 日志位置（如果提供）的有效性；在此之前不会进行检查。
4. Error Log Management API 当前只能通过 CUDA Driver 使用。未来版本会向 CUDA Runtime 添加等价 API。
5. 日志消息尚未本地化到任何语言，所有提供的日志均为美国英语。
