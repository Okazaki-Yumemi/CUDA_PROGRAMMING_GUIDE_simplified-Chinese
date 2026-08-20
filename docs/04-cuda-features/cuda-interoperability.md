---
title: 4.19 CUDA 与 API 的互操作（CUDA Interoperability with APIs）
description: 在 CUDA、OpenGL、Direct3D、Vulkan 和 NVSCI 之间共享 GPU 数据、内存与同步对象。
---

# 4.19 CUDA 与 API 的互操作（CUDA Interoperability with APIs）

在 CUDA 中直接访问其他 API 的 GPU 数据，可以使用 CUDA Kernel 读写这些数据，从而在其他 API 消费数据的同时使用 CUDA 的功能。本节包含两种主要方式：第一种是较直接的[图形互操作](#4191-图形互操作graphics-interoperability)，它支持 OpenGL 与 Direct3D[9–11]，可将 OpenGL 和 Direct3D 资源映射到 CUDA 地址空间；第二种是更灵活的[外部资源互操作](#4192-外部资源互操作external-resource-interoperability)，通过操作系统级句柄导入和导出内存对象与同步对象。后者支持 Direct3D[11–12]、Vulkan 以及 NVIDIA Software Communication Interface Interoperability（NVSCI）。

## 4.19.1 图形互操作（Graphics Interoperability）

在 CUDA 访问 Direct3D 或 OpenGL 资源之前，例如访问 VBO（vertex buffer object，顶点缓冲对象），必须先注册并映射资源。使用相应 CUDA 函数注册资源（见下方示例）会返回一个 `struct cudaGraphicsResource` 类型的 CUDA 图形资源，其中保存 CUDA device pointer 或 CUDA array。要在 Kernel 中访问设备数据，必须先映射该资源。资源处于注册状态时，可以按需要反复映射和取消映射。对于缓冲区，Kernel 使用 `cudaGraphicsResourceGetMappedPointer()` 返回的设备内存地址访问已映射资源；对于 CUDA array，则使用 `cudaGraphicsSubResourceGetMappedArray()`。CUDA 不再需要资源后，可以取消注册。

主要步骤如下：

1. 向 CUDA 注册图形缓冲区。
2. 映射资源。
3. 获取已映射资源的 device pointer 或 array。
4. 在 CUDA Kernel 中使用 device pointer 或 array。
5. 取消映射资源。
6. 取消注册资源。

注册资源的开销较高，因此理想情况下每个资源只注册一次；但每个准备使用该资源的 CUDA context 都必须分别注册资源。可以调用 `cudaGraphicsResourceSetMapFlags()` 指定使用提示（只写、只读），供 CUDA 驱动优化资源管理。还要注意：在资源已映射时，如果通过 OpenGL、Direct3D 或其他 CUDA context 访问该资源，会产生未定义结果。

### 4.19.1.1 OpenGL 互操作（OpenGL Interoperability）

可以映射到 CUDA 地址空间的 OpenGL 资源包括 OpenGL buffer、texture 和 renderbuffer 对象。使用 `cudaGraphicsGLRegisterBuffer()` 注册 buffer 对象后，它在 CUDA 中表现为普通 device pointer。使用 `cudaGraphicsGLRegisterImage()` 注册 texture 或 renderbuffer 对象后，它在 CUDA 中表现为 CUDA array。

如果 texture 或 renderbuffer 对象注册时使用了 `cudaGraphicsRegisterFlagsSurfaceLoadStore` 标志，则可以写入该对象。`cudaGraphicsGLRegisterImage()` 支持具有 1、2 或 4 个分量、内部类型为 float 的所有纹理格式（例如 `GL_RGBA_FLOAT32`）、归一化整数格式（例如 `GL_RGBA8`、`GL_INTENSITY16`）以及非归一化整数格式（例如 `GL_RGBA8UI`）。

**示例：simpleGL 互操作（Example: simpleGL interoperability）**

下面的代码示例使用一个 Kernel 动态修改存储在 vertex buffer object（VBO）中的二维 `width` × `height` 顶点网格，主要步骤是：

1. 向 CUDA 注册 VBO。
2. 循环：将 VBO 映射为 CUDA 可写资源。
3. 循环：运行 CUDA Kernel 修改顶点位置。
4. 循环：取消映射 VBO。
5. 循环：使用 OpenGL 渲染结果。
6. 取消注册并删除 VBO。

本节的完整 simpleGL 示例位于 [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleGL)。

```cuda
__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // calculate uv coordinates
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;

    // write output vertex
    pos[y * width + x] = make_float4(u, w, v, 1.0f);
}

int main(int argc, char **argv)
{
    char *ref_file = NULL;

    pArgc = &argc;
    pArgv = argv;

#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    printf("%s starting...\n", sSDKsample);

    if (argc > 1) {
        if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
            // In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
            getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
        }
    }

    printf("\n");

    // First initialize OpenGL context
    if (false == initGL(&argc, argv)) {
        return false;
    }

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutCloseFunc(cleanup);

    // Create an empty vertex buffer object (VBO)
    // 1. Register the VBO with CUDA
    createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

    // start rendering mainloop
    //  5. Render the results using OpenGL
    glutMainLoop();

    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags)
{
    assert(vbo);
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
    SDK_CHECK_ERROR_GL();
}

void display()
{
    float4 *dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cuda_vbo_resource));
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    simple_vbo_kernel<<<grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);
    glutSwapBuffers();
    g_fAnim += 0.01f;
}

void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));
    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);
    *vbo = 0;
}

void cleanup()
{
    if (vbo) {
        deleteVBO(&vbo, cuda_vbo_resource);
    }
}
```

**限制与注意事项（Limitations and considerations）**

- 共享资源所属的 OpenGL context，必须对执行 OpenGL 互操作 API 调用的 host 线程保持 current。
- 当 OpenGL texture 被设为 bindless（例如通过 `glGetTextureHandle` 或 `glGetImageHandle` API 请求 image 或 texture handle）后，不能再向 CUDA 注册。应用必须在请求 image 或 texture handle 之前完成纹理互操作注册。

### 4.19.1.2 Direct3D 互操作（Direct3D Interoperability）

Direct3D 互操作支持 Direct3D9、Direct3D10 和 Direct3D11，不支持 Direct3D12。本节聚焦 Direct3D11；关于 Direct3D9 和 Direct3D10，请参阅 CUDA Programming Guide 12.9。可以映射到 CUDA 地址空间的 Direct3D 资源包括 Direct3D buffer、texture 和 surface，这些资源使用 `cudaGraphicsD3D11RegisterResource()` 注册。

一个 CUDA context 只能与创建时 `DriverType` 设置为 `D3D_DRIVER_TYPE_HARDWARE` 的 Direct3D11 device 进行互操作。

**示例：Direct3D11 二维纹理互操作（Example: 2D Texture Direct3D11 interoperability）**

下面的代码片段来自 [simpleD3D11Texture](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleD3D11Texture) 示例。完整示例包含大量 DX11 样板代码，这里聚焦 CUDA 部分。

CUDA Kernel `cuda_kernel_texture_2d` 在闪烁的蓝色背景上绘制移动的红/绿色网纹；它依赖纹理之前的值。底层数据是二维 CUDA array，行偏移由 pitch 定义。

```cuda
/* Paint a 2D texture with a moving red/green hatch pattern on a
 * strobing blue background.  Note that this kernel reads to and
 * writes from the texture, hence why this texture was not mapped
 * as WriteDiscard. */
__global__ void cuda_kernel_texture_2d(unsigned char *surface, int width,
                                       int height, size_t pitch, float t) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  float *pixel;
  if (x >= width || y >= height) return;
  pixel = (float *)(surface + y * pitch) + 4 * x;
  float value_x = 0.5f + 0.5f * cos(t + 10.0f * ((2.0f * x) / width - 1.0f));
  float value_y = 0.5f + 0.5f * cos(t + 10.0f * ((2.0f * y) / height - 1.0f));
  pixel[0] = 0.5 * pixel[0] + 0.5 * pow(value_x, 3.0f);
  pixel[1] = 0.5 * pixel[1] + 0.5 * pow(value_y, 3.0f);
  pixel[2] = 0.5f + 0.5f * cos(t);
  pixel[3] = 1;
}

extern "C" void cuda_texture_2d(void *surface, int width, int height,
                                 size_t pitch, float t) {
  cudaError_t error = cudaSuccess;
  dim3 Db = dim3(16, 16);
  dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);
  cuda_kernel_texture_2d<<<Dg, Db>>>((unsigned char *)surface, width, height,
                                     pitch, t);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
  }
}
```

为了让指针和所属数据缓冲区保持在一起，使用下面的数据结构：

```cuda
// Data structure for 2D texture shared between DX11 and CUDA
struct {
  ID3D11Texture2D *pTexture;
  ID3D11ShaderResourceView *pSRView;
  cudaGraphicsResource *cudaResource;
  void *cudaLinearMemory;
  size_t pitch;
  int width;
  int height;
  int offsetInShader;
} g_texture_2d;
```

Direct3D device 和纹理初始化后，资源只向 CUDA 注册一次。为了匹配 Direct3D 像素格式，CUDA array 使用相同的宽高进行分配，并使用与 Direct3D 纹理行 pitch 匹配的 pitch。

```cuda
cudaGraphicsD3D11RegisterResource(&g_texture_2d.cudaResource,
                                  g_texture_2d.pTexture,
                                  cudaGraphicsRegisterFlagsNone);
getLastCudaError("cudaGraphicsD3D11RegisterResource (g_texture_2d) failed");
cudaMallocPitch(&g_texture_2d.cudaLinearMemory, &g_texture_2d.pitch,
                g_texture_2d.width * sizeof(float) * 4,
                g_texture_2d.height);
getLastCudaError("cudaMallocPitch (g_texture_2d) failed");
cudaMemset(g_texture_2d.cudaLinearMemory, 1,
           g_texture_2d.pitch * g_texture_2d.height);
```

在渲染循环中，先映射资源，启动 CUDA Kernel 更新纹理数据，再取消映射资源。之后由 Direct3D device 将更新后的纹理绘制到屏幕上。

```cuda
cudaStream_t stream = 0;
const int nbResources = 3;
cudaGraphicsResource *ppResources[nbResources] = {
    g_texture_2d.cudaResource, g_texture_3d.cudaResource,
    g_texture_cube.cudaResource,
};
cudaGraphicsMapResources(nbResources, ppResources, stream);
getLastCudaError("cudaGraphicsMapResources(3) failed");
RunKernels();
cudaGraphicsUnmapResources(nbResources, ppResources, stream);
getLastCudaError("cudaGraphicsUnmapResources(3) failed");
```

CUDA 不再需要资源后，取消资源注册并释放设备 array：

```cuda
cudaGraphicsUnregisterResource(g_texture_2d.cudaResource);
getLastCudaError("cudaGraphicsUnregisterResource (g_texture_2d) failed");
cudaFree(g_texture_2d.cudaLinearMemory);
getLastCudaError("cudaFree (g_texture_2d) failed");
```

### 4.19.1.3 可扩展链路接口（SLI）配置下的互操作（Interoperability in a Scalable Link Interface (SLI) configuration）

在多 GPU 系统中，所有支持 CUDA 的 GPU 都会通过 CUDA driver 和 runtime 作为独立 device 访问。系统处于 SLI 模式时情况不同。SLI 是一种硬件配置的多 GPU 方案，通过在多个 GPU 之间划分工作负载来提高渲染性能。隐式 SLI 模式（由驱动自动作出假设）已不再支持，但显式 SLI 仍受支持。显式 SLI 表示应用通过 API（例如 Vulkan、DirectX、GL）了解并管理 SLI 组中所有 device 的 SLI 状态。

系统处于 SLI 模式时需要特别注意：

- 在一个 GPU 上的 CUDA device 中进行分配，会消耗 Direct3D 或 OpenGL device 的 SLI 配置中其他 GPU 的内存。因此，分配可能会比预期更早失败。
- 应用应为 SLI 配置中的每个 GPU 创建一个 CUDA context。虽然这不是严格要求，但可以避免 device 之间不必要的数据传输。应用可以使用 Direct3D 的 `cudaD3D[9|10|11]GetDevices()` 和 OpenGL 的 `cudaGLGetDevices()` 调用族，识别当前帧和下一帧中执行渲染的 device 对应的 CUDA device handle。随后，应用通常会选择合适的 device，并将 Direct3D 或 OpenGL 资源映射到 `cudaD3D[9|10|11]GetDevices()` 或 `cudaGLGetDevices()` 返回的 CUDA device；此时 `deviceList` 参数应设置为 `cudaD3D[9|10|11]DeviceListCurrentFrame` 或 `cudaGLDeviceListCurrentFrame`。
- `cudaGraphicsD3D[9|10|11]RegisterResource` 和 `cudaGraphicsGLRegister[Buffer|Image]` 返回的资源只能在发生注册的 device 上使用。因此，在不同帧由不同 CUDA device 计算数据的 SLI 配置中，必须分别为每个 device 注册资源。

## 4.19.2 外部资源互操作（External resource interoperability）

外部资源互操作允许 CUDA 导入由其他 API 显式导出的资源。这些对象通常使用操作系统原生句柄导出，例如 Linux 上的文件描述符或 Windows 上的 NT handle。这样可以在其他 API 和 CUDA 之间高效共享资源，而无需复制或重复创建。该功能支持 Direct3D[11–12]、Vulkan 和 NVIDIA Software Communication Interface Interoperability。可导入的资源有两类：

- **内存对象**：使用 `cudaImportExternalMemory()` 导入。之后，可以使用映射到该内存对象上的 device pointer（通过 `cudaExternalMemoryGetMappedBuffer()` 获取）或 CUDA mipmapped array（通过 `cudaExternalMemoryGetMappedMipmappedArray()` 获取）在 Kernel 中访问。根据内存对象类型，一个内存对象可能允许建立多个映射；这些映射必须与导出 API 建立的映射匹配，任何不匹配都会产生未定义行为。导入的内存对象必须使用 `cudaDestroyExternalMemory()` 释放。释放内存对象不会释放其映射，因此必须显式使用 `cudaFree()` 释放映射到该对象上的 device pointer，并使用 `cudaFreeMipmappedArray()` 释放映射到该对象上的 CUDA mipmapped array。对象销毁后访问其映射是非法的。
- **同步对象**：使用 `cudaImportExternalSemaphore()` 导入。导入后，可以使用 `cudaSignalExternalSemaphoresAsync()` 发出信号，并使用 `cudaWaitExternalSemaphoresAsync()` 等待信号。在对应 signal 发出之前发起 wait 是非法的。根据导入的同步对象类型，signal 和 wait 的方式还可能受到后续小节所述的额外限制。导入的 semaphore 对象必须使用 `cudaDestroyExternalSemaphore()` 释放；销毁 semaphore 之前，所有未完成的 signal 和 wait 都必须完成。

### 4.19.2.1 Vulkan 互操作（Vulkan interoperability）

在同一硬件上耦合执行 Vulkan 图形和计算工作负载，可以提高 GPU 利用率并避免不必要的拷贝。本节不是 Vulkan 教程，只聚焦 CUDA 互操作；Vulkan 教程请参阅 [Vulkan Tutorials](https://www.vulkan.org/learn#vulkan-tutorials)。

实现 Vulkan–CUDA 互操作的主要步骤是：

1. 初始化 Vulkan，创建并导出外部 buffer 和/或同步对象。
2. 通过匹配的 device UUID，将运行 Vulkan 的 CUDA device 设置为对应 device。
3. 获取内存和/或同步句柄。
4. 在 CUDA 中使用这些句柄导入内存和/或同步对象。
5. 将 device pointer 或 mipmapped array 映射到内存对象。
6. 在 CUDA 和 Vulkan 中交替使用导入的内存对象，并通过对同步对象执行 signal 和 wait 定义执行顺序。

下面借助 `simpleVulkan` 示例 [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) 逐步解释这些步骤，重点关注 CUDA 互操作所需的部分；部分变体使用独立代码片段说明。

> **注意**
>
> 本节代码示例使用直接的内存分配和资源创建方式。由于可创建实例数量受限等原因，这不是当前最先进的方式。不过，要理解互操作，仍需要了解底层 Vulkan 代码和具体标志。对于使用 VulkanMemoryAllocator 的更现代示例，请参阅 NVProSamples 仓库中的 `sample_cuda_interop`。

示例中一直使用以下数据结构：

```cpp
class VulkanCudaSineWave : public VulkanBaseApp {
  typedef struct UniformBufferObject_st {
    mat4x4 modelViewProj;
  } UniformBufferObject;

  VkBuffer m_heightBuffer, m_xyBuffer, m_indexBuffer;
  VkDeviceMemory m_heightMemory, m_xyMemory, m_indexMemory;
  UniformBufferObject m_ubo;
  VkSemaphore m_vkWaitSemaphore, m_vkSignalSemaphore;
  SineWaveSimulation m_sim;
  cudaStream_t m_stream;
  cudaExternalSemaphore_t m_cudaWaitSemaphore, m_cudaSignalSemaphore, m_cudaTimelineSemaphore;
  cudaExternalMemory_t m_cudaVertMem;
  float *m_cudaHeightMap;
  // ...
};
```

#### 4.19.2.1.1 设置 Vulkan device（Setting up a Vulkan device）

要导出内存对象，必须启用 `VK_KHR_external_memory_capabilities` 扩展创建 Vulkan instance，并启用 `VK_KHR_external_memory` 创建 device。此外，还必须启用与平台相关的句柄类型：Windows 使用 `VK_KHR_external_memory_win32`，基于 UNIX 的系统使用 `VK_KHR_external_memory_fd`。

类似地，要导出同步对象，需要在 device 级别启用 `VK_KHR_external_semaphore_capabilities`，在 instance 级别启用 `VK_KHR_external_semaphore`，并启用句柄对应的平台扩展：Windows 使用 `VK_KHR_external_semaphore_win32`，Unix 系统使用 `VK_KHR_external_semaphore_fd`。

在 `simpleVulkan` 示例中，这些扩展通过以下枚举启用：

```cpp
std::vector getRequiredExtensions() const {
  std::vector extensions;
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
  return extensions;
}

std::vector getRequiredDeviceExtensions() const {
  std::vector extensions;
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
#ifdef _WIN64
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif /* _WIN64 */
  return extensions;
}
```

随后将这些扩展加入 Vulkan instance 和 device 创建信息中；具体细节请参阅 `simpleVulkan` 示例。

#### 4.19.2.1.2 使用匹配的 device UUID 初始化 CUDA（Initializing CUDA with matching device UUIDs）

导入 Vulkan 导出的内存和同步对象时，必须在创建它们的同一 device 上导入和映射。可以比较 CUDA device 的 UUID 与创建对象的 Vulkan physical device 的 UUID，确定对应的 CUDA device。下面的代码来自 `simpleVulkan` 示例，其中 `vkDeviceUUID` 是 Vulkan API 结构 `vkPhysicalDeviceIDProperties.deviceUUID` 的成员，表示当前 Vulkan instance 的 physical device ID。

```cpp
// from the CUDA example `simpleVulkan`
int SineWaveSimulation::initCuda(uint8_t *vkDeviceUUID, size_t UUID_SIZE) {
  int current_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;
  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);
    if ((deviceProp.computeMode != cudaComputeModeProhibited)) {
      int ret = memcmp((void *)&deviceProp.uuid, vkDeviceUUID, UUID_SIZE);
      if (ret == 0) {
        checkCudaErrors(cudaSetDevice(current_device));
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
               current_device, deviceProp.name, deviceProp.major,
               deviceProp.minor);
        return current_device;
      }
    } else {
      devices_prohibited++;
    }
    current_device++;
  }
  if (devices_prohibited == device_count) {
    fprintf(stderr, "CUDA error: No Vulkan-CUDA Interop capable GPU found.\n");
    exit(EXIT_FAILURE);
  }
  return -1;
}
```

注意，Vulkan physical device 不应属于包含多个 Vulkan physical device 的 device group。也就是说，`vkEnumeratePhysicalDeviceGroups` 返回的、包含给定 Vulkan physical device 的 device group，其 physical device 数量必须为 1。

#### 4.19.2.1.3 导出 Vulkan 内存对象（Exporting Vulkan memory objects）

要导出 Vulkan 内存对象，必须使用相应导出标志创建 buffer。句柄类型的枚举值与平台相关。

```cpp
void VulkanBaseApp::createExternalBuffer(
    VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkExternalMemoryHandleTypeFlagsKHR extMemHandleType, VkBuffer &buffer,
    VkDeviceMemory &bufferMemory) {
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkExternalMemoryBufferCreateInfo externalMemoryBufferInfo = {};
  externalMemoryBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  externalMemoryBufferInfo.handleTypes = extMemHandleType;
  bufferInfo.pNext = &externalMemoryBufferInfo;
  if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to create buffer!");
  }
  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);
  VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
  vulkanExportMemoryAllocateInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
  vulkanExportMemoryAllocateInfoKHR.handleTypes = extMemHandleType;
  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.pNext = &vulkanExportMemoryAllocateInfoKHR;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      m_physicalDevice, memRequirements.memoryTypeBits, properties);
  if (vkAllocateMemory(m_device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate external buffer memory!");
  }
  vkBindBufferMemory(m_device, buffer, bufferMemory, 0);
}
```

#### 4.19.2.1.4 导出 Vulkan 同步对象（Exporting Vulkan synchronization objects）

在 GPU 上执行的 Vulkan API 调用是异步的。Vulkan 提供 semaphore 和 fence 来定义执行顺序，这些对象可以与 CUDA 共享。和内存对象类似，semaphore 由 Vulkan 导出时必须根据 semaphore 类型使用对应导出标志创建。Vulkan 有 binary semaphore 和 timeline semaphore：binary semaphore 只有 1 bit 计数器，表示已 signal 或未 signal；timeline semaphore 有 64 bit 计数器，可以使用同一个 semaphore 定义执行顺序。`simpleVulkan` 示例同时包含 timeline 和 binary semaphore 的代码路径。

```cpp
void VulkanBaseApp::createExternalSemaphore(
    VkSemaphore &semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType) {
  VkSemaphoreCreateInfo semaphoreInfo = {};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkExportSemaphoreCreateInfoKHR exportSemaphoreCreateInfo = {};
  exportSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
#ifdef _VK_TIMELINE_SEMAPHORE
  VkSemaphoreTypeCreateInfo timelineCreateInfo;
  timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timelineCreateInfo.pNext = NULL;
  timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timelineCreateInfo.initialValue = 0;
  exportSemaphoreCreateInfo.pNext = &timelineCreateInfo;
#else
  exportSemaphoreCreateInfo.pNext = NULL;
#endif /* _VK_TIMELINE_SEMAPHORE */
  exportSemaphoreCreateInfo.handleTypes = handleType;
  semaphoreInfo.pNext = &exportSemaphoreCreateInfo;
  if (vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
    throw std::runtime_error("failed to create synchronization objects for a CUDA-Vulkan!");
  }
}
```

#### 4.19.2.1.5 导入内存对象（Importing memory objects）

Vulkan 导出的 dedicated 和 non-dedicated 内存对象都可以导入 CUDA。导入 Vulkan dedicated memory object 时，必须设置 `cudaExternalMemoryDedicated` 标志。

在 Windows 中，使用 `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT` 导出的 Vulkan 内存对象，可以通过关联的 NT handle 导入 CUDA。CUDA 不接管 NT handle 的所有权；应用在不再需要该 handle 时负责关闭它。NT handle 持有资源引用，因此必须在底层内存释放前显式释放。

在 Linux 中，使用 `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT` 导出的 Vulkan 内存对象，可以通过关联的 file descriptor 导入 CUDA。成功导入后 CUDA 接管 file descriptor 的所有权；成功导入后继续使用该 file descriptor 会产生未定义行为。

```cpp
void importCudaExternalMemory(void **cudaPtr, cudaExternalMemory_t &cudaMem,
                              VkDeviceMemory &vkMem, VkDeviceSize size,
                              VkExternalMemoryHandleTypeFlagBits handleType) {
  cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
  if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
  } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
  } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  } else {
    throw std::runtime_error("Unknown handle type requested!");
  }
  externalMemoryHandleDesc.size = size;
#ifdef _WIN64
  externalMemoryHandleDesc.handle.win32.handle =
      (HANDLE)getMemHandle(vkMem, handleType);
#else
  externalMemoryHandleDesc.handle.fd =
      (int)(uintptr_t)getMemHandle(vkMem, handleType);
#endif
  checkCudaErrors(cudaImportExternalMemory(&cudaMem, &externalMemoryHandleDesc));
}
```

使用 Windows named handle 也可以导入使用 `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT` 导出的 Vulkan 内存对象：

```cpp
cudaExternalMemory_t importVulkanMemoryObjectFromNamedNTHandle(
    LPCWSTR name, unsigned long long size, bool isDedicated) {
  cudaExternalMemory_t extMem = NULL;
  cudaExternalMemoryHandleDesc desc = {};
  memset(&desc, 0, sizeof(desc));
  desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
  desc.handle.win32.name = (void *)name;
  desc.size = size;
  if (isDedicated) {
    desc.flags |= cudaExternalMemoryDedicated;
  }
  cudaImportExternalMemory(&extMem, &desc);
  return extMem;
}
```

#### 4.19.2.1.6 将 buffer 映射到已导入内存对象（Mapping buffers onto imported memory objects）

导入内存对象后，必须先映射才能使用。下面展示如何将 device pointer 映射到导入的内存对象。映射的 offset 和 size 必须与对应 Vulkan API 创建映射时指定的值一致。所有已映射的 device pointer 都必须使用 `cudaFree()` 释放。

```cpp
cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
externalMemBufferDesc.offset = 0;
externalMemBufferDesc.size = size;
externalMemBufferDesc.flags = 0;
checkCudaErrors(cudaExternalMemoryGetMappedBuffer(cudaPtr, cudaMem,
                                                  &externalMemBufferDesc));
```

#### 4.19.2.1.7 将 mipmapped array 映射到已导入内存对象（Mapping mipmapped arrays onto imported memory objects）

可以将 CUDA mipmapped array 映射到导入的内存对象。映射的 offset、dimensions、format 和 mip level 数量必须与对应 Vulkan API 创建映射时指定的值一致。此外，如果 Vulkan 中将 mipmapped array 绑定为 color target，则必须设置 `cudaArrayColorAttachment` 标志。所有已映射的 mipmapped array 都必须使用 `cudaFreeMipmappedArray()` 释放。下面的独立代码片段展示了将 Vulkan 参数转换为映射 mipmapped array 时对应 CUDA 参数的方法。

```cpp
cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(
    cudaExternalMemory_t extMem, unsigned long long offset,
    cudaChannelFormatDesc *formatDesc, cudaExtent *extent,
    unsigned int flags, unsigned int numLevels) {
  cudaMipmappedArray_t mipmap = NULL;
  cudaExternalMemoryMipmappedArrayDesc desc = {};
  memset(&desc, 0, sizeof(desc));
  desc.offset = offset;
  desc.formatDesc = *formatDesc;
  desc.extent = *extent;
  desc.flags = flags;
  desc.numLevels = numLevels;
  cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);
  return mipmap;
}

cudaChannelFormatDesc getCudaChannelFormatDescForVulkanFormat(VkFormat format) {
  cudaChannelFormatDesc d;
  memset(&d, 0, sizeof(d));
  switch (format) {
    case VK_FORMAT_R8_UINT: d.x = 8; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R8_SINT: d.x = 8; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindSigned; break;
    case VK_FORMAT_R8G8_UINT: d.x = 8; d.y = 8; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R8G8_SINT: d.x = 8; d.y = 8; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindSigned; break;
    case VK_FORMAT_R8G8B8A8_UINT: d.x = 8; d.y = 8; d.z = 8; d.w = 8; d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R8G8B8A8_SINT: d.x = 8; d.y = 8; d.z = 8; d.w = 8; d.f = cudaChannelFormatKindSigned; break;
    case VK_FORMAT_R16_UINT: d.x = 16; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R16_SINT: d.x = 16; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindSigned; break;
    case VK_FORMAT_R32_UINT: d.x = 32; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R32_SINT: d.x = 32; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindSigned; break;
    case VK_FORMAT_R32_SFLOAT: d.x = 32; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindFloat; break;
    case VK_FORMAT_R32G32B32A32_UINT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R32G32B32A32_SINT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindSigned; break;
    case VK_FORMAT_R32G32B32A32_SFLOAT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindFloat; break;
    default: assert(0);
  }
  return d;
}

cudaExtent getCudaExtentForVulkanExtent(VkExtent3D vkExt, uint32_t arrayLayers,
                                        VkImageViewType vkImageViewType) {
  cudaExtent e = {0, 0, 0};
  switch (vkImageViewType) {
    case VK_IMAGE_VIEW_TYPE_1D: e.width = vkExt.width; break;
    case VK_IMAGE_VIEW_TYPE_2D: e.width = vkExt.width; e.height = vkExt.height; break;
    case VK_IMAGE_VIEW_TYPE_3D: e.width = vkExt.width; e.height = vkExt.height; e.depth = vkExt.depth; break;
    case VK_IMAGE_VIEW_TYPE_CUBE: e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: e.width = vkExt.width; e.depth = arrayLayers; break;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
    default: assert(0);
  }
  return e;
}

unsigned int getCudaMipmappedArrayFlagsForVulkanImage(
    VkImageViewType vkImageViewType, VkImageUsageFlags vkImageUsageFlags,
    bool allowSurfaceLoadStore) {
  unsigned int flags = 0;
  switch (vkImageViewType) {
    case VK_IMAGE_VIEW_TYPE_CUBE: flags |= cudaArrayCubemap; break;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: flags |= cudaArrayCubemap | cudaArrayLayered; break;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: flags |= cudaArrayLayered; break;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: flags |= cudaArrayLayered; break;
    default: break;
  }
  if (vkImageUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
    flags |= cudaArrayColorAttachment;
  }
  if (allowSurfaceLoadStore) {
    flags |= cudaArraySurfaceLoadStore;
  }
  return flags;
}
```

#### 4.19.2.1.8 导入同步对象（Importing Synchronization Objects）

使用 `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT` 导出的 Vulkan semaphore，可以通过关联的 file descriptor 导入 CUDA；成功导入后 CUDA 接管 file descriptor 的所有权，继续使用它会产生未定义行为。

使用 `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT` 导出的 Vulkan semaphore，可以通过关联的 NT handle 导入 CUDA；CUDA 不接管 NT handle 的所有权，应用在不再需要它时负责关闭。NT handle 持有资源引用，因此必须在底层 semaphore 释放前显式释放。

使用 `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT` 导出的 Vulkan semaphore，可以通过关联的全局共享 D3DKMT handle 导入 CUDA。由于全局共享 D3DKMT handle 不持有底层 semaphore 的引用，因此当资源的所有其他引用都销毁后，它会自动销毁。

```cpp
void importCudaExternalSemaphore(
    cudaExternalSemaphore_t &cudaSem, VkSemaphore &vkSem,
    VkExternalSemaphoreHandleTypeFlagBits handleType) {
  cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {};
#ifdef _VK_TIMELINE_SEMAPHORE
  if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
    externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
  } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
    externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
  } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
    externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  }
#else
  if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
    externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
  } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
    externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
  } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
    externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
  }
#endif /* _VK_TIMELINE_SEMAPHORE */
  else {
    throw std::runtime_error("Unknown handle type requested!");
  }
#ifdef _WIN64
  externalSemaphoreHandleDesc.handle.win32.handle =
      (HANDLE)getSemaphoreHandle(vkSem, handleType);
#else
  externalSemaphoreHandleDesc.handle.fd =
      (int)(uintptr_t)getSemaphoreHandle(vkSem, handleType);
#endif
  externalSemaphoreHandleDesc.flags = 0;
  checkCudaErrors(cudaImportExternalSemaphore(&cudaSem,
                                              &externalSemaphoreHandleDesc));
}
```

#### 4.19.2.1.9 对已导入同步对象发出信号/等待（Signaling/Waiting on Imported Synchronization Objects）

导入的 Vulkan semaphore 可以按下面的方式发出信号并等待。对 semaphore 发出信号会将其置为 signaled 状态；对于 timeline semaphore，还会把计数器设置为 signal 调用指定的值。与该 signal 对应的 wait 必须在 Vulkan 中发出。此外，对于 binary semaphore，对该 signal 的 wait 必须在 signal 发出后才能发出。

等待 semaphore 会一直等待，直到它达到 signaled 状态或指定的等待值。已 signaled 的 binary semaphore 随后会被重置为 unsignaled 状态。对应的 signal 必须在 Vulkan 中发出；对于 binary semaphore，该 signal 必须先于 wait 发出。

在下面来自 `simpleVulkan` 的代码中，只有当 Vulkan 已经对顶点 buffer 周围的 semaphore 发出信号后，才会执行 simulation step/CUDA Kernel。simulation step 完成后，CUDA 会对另一个 semaphore 发出信号，或者对于 timeline semaphore 增加同一个 semaphore 的值，使等待该 semaphore 的 Vulkan 部分继续使用更新后的顶点 buffer 进行渲染。

```cpp
#ifdef _VK_TIMELINE_SEMAPHORE
static uint64_t waitValue = 1;
static uint64_t signalValue = 2;
cudaExternalSemaphoreWaitParams waitParams = {};
waitParams.flags = 0;
waitParams.params.fence.value = waitValue;
cudaExternalSemaphoreSignalParams signalParams = {};
signalParams.flags = 0;
signalParams.params.fence.value = signalValue;
checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_cudaTimelineSemaphore,
                                                &waitParams, 1, m_stream));
m_sim.stepSimulation(time, m_stream);
checkCudaErrors(cudaSignalExternalSemaphoresAsync(
    &m_cudaTimelineSemaphore, &signalParams, 1, m_stream));
waitValue += 2;
signalValue += 2;
#else
cudaExternalSemaphoreWaitParams waitParams = {};
waitParams.flags = 0;
waitParams.params.fence.value = 0;
cudaExternalSemaphoreSignalParams signalParams = {};
signalParams.flags = 0;
signalParams.params.fence.value = 0;
checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_cudaWaitSemaphore,
                                                &waitParams, 1, m_stream));
m_sim.stepSimulation(time, m_stream);
checkCudaErrors(cudaSignalExternalSemaphoresAsync(
    &m_cudaSignalSemaphore, &signalParams, 1, m_stream));
#endif /* _VK_TIMELINE_SEMAPHORE */
```

#### 4.19.2.1.10 OpenGL 互操作（OpenGL Interoperability）

传统的 OpenGL–CUDA 互操作如[OpenGL 互操作](#41911-opengl-互操作opengl-interoperability)所述，由 CUDA 直接消费 OpenGL 创建的句柄实现。然而，OpenGL 也可以消费 Vulkan 创建的内存和同步对象，因此存在另一种 OpenGL–CUDA 互操作方式：将 Vulkan 导出的内存和同步对象同时导入 OpenGL 与 CUDA，然后用它们协调 OpenGL 与 CUDA 的内存访问。关于导入 Vulkan 导出的内存和同步对象，请参阅以下 OpenGL 扩展：

- `GL_EXT_memory_object`
- `GL_EXT_memory_object_fd`
- `GL_EXT_memory_object_win32`
- `GL_EXT_semaphore`
- `GL_EXT_semaphore_fd`
- `GL_EXT_semaphore_win32`

### 4.19.2.2 Direct3D 互操作（Direct3D Interoperability）

CUDA 支持将 Direct3D[11|12] 资源导入 Direct3D11 和 Direct3D12。本节只讨论 Direct3D12；关于 Direct3D11，请参阅 CUDA Programming Guide 12.9。

#### 4.19.2.2.1 匹配 device LUID（Matching Device LUIDs）

导入 Direct3D12 导出的内存和同步对象时，必须在创建对象的同一 device 上导入和映射。可以比较 CUDA device 的 LUID 与创建对象的 Direct3D12 device 的 LUID，确定对应的 CUDA device。Direct3D12 device 不能创建在 linked node adapter 上，即 `ID3D12Device::GetNodeCount` 返回的 node count 必须为 1。

```cpp
int getCudaDeviceForD3D12Device(ID3D12Device *d3d12Device) {
    LUID d3d12Luid = d3d12Device->GetAdapterLuid();
    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);
    for (int cudaDevice = 0; cudaDevice < cudaDeviceCount; cudaDevice++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, cudaDevice);
        char *cudaLuid = deviceProp.luid;
        if (!memcmp(&d3d12Luid.LowPart, cudaLuid, sizeof(d3d12Luid.LowPart)) &&
            !memcmp(&d3d12Luid.HighPart, cudaLuid + sizeof(d3d12Luid.LowPart), sizeof(d3d12Luid.HighPart))) {
            return cudaDevice;
        }
    }
    return cudaInvalidDeviceId;
}
```

#### 4.19.2.2.2 导入内存对象（Importing Memory Objects）

从 NT handle 导入内存对象有几种方式。应用在不再需要 NT handle 时负责关闭它。NT handle 持有资源引用，因此必须在底层内存释放前显式释放。导入 Direct3D 资源时，必须像下面的片段一样设置 `cudaExternalMemoryDedicated` 标志。

由 `ID3D12Device::CreateHeap` 调用中设置 `D3D12_HEAP_FLAG_SHARED` 创建的可共享 Direct3D12 heap memory object，可以通过其 NT handle 导入 CUDA：

```cpp
cudaExternalMemory_t importD3D12HeapFromNTHandle(HANDLE handle, unsigned long long size) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
    desc.handle.win32.handle = (void *)handle;
    desc.size = size;
    cudaImportExternalMemory(&extMem, &desc);
    CloseHandle(handle);
    return extMem;
}
```

如果存在 named handle，也可以使用它导入可共享 Direct3D12 heap memory object：

```cpp
cudaExternalMemory_t importD3D12HeapFromNamedNTHandle(LPCWSTR name, unsigned long long size) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
    desc.handle.win32.name = (void *)name;
    desc.size = size;
    cudaImportExternalMemory(&extMem, &desc);
    return extMem;
}
```

由 `D3D12Device::CreateCommittedResource` 调用中设置 `D3D12_HEAP_FLAG_SHARED` 创建的可共享 Direct3D12 committed resource，可以通过其 NT handle 导入 CUDA。导入 Direct3D12 committed resource 时，必须设置 `cudaExternalMemoryDedicated`：

```cpp
cudaExternalMemory_t importD3D12CommittedResourceFromNTHandle(
    HANDLE handle, unsigned long long size) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    desc.handle.win32.handle = (void *)handle;
    desc.size = size;
    desc.flags |= cudaExternalMemoryDedicated;
    cudaImportExternalMemory(&extMem, &desc);
    CloseHandle(handle);
    return extMem;
}
```

如果存在 named handle，也可以使用它导入可共享 Direct3D12 committed resource：

```cpp
cudaExternalMemory_t importD3D12CommittedResourceFromNamedNTHandle(
    LPCWSTR name, unsigned long long size) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    desc.handle.win32.name = (void *)name;
    desc.size = size;
    desc.flags |= cudaExternalMemoryDedicated;
    cudaImportExternalMemory(&extMem, &desc);
    return extMem;
}
```

#### 4.19.2.2.3 将 buffer 映射到已导入内存对象（Mapping Buffers onto Imported Memory Objects）

可以将 device pointer 映射到导入的内存对象。映射的 offset 和 size 必须与对应 Direct3D12 API 创建映射时指定的值一致。所有已映射 device pointer 都必须使用 `cudaFree()` 释放。

```cpp
void *mapBufferOntoExternalMemory(cudaExternalMemory_t extMem,
                                   unsigned long long offset,
                                   unsigned long long size) {
    void *ptr = NULL;
    cudaExternalMemoryBufferDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.offset = offset;
    desc.size = size;
    cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);
    return ptr;
}
```

#### 4.19.2.2.4 将 mipmapped array 映射到已导入内存对象（Mapping Mipmapped Arrays onto Imported Memory Objects）

可以将 CUDA mipmapped array 映射到导入的内存对象。映射的 offset、dimensions、format 和 mip level 数量必须与对应 Direct3D12 API 创建映射时指定的值一致。此外，如果 Direct3D12 中可以将 mipmapped array 绑定为 render target，则必须设置 `cudaArrayColorAttachment` 标志。所有已映射的 mipmapped array 都必须使用 `cudaFreeMipmappedArray()` 释放。下面的代码展示如何将参数转换为映射 mipmapped array 时对应的 CUDA 参数。

```cpp
cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(
    cudaExternalMemory_t extMem, unsigned long long offset,
    cudaChannelFormatDesc *formatDesc, cudaExtent *extent,
    unsigned int flags, unsigned int numLevels) {
    cudaMipmappedArray_t mipmap = NULL;
    cudaExternalMemoryMipmappedArrayDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.offset = offset;
    desc.formatDesc = *formatDesc;
    desc.extent = *extent;
    desc.flags = flags;
    desc.numLevels = numLevels;
    cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);
    return mipmap;
}

cudaChannelFormatDesc getCudaChannelFormatDescForDxgiFormat(DXGI_FORMAT dxgiFormat) {
    cudaChannelFormatDesc d;
    memset(&d, 0, sizeof(d));
    switch (dxgiFormat) {
        case DXGI_FORMAT_R8_UINT: d.x = 8; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindUnsigned; break;
        case DXGI_FORMAT_R8_SINT: d.x = 8; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindSigned; break;
        case DXGI_FORMAT_R8G8_UINT: d.x = 8; d.y = 8; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindUnsigned; break;
        case DXGI_FORMAT_R8G8_SINT: d.x = 8; d.y = 8; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindSigned; break;
        case DXGI_FORMAT_R8G8B8A8_UINT: d.x = 8; d.y = 8; d.z = 8; d.w = 8; d.f = cudaChannelFormatKindUnsigned; break;
        case DXGI_FORMAT_R8G8B8A8_SINT: d.x = 8; d.y = 8; d.z = 8; d.w = 8; d.f = cudaChannelFormatKindSigned; break;
        case DXGI_FORMAT_R16_UINT: d.x = 16; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindUnsigned; break;
        case DXGI_FORMAT_R16_SINT: d.x = 16; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindSigned; break;
        case DXGI_FORMAT_R32_UINT: d.x = 32; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindUnsigned; break;
        case DXGI_FORMAT_R32_SINT: d.x = 32; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindSigned; break;
        case DXGI_FORMAT_R32_FLOAT: d.x = 32; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindFloat; break;
        case DXGI_FORMAT_R32G32B32A32_UINT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindUnsigned; break;
        case DXGI_FORMAT_R32G32B32A32_SINT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindSigned; break;
        case DXGI_FORMAT_R32G32B32A32_FLOAT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindFloat; break;
        default: assert(0);
    }
    return d;
}

cudaExtent getCudaExtentForD3D12Extent(
    UINT64 width, UINT height, UINT16 depthOrArraySize,
    D3D12_SRV_DIMENSION d3d12SRVDimension) {
    cudaExtent e = {0, 0, 0};
    switch (d3d12SRVDimension) {
        case D3D12_SRV_DIMENSION_TEXTURE1D: e.width = width; break;
        case D3D12_SRV_DIMENSION_TEXTURE2D: e.width = width; e.height = height; break;
        case D3D12_SRV_DIMENSION_TEXTURE3D: e.width = width; e.height = height; e.depth = depthOrArraySize; break;
        case D3D12_SRV_DIMENSION_TEXTURECUBE: e.width = width; e.height = height; e.depth = depthOrArraySize; break;
        case D3D12_SRV_DIMENSION_TEXTURE1DARRAY: e.width = width; e.depth = depthOrArraySize; break;
        case D3D12_SRV_DIMENSION_TEXTURE2DARRAY: e.width = width; e.height = height; e.depth = depthOrArraySize; break;
        case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: e.width = width; e.height = height; e.depth = depthOrArraySize; break;
        default: assert(0);
    }
    return e;
}

unsigned int getCudaMipmappedArrayFlagsForD3D12Resource(
    D3D12_SRV_DIMENSION d3d12SRVDimension,
    D3D12_RESOURCE_FLAGS d3d12ResourceFlags, bool allowSurfaceLoadStore) {
    unsigned int flags = 0;
    switch (d3d12SRVDimension) {
        case D3D12_SRV_DIMENSION_TEXTURECUBE: flags |= cudaArrayCubemap; break;
        case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: flags |= cudaArrayCubemap | cudaArrayLayered; break;
        case D3D12_SRV_DIMENSION_TEXTURE1DARRAY: flags |= cudaArrayLayered; break;
        case D3D12_SRV_DIMENSION_TEXTURE2DARRAY: flags |= cudaArrayLayered; break;
        default: break;
    }
    if (d3d12ResourceFlags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) {
        flags |= cudaArrayColorAttachment;
    }
    if (allowSurfaceLoadStore) {
        flags |= cudaArraySurfaceLoadStore;
    }
    return flags;
}
```

#### 4.19.2.2.5 导入同步对象（Importing Synchronization Objects）

在调用 `ID3D12Device::CreateFence` 时设置 `D3D12_FENCE_FLAG_SHARED` 创建的可共享 Direct3D12 fence，可以使用关联的 NT handle 导入 CUDA。应用在不再需要 handle 时负责关闭它。NT handle 持有资源引用，因此必须在底层 semaphore 释放前显式释放。

```cpp
cudaExternalSemaphore_t importD3D12FenceFromNTHandle(HANDLE handle) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    desc.handle.win32.handle = handle;
    cudaImportExternalSemaphore(&extSem, &desc);
    CloseHandle(handle);
    return extSem;
}
```

如果存在 named handle，也可以使用它导入可共享 Direct3D12 fence：

```cpp
cudaExternalSemaphore_t importD3D12FenceFromNamedNTHandle(LPCWSTR name) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    desc.handle.win32.name = (void *)name;
    cudaImportExternalSemaphore(&extSem, &desc);
    return extSem;
}
```

#### 4.19.2.2.6 对已导入同步对象发出信号/等待（Signaling/Waiting on Imported Synchronization Objects）

从 Direct3D12 导入带 fence 的 semaphore 后，可以对它们发出信号并等待。

对 fence 对象发出信号会设置其值。等待该 signal 的对应 wait 必须在 Direct3D12 中发出，而且必须在 signal 发出之后发出。

```cpp
void signalExternalSemaphore(cudaExternalSemaphore_t extSem,
                              unsigned long long value, cudaStream_t stream) {
    cudaExternalSemaphoreSignalParams params = {};
    memset(&params, 0, sizeof(params));
    params.params.fence.value = value;
    cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);
}
```

fence 对象会一直等待，直到其值大于或等于指定值。它所等待的对应 signal 必须在 Direct3D12 中发出，而且必须在该 wait 发出前发出。

```cpp
void waitExternalSemaphore(cudaExternalSemaphore_t extSem,
                            unsigned long long value, cudaStream_t stream) {
    cudaExternalSemaphoreWaitParams params = {};
    memset(&params, 0, sizeof(params));
    params.params.fence.value = value;
    cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream);
}
```

### 4.19.2.3 NVIDIA Software Communication Interface 互操作（NVIDIA Software Communication Interface Interoperability，NVSCI）

NvSciBuf 和 NvSciSync 是为以下目的开发的接口：

- NvSciBuf：允许应用在内存中分配和交换 buffer。
- NvSciSync：允许应用在操作边界管理同步对象。

这些接口的更多信息请参阅 [NVIDIA DRIVE 文档](https://docs.nvidia.com/drive)。

#### 4.19.2.3.1 导入内存对象（Importing Memory Objects）

要分配与给定 CUDA device 兼容的 NvSciBuf 对象，必须像下面这样，在 NvSciBuf 属性列表中通过 `NvSciBufGeneralAttrKey_GpuId` 设置对应 GPU ID。应用还可以指定以下属性：

- `NvSciBufGeneralAttrKey_NeedCpuAccess`：指定 buffer 是否需要 CPU 访问。
- `NvSciBufRawBufferAttrKey_Align`：指定 `NvSciBufType_RawBuffer` 的对齐要求。
- `NvSciBufGeneralAttrKey_RequiredPerm`：每个 NvSciBuf 内存对象实例可以为不同 UMD 配置不同访问权限。例如，要让 GPU 只读访问 buffer，可以使用 `NvSciBufAccessPerm_Readonly` 作为输入参数调用 `NvSciBufObjDupWithReducePerm()`，创建具有更低权限的 NvSciBuf 副本，然后像下面这样将副本导入 CUDA。
- `NvSciBufGeneralAttrKey_EnableGpuCache`：控制 GPU L2 cacheability。
- `NvSciBufGeneralAttrKey_EnableGpuCompression`：指定 GPU compression。

> **注意**
>
> 关于这些属性及其有效输入选项的更多信息，请参阅 NvSciBuf 文档。

下面的代码片段展示了这些属性的示例用法：

```cpp
NvSciBufObj createNvSciBufObject() {
    NvSciBufType bufType = NvSciBufType_RawBuffer;
    uint64_t rawsize = SIZE;
    uint64_t align = 0;
    bool cpuaccess_flag = true;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
    NvSciRmGpuId gpuid[] ={};
    CUuuid uuid;
    cuDeviceGetUuid(&uuid, dev);
    memcpy(&gpuid[0].bytes, &uuid.bytes, sizeof(uuid.bytes));
    NvSciBufAttrValGpuCache gpuCache[] = {{gpuid[0], false}};
    NvSciBufAttrValGpuCompression gpuCompression[] = {{gpuid[0], NvSciBufCompressionType_GenericCompressible}};
    NvSciBufAttrKeyValuePair rawbuffattrs[] = {
         { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
         { NvSciBufRawBufferAttrKey_Size, &rawsize, sizeof(rawsize) },
         { NvSciBufRawBufferAttrKey_Align, &align, sizeof(align) },
         { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag, sizeof(cpuaccess_flag) },
         { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
         { NvSciBufGeneralAttrKey_GpuId, &gpuid, sizeof(gpuid) },
         { NvSciBufGeneralAttrKey_EnableGpuCache, &gpuCache, sizeof(gpuCache) },
         { NvSciBufGeneralAttrKey_EnableGpuCompression, &gpuCompression, sizeof(gpuCompression) }
    };
    err = NvSciBufAttrListSetAttrs(attrListBuffer, rawbuffattrs,
            sizeof(rawbuffattrs)/sizeof(NvSciBufAttrKeyValuePair));
    NvSciBufAttrListCreate(NvSciBufModule, &attrListBuffer);
    NvSciBufAttrListReconcile(&attrListBuffer, 1, &attrListReconciledBuffer,
                       &attrListConflictBuffer);
    NvSciBufObjAlloc(attrListReconciledBuffer, &bufferObjRaw);
    return bufferObjRaw;
}
```

```cpp
NvSciBufObj bufferObjRo; // Readonly NvSciBuf memory obj
NvSciBufObjDupWithReducePerm(bufferObjRaw, NvSciBufAccessPerm_Readonly, &bufferObjRo);
return bufferObjRo;
```

可以使用 NvSciBufObj handle 将已分配的 NvSciBuf 内存对象导入 CUDA。应用应查询已分配的 NvSciBufObj，获取填充 CUDA External Memory Descriptor 所需的属性。属性列表和 NvSciBuf 对象必须由应用维护。如果导入 CUDA 的 NvSciBuf 对象也被其他驱动映射，则应用必须根据输出属性 `NvSciBufGeneralAttrKey_GpuSwNeedCacheCoherency` 的值，适当使用 NvSciSync 对象作为屏障，以保持 CUDA 与其他驱动之间的一致性。

> **注意**
>
> 关于分配和维护 NvSciBuf 对象的更多信息，请参阅 [NvSciBuf API 文档](https://developer.nvidia.com/docs/drive/drive-os/6.0.6/public/drive-os-linux-sdk/common/topics/nvsci/NvStreams1.html)。

```cpp
cudaExternalMemory_t importNvSciBufObject(NvSciBufObj bufferObjRaw) {
    NvSciBufAttrKeyValuePair bufattrs[] = {
        { NvSciBufRawBufferAttrKey_Size, NULL, 0 },
        { NvSciBufGeneralAttrKey_GpuSwNeedCacheCoherency, NULL, 0 },
        { NvSciBufGeneralAttrKey_EnableGpuCompression, NULL, 0 }
    };
    NvSciBufAttrListGetAttrs(retList, bufattrs,
        sizeof(bufattrs)/sizeof(NvSciBufAttrKeyValuePair));
    ret_size = *(static_cast<const uint64_t*>(bufattrs[0].value));
    int numGpus = bufattrs[1].len / sizeof(NvSciBufAttrValGpuCache);
    NvSciBufAttrValGpuCache[] cacheVal = (NvSciBufAttrValGpuCache *)bufattrs[1].value;
    bool ret_cacheVal;
    for (int i = 0; i < numGpus; i++) {
        if (memcmp(gpuid[0].bytes, cacheVal[i].gpuId.bytes, sizeof(CUuuid)) == 0) {
            ret_cacheVal = cacheVal[i].cacheability;
        }
    }
    numGpus = bufattrs[2].len / sizeof(NvSciBufAttrValGpuCompression);
    NvSciBufAttrValGpuCompression[] compVal = (NvSciBufAttrValGpuCompression *)bufattrs[2].value;
    NvSciBufCompressionType ret_compVal;
    for (int i = 0; i < numGpus; i++) {
        if (memcmp(gpuid[0].bytes, compVal[i].gpuId.bytes, sizeof(CUuuid)) == 0) {
            ret_compVal = compVal[i].compressionType;
        }
    }
    cudaExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    memHandleDesc.handle.nvSciBufObject = bufferObjRaw;
    memHandleDesc.handle.nvSciBufObject = bufferObjRo;
    memHandleDesc.size = ret_size;
    cudaImportExternalMemory(&extMemBuffer, &memHandleDesc);
    return extMemBuffer;
}
```

#### 4.19.2.3.2 将 buffer 映射到已导入内存对象（Mapping Buffers onto Imported Memory Objects）

可以将 device pointer 映射到导入的内存对象。映射的 offset 和 size 可以按照已分配 `NvSciBufObj` 的属性填充。所有已映射的 device pointer 都必须使用 `cudaFree()` 释放。

```cpp
void *mapBufferOntoExternalMemory(cudaExternalMemory_t extMem,
                                  unsigned long long offset,
                                  unsigned long long size) {
    void *ptr = NULL;
    cudaExternalMemoryBufferDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.offset = offset;
    desc.size = size;
    cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);
    return ptr;
}
```

#### 4.19.2.3.3 将 mipmapped array 映射到已导入内存对象（Mapping Mipmapped Arrays onto Imported Memory Objects）

可以将 CUDA mipmapped array 映射到导入的内存对象。offset、dimensions 和 format 可以按照已分配 `NvSciBufObj` 的属性填充。所有已映射的 mipmapped array 都必须使用 `cudaFreeMipmappedArray()` 释放。下面的代码展示如何将 NvSciBuf 属性转换为映射 mipmapped array 时对应的 CUDA 参数。

> **注意**
>
> mip level 数量必须为 1。

```cpp
cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(
    cudaExternalMemory_t extMem, unsigned long long offset,
    cudaChannelFormatDesc *formatDesc, cudaExtent *extent,
    unsigned int flags, unsigned int numLevels) {
    cudaMipmappedArray_t mipmap = NULL;
    cudaExternalMemoryMipmappedArrayDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.offset = offset;
    desc.formatDesc = *formatDesc;
    desc.extent = *extent;
    desc.flags = flags;
    desc.numLevels = numLevels;
    cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);
    return mipmap;
}
```

#### 4.19.2.3.4 导入同步对象（Importing Synchronization Objects）

可以使用 `cudaDeviceGetNvSciSyncAttributes()` 生成与给定 CUDA device 兼容的 NvSciSync 属性。返回的属性列表可以用来创建一个保证与给定 CUDA device 兼容的 `NvSciSyncObj`。

```cpp
NvSciSyncObj createNvSciSyncObject() {
    NvSciSyncObj nvSciSyncObj;
    int cudaDev0 = 0;
    int cudaDev1 = 1;
    NvSciSyncAttrList signalerAttrList = NULL;
    NvSciSyncAttrList waiterAttrList = NULL;
    NvSciSyncAttrList reconciledList = NULL;
    NvSciSyncAttrList newConflictList = NULL;
    NvSciSyncAttrListCreate(module, &signalerAttrList);
    NvSciSyncAttrListCreate(module, &waiterAttrList);
    NvSciSyncAttrList unreconciledList[2] = {NULL, NULL};
    unreconciledList[0] = signalerAttrList;
    unreconciledList[1] = waiterAttrList;
    cudaDeviceGetNvSciSyncAttributes(signalerAttrList, cudaDev0, CUDA_NVSCISYNC_ATTR_SIGNAL);
    cudaDeviceGetNvSciSyncAttributes(waiterAttrList, cudaDev1, CUDA_NVSCISYNC_ATTR_WAIT);
    NvSciSyncAttrListReconcile(unreconciledList, 2, &reconciledList, &newConflictList);
    NvSciSyncObjAlloc(reconciledList, &nvSciSyncObj);
    return nvSciSyncObj;
}
```

可以使用 NvSciSyncObj handle 将上述创建的 NvSciSync 对象导入 CUDA。即使导入完成，NvSciSyncObj handle 的所有权仍归应用所有。

```cpp
cudaExternalSemaphore_t importNvSciSyncObject(void* nvSciSyncObj) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    desc.handle.nvSciSyncObj = nvSciSyncObj;
    cudaImportExternalSemaphore(&extSem, &desc);
    // Deleting/Freeing the nvSciSyncObj beyond this point will lead to undefined behavior in CUDA
    return extSem;
}
```

#### 4.19.2.3.5 对已导入同步对象发出信号/等待（Signaling/Waiting on Imported Synchronization Objects）

可以按下面的方式对已导入的 `NvSciSyncObj` 发出信号。对 NvSciSync 支持的 semaphore 对象发出信号，会初始化作为输入传入的 *fence* 参数；对应的 wait 操作会等待该 fence。对该 signal 的 wait 必须在 signal 发出后发出。如果将标志设置为 `cudaExternalSemaphoreSignalSkipNvSciBufMemSync`，则默认情况下，signal 操作中执行的内存同步操作（针对该进程中导入的所有 NvSciBuf）会被跳过。当 `NvsciBufGeneralAttrKey_GpuSwNeedCacheCoherency` 为 FALSE 时，应设置该标志。

```cpp
void signalExternalSemaphore(cudaExternalSemaphore_t extSem,
                              cudaStream_t stream, void *fence) {
    cudaExternalSemaphoreSignalParams signalParams = {};
    memset(&signalParams, 0, sizeof(signalParams));
    signalParams.params.nvSciSync.fence = (void*)fence;
    signalParams.flags = 0; // OR cudaExternalSemaphoreSignalSkipNvSciBufMemSync
    cudaSignalExternalSemaphoresAsync(&extSem, &signalParams, 1, stream);
}
```

可以按下面的方式等待已导入的 `NvSciSyncObj`。对 NvSciSync 支持的 semaphore 对象执行 wait，会等待输入 *fence* 参数由对应 signaler 发出信号。发出 signal 必须先于该 wait。如果将标志设置为 `cudaExternalSemaphoreWaitSkipNvSciBufMemSync`，则默认情况下，signal 操作中执行的内存同步操作会被跳过。当 `NvsciBufGeneralAttrKey_GpuSwNeedCacheCoherency` 为 FALSE 时，应设置该标志。

```cpp
void waitExternalSemaphore(cudaExternalSemaphore_t extSem,
                           cudaStream_t stream, void *fence) {
    cudaExternalSemaphoreWaitParams waitParams = {};
    memset(&waitParams, 0, sizeof(waitParams));
    waitParams.params.nvSciSync.fence = (void*)fence;
    waitParams.flags = 0; // OR cudaExternalSemaphoreWaitSkipNvSciBufMemSync
    cudaWaitExternalSemaphoresAsync(&extSem, &waitParams, 1, stream);
}
```
