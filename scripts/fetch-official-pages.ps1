param(
  [string]$OutputRoot = (Join-Path $PSScriptRoot '..\tmp\official-pages')
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$pages = [ordered]@{
  '01-introduction/introduction.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html'
  '01-introduction/programming-model.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html'
  '01-introduction/cuda-platform.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html'
  '02-basics/intro-to-cuda-cpp.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html'
  '02-basics/intro-to-cuda-python.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-python.html'
  '02-basics/writing-cuda-kernels.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html'
  '02-basics/writing-tile-kernels.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html'
  '02-basics/asynchronous-execution.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html'
  '02-basics/understanding-memory.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html'
  '02-basics/nvcc.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html'
  '03-advanced/advanced-host-programming.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html'
  '03-advanced/advanced-kernel-programming.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html'
  '03-advanced/driver-api.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html'
  '03-advanced/multi-gpu-systems.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html'
  '03-advanced/feature-survey.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html'
  '04-special-topics/unified-memory.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html'
  '04-special-topics/cuda-graphs.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html'
  '04-special-topics/stream-ordered-memory-allocation.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html'
  '04-special-topics/cooperative-groups.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html'
  '04-special-topics/programmatic-dependent-launch.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html'
  '04-special-topics/green-contexts.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html'
  '04-special-topics/lazy-loading.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html'
  '04-special-topics/error-log-management.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html'
  '04-special-topics/async-barriers.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html'
  '04-special-topics/pipelines.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html'
  '04-special-topics/async-copies.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html'
  '04-special-topics/cluster-launch-control.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html'
  '04-special-topics/l2-cache-control.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html'
  '04-special-topics/memory-sync-domains.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/memory-sync-domains.html'
  '04-special-topics/inter-process-communication.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html'
  '04-special-topics/virtual-memory-management.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html'
  '04-special-topics/extended-gpu-memory.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html'
  '04-special-topics/dynamic-parallelism.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html'
  '04-special-topics/graphics-interop.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html'
  '04-special-topics/driver-entry-point-access.html' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html'
}

foreach ($page in $pages.GetEnumerator()) {
  $destination = Join-Path $OutputRoot $page.Key
  $parent = Split-Path -Parent $destination
  New-Item -ItemType Directory -Force -Path $parent | Out-Null
  Invoke-WebRequest -UseBasicParsing -Uri $page.Value -OutFile $destination -TimeoutSec 90
  Write-Output ("downloaded {0} ({1} bytes)" -f $page.Key, (Get-Item -LiteralPath $destination).Length)
}
