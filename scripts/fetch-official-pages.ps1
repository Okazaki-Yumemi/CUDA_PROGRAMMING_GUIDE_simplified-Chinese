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
}

foreach ($page in $pages.GetEnumerator()) {
  $destination = Join-Path $OutputRoot $page.Key
  $parent = Split-Path -Parent $destination
  New-Item -ItemType Directory -Force -Path $parent | Out-Null
  Invoke-WebRequest -UseBasicParsing -Uri $page.Value -OutFile $destination -TimeoutSec 90
  Write-Output ("downloaded {0} ({1} bytes)" -f $page.Key, (Get-Item -LiteralPath $destination).Length)
}
