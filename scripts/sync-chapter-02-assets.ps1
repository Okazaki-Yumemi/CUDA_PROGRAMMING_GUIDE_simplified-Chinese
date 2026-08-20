$ErrorActionPreference = 'Stop'

$assetRoot = Join-Path $PSScriptRoot '..\docs\.vuepress\public\images\chapter-02'
New-Item -ItemType Directory -Force -Path $assetRoot | Out-Null

$assets = @{
  'grid-of-thread-blocks.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/grid-of-thread-blocks.png'
  'perfect_coalescing_32byte_segments.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/perfect_coalescing_32byte_segments.png'
  'no_coalescing_32byte_segments.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/no_coalescing_32byte_segments.png'
  'global_transpose.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/global_transpose.png'
  'examples-of-strided-shared-memory-accesses.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/examples-of-strided-shared-memory-accesses.png'
  'examples-of-irregular-shared-memory-accesses.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/examples-of-irregular-shared-memory-accesses.png'
  'bank-conflicts-shared-mem.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/bank-conflicts-shared-mem.png'
  'no-bank-conflicts-shared-mem.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/no-bank-conflicts-shared-mem.png'
  'cutile-tile-space-indexing.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/cutile-tile-space-indexing.png'
  'nvcc-flow.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/nvcc-flow.png'
  'nvcc-flow-multi-archs.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/nvcc-flow-multi-archs.png'
  'cuda_streams.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/cuda_streams.png'
  'unified-memory-explainer.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/unified-memory-explainer.png'
}

foreach ($asset in $assets.GetEnumerator()) {
  $destination = Join-Path $assetRoot $asset.Key
  Invoke-WebRequest -UseBasicParsing -Uri $asset.Value -OutFile $destination -TimeoutSec 60
  Write-Output ("downloaded {0} ({1} bytes)" -f $asset.Key, (Get-Item -LiteralPath $destination).Length)
}
