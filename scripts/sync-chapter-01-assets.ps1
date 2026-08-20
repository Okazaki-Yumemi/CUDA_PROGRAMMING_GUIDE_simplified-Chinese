$ErrorActionPreference = 'Stop'

$assetRoot = Join-Path $PSScriptRoot '..\docs\.vuepress\public\images\chapter-01'
New-Item -ItemType Directory -Force -Path $assetRoot | Out-Null

$assets = @{
  'figure-01-gpu-devotes-more-transistors.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/gpu-devotes-more-transistors-to-data-processing.png'
  'figure-02-gpu-cpu-system-diagram.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/gpu-cpu-system-diagram.png'
  'figure-03-grid-of-thread-blocks.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/grid-of-thread-blocks.png'
  'figure-04-thread-block-scheduling.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/thread-block-scheduling.png'
  'figure-05-grid-of-clusters.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/grid-of-clusters.png'
  'figure-06-thread-block-scheduling-with-clusters.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/thread-block-scheduling-with-clusters.png'
  'figure-07-active-warp-lanes.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/active-warp-lanes.png'
  'figure-08-tile-simt.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/tile-simt.png'
  'figure-09-tile-data-movement.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/tile-data-movement.png'
  'figure-10-fatbin.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/fatbin.png'
}

foreach ($asset in $assets.GetEnumerator()) {
  $destination = Join-Path $assetRoot $asset.Key
  Invoke-WebRequest -UseBasicParsing -Uri $asset.Value -OutFile $destination -TimeoutSec 60
  Write-Output ("downloaded {0} ({1} bytes)" -f $asset.Key, (Get-Item -LiteralPath $destination).Length)
}
