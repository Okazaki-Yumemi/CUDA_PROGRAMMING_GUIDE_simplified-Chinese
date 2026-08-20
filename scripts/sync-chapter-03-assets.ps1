$ErrorActionPreference = 'Stop'

$assetRoot = Join-Path $PSScriptRoot '..\docs\.vuepress\public\images\chapter-03'
New-Item -ItemType Directory -Force -Path $assetRoot | Out-Null

$assets = @{
  'pdl.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/pdl.png'
  'warps-in-a-block.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/warps-in-a-block.png'
  'library-context-management.png' = 'https://docs.nvidia.com/cuda/cuda-programming-guide/_images/library-context-management.png'
}

foreach ($asset in $assets.GetEnumerator()) {
  $destination = Join-Path $assetRoot $asset.Key
  Invoke-WebRequest -UseBasicParsing -Uri $asset.Value -OutFile $destination -TimeoutSec 60
  Write-Output ("downloaded {0} ({1} bytes)" -f $asset.Key, (Get-Item -LiteralPath $destination).Length)
}
