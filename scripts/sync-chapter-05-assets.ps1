param(
  [string]$OutputRoot = (Join-Path $PSScriptRoot '..\docs\.vuepress\public\images\chapter-05')
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$assets = @(
  'floating-point-192.drawio.png',
  'floating-point-encoding.drawio.png',
  'floating-point-ieee.drawio.png',
  'floating-point-special-values.drawio.png',
  'floating-point-subnormal.drawio.png',
  'floating-point.drawio.png'
)

foreach ($asset in $assets) {
  $destination = Join-Path $OutputRoot $asset
  $uri = "https://docs.nvidia.com/cuda/cuda-programming-guide/_images/$asset"
  Invoke-WebRequest -UseBasicParsing -Uri $uri -OutFile $destination -TimeoutSec 90
  Write-Output ("downloaded {0} ({1} bytes)" -f $asset, (Get-Item -LiteralPath $destination).Length)
}
