$ErrorActionPreference = 'Stop'

$assetRoot = Join-Path $PSScriptRoot '..\docs\.vuepress\public\images\chapter-04'
New-Item -ItemType Directory -Force -Path $assetRoot | Out-Null

$assets = @(
  'adding-new-alloc-nodes.png'
  'child-graph.png'
  'cluster_launch_control.png'
  'conditional-if-node.png'
  'conditional-switch-node.png'
  'conditional-while-node.png'
  'create-a-graph.png'
  'cuda_graph_reduction.png'
  'device-graph-stream-environment.png'
  'egm-c2c-intro.png'
  'fire-and-forget-environments.png'
  'fire-and-forget-nested-environments.png'
  'fire-and-forget-simple.png'
  'gpu-activity.png'
  'green_contexts_motivation.png'
  'green_contexts_ncu_mask.png'
  'green_contexts_nsys_example_no_GCs_with_prio.png'
  'green_contexts_nsys_example_w_GCs.png'
  'green_contexts_resource_split_by_count.png'
  'green_contexts_resource_split.png'
  'kernel-nodes.png'
  'new-alloc-node.png'
  'parent-child-launch-nesting.png'
  'preamble-overlap.png'
  'secondary-kernel-preamble.png'
  'sequentially-launched-graphs.png'
  'sibling-launch-simple.png'
  'swizzle-example1.png'
  'swizzle-example2.png'
  'swizzle-pattern.png'
  'tail-launch-ordering-complex.png'
  'tail-launch-ordering-simple.png'
  'tail-launch-simple.png'
  'unicast-memory-sharing.png'
  'vmm-overview-diagram.png'
)

foreach ($name in $assets) {
  $destination = Join-Path $assetRoot $name
  $url = "https://docs.nvidia.com/cuda/cuda-programming-guide/_images/$name"
  Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $destination -TimeoutSec 60
  Write-Output ("downloaded {0} ({1} bytes)" -f $name, (Get-Item -LiteralPath $destination).Length)
}
