param(
  [string]$InputRoot = (Join-Path $PSScriptRoot '..\tmp\official-pages'),
  [string]$OutputRoot = (Join-Path $PSScriptRoot '..\tmp\official-markdown')
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

foreach ($source in Get-ChildItem -LiteralPath $InputRoot -Recurse -File -Filter '*.html') {
  $html = Get-Content -Raw -LiteralPath $source.FullName
  $match = [regex]::Match($html, '(?is)<article\b[^>]*class="[^"]*\bbd-article\b[^"]*"[^>]*>(.*?)</article>')
  if (-not $match.Success) {
    throw "Could not find bd-article in $($source.FullName)"
  }

  $relative = $source.FullName.Substring((Resolve-Path $InputRoot).Path.Length).TrimStart('\', '/')
  $output = Join-Path $OutputRoot ([IO.Path]::ChangeExtension($relative, '.md'))
  $parent = Split-Path -Parent $output
  New-Item -ItemType Directory -Force -Path $parent | Out-Null

  $fragment = Join-Path $source.DirectoryName ($source.BaseName + '.article.html')
  [IO.File]::WriteAllText($fragment, "<!doctype html><html><body>$($match.Groups[1].Value)</body></html>", [Text.UTF8Encoding]::new($false))
  & pandoc -f html -t gfm --wrap=none $fragment -o $output
  if ($LASTEXITCODE -ne 0) { throw "pandoc failed for $($source.FullName)" }
  Remove-Item -LiteralPath $fragment -Force
  Write-Output ("extracted {0}" -f $relative)
}
