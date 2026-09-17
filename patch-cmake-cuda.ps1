# patch-cmake-cuda.ps1
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
& (Join-Path $PSScriptRoot "patch-cmake.ps1") -SkipVulkanPortabilityPatch
