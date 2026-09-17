# patch-cmake.ps1
# Parchea los CMake files para compilar s2.cpp en Windows con MSVC.
# Crow se compila sin CROW_ENABLE_SSL; este build no depende de OpenSSL.

param([switch]$SkipVulkanPortabilityPatch)

$ErrorActionPreference = "Stop"

function Invoke-DownloadWithRetry {
    param([Parameter(Mandatory=$true)][string]$Uri,
          [Parameter(Mandatory=$true)][string]$OutFile,
          [int]$Attempts = 3)
    for ($attempt = 1; $attempt -le $Attempts; ++$attempt) {
        try {
            if (Test-Path $OutFile) { Remove-Item $OutFile -Force }
            Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing -ErrorAction Stop
            if (-Not (Test-Path $OutFile) -or (Get-Item $OutFile).Length -le 0) {
                throw "download produced an empty file: $OutFile"
            }
            return
        } catch {
            if ($attempt -eq $Attempts) { throw }
            Write-Warning "Download failed (attempt $attempt/$Attempts): $($_.Exception.Message)"
            Start-Sleep -Seconds (2 * $attempt)
        }
    }
}

function Assert-Sha256 {
    param([Parameter(Mandatory=$true)][string]$Path,
          [Parameter(Mandatory=$true)][string]$Expected)
    $actual = (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch for ${Path}: expected $Expected got $actual"
    }
}

Write-Host "=== Parcheando CMake files para Windows ==="

$depsDir = "build\_deps"
New-Item -ItemType Directory -Force -Path $depsDir | Out-Null

# ── 1. Descargar headers originales de Crow (source tarball) ─────────────────
# Usamos el source tarball en vez de crow_all.h porque:
# - crow_all.h incluye asio::ssl incondicionalmente
# - Los headers originales tienen #ifdef CROW_ENABLE_SSL
$crowVersion = "1.3.4"
$crowCommit = "ae0fef0ee67eec897e401321b99b6dd7cfbdc155"
$crowDir  = "$depsDir\crow-include"
$crowFile = "$crowDir\crow.h"   # el tarball pone crow.h directamente en include/
$crowMarker = "$crowDir\.s2-crow-version"
$crowVersionOk = (Test-Path $crowMarker) -and ((Get-Content $crowMarker -Raw).Trim() -eq $crowCommit)
if (-Not (Test-Path $crowFile) -or -Not $crowVersionOk) {
    # A local build may reuse build/_deps from an older script revision. Do not
    # silently keep stale Crow headers merely because crow.h exists.
    if (Test-Path $crowDir) { Remove-Item $crowDir -Recurse -Force }
    Write-Host "Descargando Crow v$($crowVersion) source tarball..."
    $crowTar = "$depsDir\crow.tar.gz"
    Invoke-DownloadWithRetry `
        -Uri "https://github.com/CrowCpp/Crow/archive/$crowCommit.tar.gz" `
        -OutFile $crowTar
    # Extraer con tar (disponible en Windows 10+ y en los runners hospedados).
    $crowExtracted = "$depsDir\crow-extracted"
    if (Test-Path $crowExtracted) { Remove-Item $crowExtracted -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $crowExtracted | Out-Null
    tar -xzf $crowTar -C $crowExtracted
    if ($LASTEXITCODE -ne 0) { throw "tar failed while extracting Crow (exit $LASTEXITCODE)" }
    # El archive del commit extrae bajo Crow-<sha>/include/.
    $crowSrc = "$depsDir\crow-extracted\Crow-$crowVersion\include"
    if (-Not (Test-Path $crowSrc)) {
        # Fallback: buscar include/ en cualquier subdirectorio
        $crowSrc = Get-ChildItem "$depsDir\crow-extracted" -Recurse -Filter "crow.h" -ErrorAction SilentlyContinue |
                   Select-Object -First 1 -ExpandProperty DirectoryName
    }
    if (-Not $crowSrc -or -Not (Test-Path (Join-Path $crowSrc "crow.h"))) {
        throw "Crow archive layout is invalid: include/crow.h was not found"
    }
    New-Item -ItemType Directory -Force -Path $crowDir | Out-Null
    Copy-Item "$crowSrc\*" $crowDir -Recurse -Force
    Set-Content -Path $crowMarker -Value $crowCommit -Encoding ASCII
    Remove-Item $crowTar -Force
    Write-Host "OK: Crow headers en $crowDir"
    Write-Host "   crow.h existe: $(Test-Path $crowFile)"
} else {
    Write-Host "OK: Crow headers ya presentes (cache)"
}

# ── 2. Descargar Asio headers ─────────────────────────────────────────────────
$asioVersion = "1.30.2"
$asioDir  = "$depsDir\asio-include"
$asioFile = "$asioDir\asio.hpp"
$asioMarker = "$asioDir\.s2-asio-version"
$asioVersionOk = (Test-Path $asioMarker) -and ((Get-Content $asioMarker -Raw).Trim() -eq $asioVersion)
if (-Not (Test-Path $asioFile) -or -Not $asioVersionOk) {
    if (Test-Path $asioDir) { Remove-Item $asioDir -Recurse -Force }
    Write-Host "Descargando Asio $asioVersion..."
    $asioTar = "$depsDir\asio.tar.gz"
    $asioSha256 = "755bd7f85a4b269c67ae0ea254907c078d408cce8e1a352ad2ed664d233780e8"
    Invoke-DownloadWithRetry `
        -Uri "https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-30-2.tar.gz" `
        -OutFile $asioTar
    Assert-Sha256 -Path $asioTar -Expected $asioSha256
    $asioExtracted = "$depsDir\asio-extracted"
    if (Test-Path $asioExtracted) { Remove-Item $asioExtracted -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $asioExtracted | Out-Null
    tar -xzf $asioTar -C $asioExtracted
    if ($LASTEXITCODE -ne 0) { throw "tar failed while extracting Asio (exit $LASTEXITCODE)" }
    $asioSrc = "$asioExtracted\asio-asio-1-30-2\asio\include"
    if (-Not (Test-Path (Join-Path $asioSrc "asio.hpp"))) {
        throw "Asio archive layout is invalid: asio/include/asio.hpp was not found"
    }
    New-Item -ItemType Directory -Force -Path $asioDir | Out-Null
    Copy-Item "$asioSrc\*" $asioDir -Recurse -Force
    Remove-Item $asioTar -Force
    Set-Content -Path $asioMarker -Value $asioVersion -Encoding ASCII
    # Reemplazar asio/ssl.hpp y asio/ssl/ con stubs vacíos.
    # Crow puede alcanzar asio/ssl.hpp por el orden de inclusion de algunos headers;
    # los stubs garantizan un build HTTP/WebSocket sin OpenSSL.
    # Un stub vacío con include guard evita el error C1083 sin romper nada.
    $sslStubDir = "$asioDir\asio\ssl"
    New-Item -ItemType Directory -Force -Path $sslStubDir | Out-Null
    # asio/ssl.hpp stub
    @"
#pragma once
// Stub: SSL deshabilitado (sin OpenSSL — CROW_ENABLE_SSL no definido)
"@ | Set-Content -Encoding UTF8 "$asioDir\asio\ssl.hpp"
    # Stubs para los headers individuales que ssl.hpp incluye
    foreach ($stub in @("context.hpp","stream.hpp","error.hpp","rfc2818_verification.hpp","verify_mode.hpp")) {
        @"
#pragma once
// Stub: SSL deshabilitado
"@ | Set-Content -Encoding UTF8 "$sslStubDir\$stub"
    }
    Write-Host "OK: asio/ssl/ reemplazado con stubs vacios"
    Write-Host "OK: Asio headers en $asioDir"
} else {
    Write-Host "OK: Asio headers presentes (cache)"
    # Garantizar stubs incluso en cache (por si el cache tiene los originales)
    $sslStubDir = "$asioDir\asio\ssl"
    New-Item -ItemType Directory -Force -Path $sslStubDir | Out-Null
    @"
#pragma once
// Stub: SSL deshabilitado (sin OpenSSL — CROW_ENABLE_SSL no definido)
"@ | Set-Content -Encoding UTF8 "$asioDir\asio\ssl.hpp"
    foreach ($stub in @("context.hpp","stream.hpp","error.hpp","rfc2818_verification.hpp","verify_mode.hpp")) {
        @"
#pragma once
// Stub: SSL deshabilitado
"@ | Set-Content -Encoding UTF8 "$sslStubDir\$stub"
    }
    Write-Host "OK: asio/ssl/ stubs garantizados en cache"
}

# Rutas con / para CMake (\a, \t etc. son escapes invalidos en CMake)
$crowAbs = (Resolve-Path $crowDir).Path.Replace('\', '/')
$asioAbs = (Resolve-Path $asioDir).Path.Replace('\', '/')
Write-Host "Crow include dir : $crowAbs"
Write-Host "Asio include dir : $asioAbs"

# Verificar que crow.h existe en la ruta correcta
$crowHeader = "$crowAbs/crow.h"
if (-Not (Test-Path $crowHeader.Replace('/', '\'))) {
    Write-Error "ERROR: crow.h no encontrado en $crowAbs"
    Write-Host "Contenido de $crowAbs :"
    Get-ChildItem $crowAbs.Replace('/', '\') | Select-Object Name | Format-Table
    exit 1
}
Write-Host "OK: crow.h verificado"

# ── 3. Optional ggml-vulkan portability patch ─────────────────────────────────
# CUDA/CPU builds do not need to mutate the Vulkan CMake. The root CMakeLists.txt
# is authoritative; this script only prepares pinned Crow/Asio headers and, for
# the Vulkan build, disables shader capability probes that are not portable on CI.
if (-Not $SkipVulkanPortabilityPatch) {
    $vkPath  = "ggml\src\ggml-vulkan\CMakeLists.txt"
    $vkCmake = Get-Content $vkPath -Raw
    $vkPattern = '(?s)function\(test_shader_extension_support.*?endfunction\(\)'
    $vkMatches = [regex]::Matches($vkCmake, $vkPattern)
    if ($vkMatches.Count -ne 1) {
        throw "Expected exactly one test_shader_extension_support() function in ${vkPath}; found $($vkMatches.Count)"
    }

    $vkFunction = $vkMatches[0].Value
    if ($vkFunction -match 'execute_process') {
        $vkReplacement = @'
function(test_shader_extension_support EXTENSION_NAME TEST_SHADER_FILE RESULT_VARIABLE)
    message(STATUS "${EXTENSION_NAME} disabled (portability build)")
    set(${RESULT_VARIABLE} OFF PARENT_SCOPE)
endfunction()
'@
        $vkCmake = [regex]::Replace($vkCmake, $vkPattern, $vkReplacement)
        $verifyMatches = [regex]::Matches($vkCmake, $vkPattern)
        if ($verifyMatches.Count -ne 1 -or $verifyMatches[0].Value -match 'execute_process') {
            throw "Vulkan portability patch postcondition failed for ${vkPath}"
        }
        [System.IO.File]::WriteAllText(
            (Resolve-Path $vkPath).Path,
            $vkCmake,
            [System.Text.UTF8Encoding]::new($false)
        )
        Write-Host "OK: ggml-vulkan CMakeLists.txt parcheado (coopmat OFF)"
    } elseif ($vkFunction -match 'disabled \(portability build\)') {
        Write-Host "OK: ggml-vulkan CMakeLists.txt ya parcheado"
    } else {
        throw "Unknown test_shader_extension_support() shape in ${vkPath}; refusing a blind patch"
    }
} else {
    Write-Host "Vulkan portability patch skipped for this backend."
}

Write-Host "=== Preparacion de dependencias Windows completada ==="
