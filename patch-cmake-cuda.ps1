# patch-cmake-cuda.ps1
# Variante CUDA + AVX2 de patch-cmake.ps1 (compatible con CUDA 13.x / VS 2026).
# Genera s2-cuda.exe compilado con ggml-cuda. Las arquitecturas las selecciona GGML.
# Sin Vulkan SDK — solo CUDA toolkit (instalado por el workflow).

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

Write-Host "=== Parcheando CMake files para Windows ==="

$depsDir = "build\_deps"
New-Item -ItemType Directory -Force -Path $depsDir | Out-Null

# ── 1. Descargar headers originales de Crow (source tarball) ─────────────────
# Usamos el source tarball en vez de crow_all.h porque:
# - crow_all.h incluye asio::ssl incondicionalmente
# - Los headers originales tienen #ifdef CROW_ENABLE_SSL
$crowVersion = "1.3.3"
$crowDir  = "$depsDir\crow-include"
$crowFile = "$crowDir\crow.h"   # el tarball pone crow.h directamente en include/
$crowMarker = "$crowDir\.s2-crow-version"
$crowVersionOk = (Test-Path $crowMarker) -and ((Get-Content $crowMarker -Raw).Trim() -eq $crowVersion)
if (-Not (Test-Path $crowFile) -or -Not $crowVersionOk) {
    # A local build may reuse build/_deps from an older script revision. Do not
    # silently keep stale Crow headers merely because crow.h exists.
    if (Test-Path $crowDir) { Remove-Item $crowDir -Recurse -Force }
    Write-Host "Descargando Crow v$($crowVersion) source tarball..."
    $crowTar = "$depsDir\crow.tar.gz"
    Invoke-DownloadWithRetry `
        -Uri "https://github.com/CrowCpp/Crow/archive/refs/tags/v$crowVersion.tar.gz" `
        -OutFile $crowTar
    # Extraer con tar (disponible en Windows 10+ y en los runners hospedados).
    $crowExtracted = "$depsDir\crow-extracted"
    if (Test-Path $crowExtracted) { Remove-Item $crowExtracted -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $crowExtracted | Out-Null
    tar -xzf $crowTar -C $crowExtracted
    if ($LASTEXITCODE -ne 0) { throw "tar failed while extracting Crow (exit $LASTEXITCODE)" }
    # El tarball extrae como Crow-1.3.3/include/crow/
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
    Set-Content -Path $crowMarker -Value $crowVersion -Encoding ASCII
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
    $asioZip = "$depsDir\asio.zip"
    Invoke-DownloadWithRetry `
        -Uri "https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-30-2.zip" `
        -OutFile $asioZip
    $asioExtracted = "$depsDir\asio-extracted"
    if (Test-Path $asioExtracted) { Remove-Item $asioExtracted -Recurse -Force }
    Expand-Archive -Path $asioZip -DestinationPath $asioExtracted -Force
    $asioSrc = "$asioExtracted\asio-asio-1-30-2\asio\include"
    if (-Not (Test-Path (Join-Path $asioSrc "asio.hpp"))) {
        throw "Asio archive layout is invalid: asio/include/asio.hpp was not found"
    }
    New-Item -ItemType Directory -Force -Path $asioDir | Out-Null
    Copy-Item "$asioSrc\*" $asioDir -Recurse -Force
    Remove-Item $asioZip -Force
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

# ── 3. ggml-vulkan CMakeLists.txt -- NO parchear en build CUDA ───────────────
# GGML_VULKAN=OFF para este build -- el patch de coopmat solo es necesario
# cuando se compila con Vulkan activo (patch-cmake.ps1). Omitir aqui evita
# fallos si la estructura del subdirectorio ggml cambia en el futuro.
Write-Host "OK: ggml-vulkan patch omitido (build CUDA, GGML_VULKAN=OFF)"

# ── 4. Reescribir CMakeLists.txt raiz ────────────────────────────────────────
# main.cpp hace #include <crow.h> — con el include dir apuntando a
# crow-include/crow/, el compilador encuentra crow-include/crow/crow.h. OK.
# Crow usa #ifdef CROW_ENABLE_SSL para activar SSL. La forma correcta de
# desactivarlo es NO definir la macro — definirla con valor 0 la activa igualmente.

# Sin CROW_ENABLE_SSL definido, Crow/Asio no generan referencias a OpenSSL.
# No se necesita linkar contra libssl ni libcrypto.
$opensslLinkBlock = ""

$newCmake = @"
cmake_minimum_required(VERSION 3.15)
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "MSVC runtime library" FORCE)
project(s2cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(S2_VULKAN  "Build with Vulkan backend"  OFF)
option(S2_CUDA    "Build with CUDA backend"    OFF)
option(S2_METAL   "Build with Metal backend"   OFF)

set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_AVX512         OFF CACHE BOOL "" FORCE)
set(GGML_AVX2           ON  CACHE BOOL "" FORCE)

if((S2_VULKAN AND S2_CUDA) OR (S2_VULKAN AND S2_METAL) OR (S2_CUDA AND S2_METAL))
    message(FATAL_ERROR "Choose only one GPU backend per executable")
endif()

set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_CUDA   OFF CACHE BOOL "" FORCE)
set(GGML_METAL  OFF CACHE BOOL "" FORCE)
if(S2_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
elseif(S2_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
elseif(S2_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(ggml)

# ---------------------------------------------------------------------------
# Crow: headers originales del source tarball (NO crow_all.h).
# Los headers originales tienen #ifdef CROW_ENABLE_SSL, por lo que
# NO definir CROW_ENABLE_SSL evita completamente asio::ssl y OpenSSL.
# include_directories apunta a crow-include/ para que #include <crow/crow.h>
# funcione, y tambien a crow-include/crow/ para que #include <crow.h> funcione.
# ---------------------------------------------------------------------------
set(CROW_INCLUDE_DIR "$crowAbs")

# ---------------------------------------------------------------------------
# Asio standalone
# ---------------------------------------------------------------------------
set(ASIO_INCLUDE_DIR "$asioAbs")

add_library(asio_iface INTERFACE)
target_include_directories(asio_iface INTERFACE `${ASIO_INCLUDE_DIR})
target_compile_definitions(asio_iface INTERFACE ASIO_STANDALONE)

# ---------------------------------------------------------------------------
# s2 executable
# ---------------------------------------------------------------------------
set(S2_SOURCES
    src/s2_audio.cpp
    src/s2_json.cpp
    src/s2_text.cpp
    src/s2_tokenizer.cpp
    src/s2_sampler.cpp
    src/s2_model.cpp
    src/s2_codec.cpp
    src/s2_prompt.cpp
    src/s2_generate.cpp
    src/s2_pipeline.cpp
    src/s2_voice.cpp
    src/main.cpp
)

# tokenizer_data.{h,cpp} son un par generado por CI. main.cpp activa el
# tokenizer embebido al ver el header, por lo que aceptar solo uno de los dos
# produciria un link roto o una configuracion ambigua.
set(S2_TOKENIZER_HEADER "`${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer_data.h")
set(S2_TOKENIZER_SOURCE "`${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer_data.cpp")
if(EXISTS "`${S2_TOKENIZER_HEADER}" AND EXISTS "`${S2_TOKENIZER_SOURCE}")
    list(APPEND S2_SOURCES src/tokenizer_data.cpp)
    message(STATUS "tokenizer embebido: src/tokenizer_data.cpp incluido")
elseif(EXISTS "`${S2_TOKENIZER_HEADER}" OR EXISTS "`${S2_TOKENIZER_SOURCE}")
    message(FATAL_ERROR "Incomplete embedded tokenizer pair: both src/tokenizer_data.h and src/tokenizer_data.cpp are required")
else()
    message(STATUS "tokenizer: se usara tokenizer.json en disco (build local)")
endif()

add_executable(s2-cuda `${S2_SOURCES})
set_target_properties(s2-cuda PROPERTIES OUTPUT_NAME "s2-cuda-core")

# Native container/launcher: deliberately links no GGML or CUDA libraries.
add_executable(s2-cuda-launcher tools/cuda_launcher.cpp)
target_compile_features(s2-cuda-launcher PRIVATE cxx_std_17)
target_link_libraries(s2-cuda-launcher PRIVATE bcrypt shell32)

# Suprimir warnings C4838/C4309/C4365 solo para tokenizer_data.cpp
# (narrowing/truncation en el array de bytes — son inofensivos con unsigned char)
if(MSVC AND EXISTS "`${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer_data.cpp")
    set_source_files_properties(src/tokenizer_data.cpp PROPERTIES  # CUDA build
        COMPILE_FLAGS "/wd4838 /wd4309 /wd4365 /wd4267")
endif()

target_include_directories(s2-cuda PRIVATE
    `${CMAKE_CURRENT_SOURCE_DIR}/include
    `${CMAKE_CURRENT_SOURCE_DIR}/third_party
    `${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    `${CMAKE_CURRENT_SOURCE_DIR}/ggml/src
    `${CROW_INCLUDE_DIR}
)

target_link_libraries(s2-cuda PRIVATE
    ggml
    asio_iface
)

if(S2_CUDA)
    target_compile_definitions(s2-cuda PRIVATE GGML_USE_CUDA)
endif()

if(WIN32)
    target_link_libraries(s2-cuda PRIVATE ws2_32 mswsock crypt32)
$opensslLinkBlock
    target_compile_definitions(s2-cuda PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _WIN32_WINNT=0x0A00
        ASIO_STANDALONE)
    if(MSVC)
        # /FI fuerza un include al inicio de cada TU — garantiza que
        # ASIO_STANDALONE se define ANTES de cualquier #include en el codigo fuente.
        # CROW_ENABLE_SSL no se define — con #ifdef, definirlo con valor 0
        # activa el bloque SSL igualmente. La ausencia de la macro lo desactiva.
        target_compile_options(s2-cuda PRIVATE
            /W3 /wd4996 /wd4267 /wd4244 /wd4566 /MP /utf-8 /EHsc
            /DASIO_STANDALONE
            /DNOMINMAX
            /DWIN32_LEAN_AND_MEAN
            /arch:AVX2
        )
    endif()
elseif(UNIX AND NOT APPLE)
    target_link_libraries(s2-cuda PRIVATE pthread m)
endif()

install(TARGETS s2-cuda RUNTIME DESTINATION bin)
"@

[System.IO.File]::WriteAllText(
    (Join-Path (Get-Location) "CMakeLists.txt"),
    $newCmake,
    [System.Text.UTF8Encoding]::new($false)
)
Write-Host "OK: CMakeLists.txt raiz reescrito."
Write-Host "=== Parche completado ==="
