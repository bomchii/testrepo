# patch-cmake.ps1
# Parchea los CMake files para compilar s2.cpp en Windows con MSVC.
# OpenSSL se instala en un step previo del workflow via Chocolatey.

Write-Host "=== Parcheando CMake files para Windows ==="

$depsDir = "build\_deps"
New-Item -ItemType Directory -Force -Path $depsDir | Out-Null

# ── 1. Descargar headers originales de Crow (source tarball) ─────────────────
# Usamos el source tarball en vez de crow_all.h porque:
# - crow_all.h incluye asio::ssl incondicionalmente
# - Los headers originales tienen #ifdef CROW_ENABLE_SSL
$crowDir  = "$depsDir\crow-include"
$crowFile = "$crowDir\crow.h"   # el tarball pone crow.h directamente en include/
if (-Not (Test-Path $crowFile)) {
    Write-Host "Descargando Crow v1.3.1 source tarball..."
    $crowTar = "$depsDir\crow.tar.gz"
    Invoke-WebRequest `
        -Uri "https://github.com/CrowCpp/Crow/archive/refs/tags/v1.3.1.tar.gz" `
        -OutFile $crowTar -UseBasicParsing
    # Extraer con tar (disponible en Windows 10+ y en todos los runners de GitHub)
    New-Item -ItemType Directory -Force -Path "$depsDir\crow-extracted" | Out-Null
    tar -xzf $crowTar -C "$depsDir\crow-extracted"
    # El tarball extrae como Crow-1.3.1/include/crow/
    $crowSrc = "$depsDir\crow-extracted\Crow-1.3.1\include"
    if (-Not (Test-Path $crowSrc)) {
        # Fallback: buscar include/ en cualquier subdirectorio
        $crowSrc = Get-ChildItem "$depsDir\crow-extracted" -Recurse -Filter "crow.h" |
                   Select-Object -First 1 |
                   ForEach-Object { $_.DirectoryName | Split-Path -Parent }
    }
    New-Item -ItemType Directory -Force -Path $crowDir | Out-Null
    Copy-Item "$crowSrc\*" $crowDir -Recurse -Force
    Remove-Item $crowTar -Force
    Write-Host "OK: Crow headers en $crowDir"
    Write-Host "   crow.h existe: $(Test-Path $crowFile)"
} else {
    Write-Host "OK: Crow headers ya presentes (cache)"
}

# ── 2. Descargar Asio headers ─────────────────────────────────────────────────
$asioDir  = "$depsDir\asio-include"
$asioFile = "$asioDir\asio.hpp"
if (-Not (Test-Path $asioFile)) {
    Write-Host "Descargando Asio 1.30.2..."
    $asioZip = "$depsDir\asio.zip"
    Invoke-WebRequest `
        -Uri "https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-30-2.zip" `
        -OutFile $asioZip -UseBasicParsing
    Expand-Archive -Path $asioZip -DestinationPath "$depsDir\asio-extracted" -Force
    $asioSrc = "$depsDir\asio-extracted\asio-asio-1-30-2\asio\include"
    New-Item -ItemType Directory -Force -Path $asioDir | Out-Null
    Copy-Item "$asioSrc\*" $asioDir -Recurse -Force
    Remove-Item $asioZip -Force
    Write-Host "OK: Asio headers en $asioDir"
} else {
    Write-Host "OK: Asio headers presentes (cache)"
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

# ── 3. Parchear ggml-vulkan/CMakeLists.txt ────────────────────────────────────
$vkPath  = "ggml\src\ggml-vulkan\CMakeLists.txt"
$vkCmake = Get-Content $vkPath -Raw

if ($vkCmake -match "execute_process") {
    $vkCmake = $vkCmake -replace '(?s)function\(test_shader_extension_support.*?endfunction\(\)', @'
function(test_shader_extension_support EXTENSION_NAME TEST_SHADER_FILE RESULT_VARIABLE)
    message(STATUS "${EXTENSION_NAME} disabled (portability build)")
    set(${RESULT_VARIABLE} OFF PARENT_SCOPE)
endfunction()
'@
    [System.IO.File]::WriteAllText(
        (Resolve-Path $vkPath).Path,
        $vkCmake,
        [System.Text.UTF8Encoding]::new($false)
    )
    Write-Host "OK: ggml-vulkan CMakeLists.txt parcheado (coopmat OFF)"
} else {
    Write-Host "OK: ggml-vulkan CMakeLists.txt ya parcheado"
}

# ── 4. Reescribir CMakeLists.txt raiz ────────────────────────────────────────
# main.cpp hace #include <crow.h> — con el include dir apuntando a
# crow-include/crow/, el compilador encuentra crow-include/crow/crow.h. OK.
# CROW_ENABLE_SSL=0 evita todo el codigo SSL en los headers originales de Crow.

# Resolver linkado de OpenSSL ANTES de escribir el CMakeLists.txt
# Usamos .Replace() de .NET (literal, no regex) para convertir backslashes
# -replace en PowerShell usa regex donde \P es escape invalido con "Program Files"
$opensslRoot  = $env:OPENSSL_ROOT_DIR.Replace('\', '/')

# Chocolatey instala OpenSSL con libs en dos posibles ubicaciones:
#   lib/VC/x64/MD/libssl.lib   (OpenSSL 1.1 via choco)
#   lib/libssl.lib              (otras instalaciones)
$sslLibMD     = "$opensslRoot/lib/VC/x64/MD/libssl.lib"
$cryptoLibMD  = "$opensslRoot/lib/VC/x64/MD/libcrypto.lib"
$sslLibFlat   = "$opensslRoot/lib/libssl.lib"
$cryptoLibFlat= "$opensslRoot/lib/libcrypto.lib"

Write-Host "OpenSSL root: $opensslRoot"

if ((Test-Path $sslLibMD) -and (Test-Path $cryptoLibMD)) {
    $sslLib    = $sslLibMD
    $cryptoLib = $cryptoLibMD
    Write-Host "OK: OpenSSL estatico (VC/x64/MD) -> $sslLib"
} elseif ((Test-Path $sslLibFlat) -and (Test-Path $cryptoLibFlat)) {
    $sslLib    = $sslLibFlat
    $cryptoLib = $cryptoLibFlat
    Write-Host "OK: OpenSSL estatico (lib/) -> $sslLib"
} else {
    $sslLib    = $null
    $cryptoLib = $null
    Write-Host "WARN: libssl.lib no encontrado — usando OpenSSL dinamico"
}

if ($sslLib) {
    $opensslLinkBlock = @"
    # OpenSSL estatico (resuelto por patch-cmake.ps1 — choco install openssl)
    target_link_libraries(s2 PRIVATE
        "$sslLib"
        "$cryptoLib")
"@
} else {
    $opensslLinkBlock = @"
    # OpenSSL dinamico (fallback — libssl.lib no encontrado)
    find_package(OpenSSL REQUIRED)
    target_link_libraries(s2 PRIVATE OpenSSL::SSL OpenSSL::Crypto)
"@
}

$newCmake = @"
cmake_minimum_required(VERSION 3.14)
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

if(S2_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(S2_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(S2_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(ggml)

# ---------------------------------------------------------------------------
# Crow: headers originales del source tarball (NO crow_all.h).
# Los headers originales tienen #ifdef CROW_ENABLE_SSL, por lo que
# definir CROW_ENABLE_SSL=0 evita completamente asio::ssl y OpenSSL.
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
# cxxopts
# ---------------------------------------------------------------------------
include(FetchContent)
find_package(cxxopts QUIET)
if(NOT cxxopts_FOUND)
    FetchContent_Declare(cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG        v3.2.1
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(cxxopts)
endif()

# ---------------------------------------------------------------------------
# s2 executable
# ---------------------------------------------------------------------------
set(S2_SOURCES
    src/s2_audio.cpp
    src/s2_tokenizer.cpp
    src/s2_sampler.cpp
    src/s2_model.cpp
    src/s2_codec.cpp
    src/s2_prompt.cpp
    src/s2_generate.cpp
    src/s2_pipeline.cpp
    src/main.cpp
)

# tokenizer_data.cpp generado por el workflow (unsigned char[], ~12MB).
# Se incluye solo si existe — build local sin workflow usa tokenizer.json en disco.
if(EXISTS "`${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer_data.cpp")
    list(APPEND S2_SOURCES src/tokenizer_data.cpp)
    message(STATUS "tokenizer embebido: src/tokenizer_data.cpp incluido")
else()
    message(STATUS "tokenizer: se usara tokenizer.json en disco (build local)")
endif()

add_executable(s2 `${S2_SOURCES})

# Suprimir warnings C4838/C4309/C4365 solo para tokenizer_data.cpp
# (narrowing/truncation en el array de bytes — son inofensivos con unsigned char)
if(MSVC AND EXISTS "`${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer_data.cpp")
    set_source_files_properties(src/tokenizer_data.cpp PROPERTIES
        COMPILE_FLAGS "/wd4838 /wd4309 /wd4365 /wd4267")
endif()

target_include_directories(s2 PRIVATE
    `${CMAKE_CURRENT_SOURCE_DIR}/include
    `${CMAKE_CURRENT_SOURCE_DIR}/third_party
    `${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    `${CMAKE_CURRENT_SOURCE_DIR}/ggml/src
    `${CROW_INCLUDE_DIR}
)

target_link_libraries(s2 PRIVATE
    ggml
    asio_iface
    cxxopts::cxxopts
)

if(S2_VULKAN)
    target_compile_definitions(s2 PRIVATE GGML_USE_VULKAN)
endif()

if(WIN32)
    target_link_libraries(s2 PRIVATE ws2_32 mswsock crypt32)
$opensslLinkBlock
    target_compile_definitions(s2 PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _WIN32_WINNT=0x0A00
        CROW_ENABLE_SSL=0
        ASIO_STANDALONE)
    if(MSVC)
        # /FI fuerza un include al inicio de cada TU — garantiza que
        # CROW_ENABLE_SSL=0 y ASIO_STANDALONE se definen ANTES de cualquier
        # #include en el codigo fuente, incluyendo crow.h
        target_compile_options(s2 PRIVATE
            /W3 /wd4996 /wd4267 /wd4244 /wd4566 /MP /utf-8
            /DCROW_ENABLE_SSL=0
            /DASIO_STANDALONE
            /DNOMINMAX
            /DWIN32_LEAN_AND_MEAN
        )
    endif()
elseif(UNIX AND NOT APPLE)
    target_link_libraries(s2 PRIVATE pthread m)
endif()

install(TARGETS s2 RUNTIME DESTINATION bin)
"@

[System.IO.File]::WriteAllText(
    (Join-Path (Get-Location) "CMakeLists.txt"),
    $newCmake,
    [System.Text.UTF8Encoding]::new($false)
)
Write-Host "OK: CMakeLists.txt raiz reescrito."
Write-Host "=== Parche completado ==="
