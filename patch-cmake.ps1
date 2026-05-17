# patch-cmake.ps1
# Usa crow_all.h (single header oficial de Crow) + Asio standalone.
# Evita por completo el CMakeLists.txt de Crow y sus dependencias de SSL.

Write-Host "=== Parcheando CMake files para Windows ==="

$depsDir = "build\_deps"
New-Item -ItemType Directory -Force -Path $depsDir | Out-Null

# ── 1. Descargar crow_all.h (single header, sin SSL, sin CMake propio) ────────
$crowDir  = "$depsDir\crow-include"
$crowFile = "$crowDir\crow.h"
if (-Not (Test-Path $crowFile)) {
    Write-Host "Descargando crow_all.h v1.2.0..."
    New-Item -ItemType Directory -Force -Path $crowDir | Out-Null
    Invoke-WebRequest `
        -Uri "https://github.com/CrowCpp/Crow/releases/download/v1.2.0/crow_all.h" `
        -OutFile $crowFile -UseBasicParsing
    Write-Host "OK: crow.h en $crowFile ($([math]::Round((Get-Item $crowFile).Length/1KB,0)) KB)"
} else {
    Write-Host "OK: crow.h ya presente (cache)"
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
    # Eliminar asio/ssl/ fisicamente y reemplazar con stubs vacios.
    # crow_all.h incluye asio/ssl.hpp incondicionalmente aunque no se use HTTPS.
    # Los stubs satisfacen el #include sin requerir OpenSSL.
    $sslPath = "$asioDir\asio\ssl"
    Remove-Item $sslPath -Recurse -Force -ErrorAction SilentlyContinue

    # Crear stubs minimos para que los #include de crow no fallen
    New-Item -ItemType Directory -Force -Path $sslPath | Out-Null
    New-Item -ItemType Directory -Force -Path "$sslPath\detail" | Out-Null

    # asio/ssl.hpp — incluido directamente por crow_all.h
    @'
#pragma once
// SSL stub: proyecto no usa HTTPS, OpenSSL no requerido
'@ | Set-Content "$asioDir\asio\ssl.hpp" -Encoding UTF8

    # Subdirs que asio/ssl.hpp normalmente incluye
    foreach ($f in @("context.hpp","context_base.hpp","error.hpp","rfc2818_verification.hpp",
                     "stream.hpp","stream_base.hpp","verify_context.hpp","verify_mode.hpp")) {
        "@`#pragma once`n// SSL stub" | Set-Content "$sslPath\$f" -Encoding UTF8
    }
    # detail/ stubs
    foreach ($f in @("openssl_types.hpp","openssl_init.hpp","engine.hpp","io.hpp",
                     "buffered_handshake_op.hpp","connect_op.hpp","handshake_op.hpp",
                     "read_op.hpp","shutdown_op.hpp","stream_core.hpp","write_op.hpp")) {
        "@`#pragma once`n// SSL stub" | Set-Content "$sslPath\detail\$f" -Encoding UTF8
    }
    Write-Host "OK: asio/ssl/ reemplazado con stubs (sin OpenSSL)"
    Write-Host "OK: Asio headers en $asioDir"
} else {
    # En cache restaurado, verificar y crear stubs si faltan
    $sslStub = "$asioDir\asio\ssl.hpp"
    if (-Not (Test-Path $sslStub)) {
        $sslPath = "$asioDir\asio\ssl"
        Remove-Item $sslPath -Recurse -Force -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Force -Path $sslPath | Out-Null
        New-Item -ItemType Directory -Force -Path "$sslPath\detail" | Out-Null
        "@`#pragma once`n// SSL stub" | Set-Content $sslStub -Encoding UTF8
        foreach ($f in @("context.hpp","context_base.hpp","error.hpp","rfc2818_verification.hpp",
                         "stream.hpp","stream_base.hpp","verify_context.hpp","verify_mode.hpp")) {
            "@`#pragma once`n// SSL stub" | Set-Content "$sslPath\$f" -Encoding UTF8
        }
        foreach ($f in @("openssl_types.hpp","openssl_init.hpp","engine.hpp","io.hpp",
                         "buffered_handshake_op.hpp","connect_op.hpp","handshake_op.hpp",
                         "read_op.hpp","shutdown_op.hpp","stream_core.hpp","write_op.hpp")) {
            "@`#pragma once`n// SSL stub" | Set-Content "$sslPath\detail\$f" -Encoding UTF8
        }
        Write-Host "OK: stubs SSL creados en cache"
    }
    Write-Host "OK: Asio headers ya presentes (cache)"
}

# Rutas con / para CMake (\ causa errores de escape)
$crowAbs = (Resolve-Path $crowDir).Path.Replace('\', '/')
$asioAbs = (Resolve-Path $asioDir).Path.Replace('\', '/')
Write-Host "Crow include dir : $crowAbs"
Write-Host "Asio include dir : $asioAbs"

# ── 3. Parchear ggml-vulkan/CMakeLists.txt ────────────────────────────────────
# Desactiva deteccion de coopmat/coopmat2 para evitar race condition entre
# vulkan-shaders-gen y ggml-vulkan.cpp compilando en paralelo con MSBuild.

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
$newCmake = @"
cmake_minimum_required(VERSION 3.14)
project(s2cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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
# Crow: usamos crow_all.h (single header) descargado por patch-cmake.ps1.
# NO se usa FetchContent ni el CMakeLists.txt de Crow para evitar que
# configure SSL y busque OpenSSL.
# ---------------------------------------------------------------------------
set(CROW_INCLUDE_DIR "$crowAbs")

# ---------------------------------------------------------------------------
# Asio standalone (requerido por Crow en Windows)
# ---------------------------------------------------------------------------
set(ASIO_INCLUDE_DIR "$asioAbs")

add_library(asio_iface INTERFACE)
target_include_directories(asio_iface INTERFACE `${ASIO_INCLUDE_DIR})
target_compile_definitions(asio_iface INTERFACE ASIO_STANDALONE ASIO_NO_SSL)

# ---------------------------------------------------------------------------
# cxxopts (unico dep que sigue usando FetchContent, sin problemas de SSL)
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

add_executable(s2 `${S2_SOURCES})

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
    target_link_libraries(s2 PRIVATE ws2_32 mswsock)
    target_compile_definitions(s2 PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _WIN32_WINNT=0x0A00
        CROW_ENABLE_SSL=0
        ASIO_STANDALONE
        ASIO_NO_SSL)
    if(MSVC)
        target_compile_options(s2 PRIVATE /W3 /wd4996 /wd4267 /wd4244 /MP)
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
