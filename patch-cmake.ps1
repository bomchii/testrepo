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
    Invoke-WebRequest `
        -Uri "https://github.com/CrowCpp/Crow/archive/refs/tags/v$crowVersion.tar.gz" `
        -OutFile $crowTar -UseBasicParsing
    # Extraer con tar (disponible en Windows 10+ y en todos los runners de GitHub)
    New-Item -ItemType Directory -Force -Path "$depsDir\crow-extracted" | Out-Null
    tar -xzf $crowTar -C "$depsDir\crow-extracted"
    # El tarball extrae como Crow-1.3.3/include/crow/
    $crowSrc = "$depsDir\crow-extracted\Crow-$crowVersion\include"
    if (-Not (Test-Path $crowSrc)) {
        # Fallback: buscar include/ en cualquier subdirectorio
        $crowSrc = Get-ChildItem "$depsDir\crow-extracted" -Recurse -Filter "crow.h" |
                   Select-Object -First 1 -ExpandProperty DirectoryName
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
    Invoke-WebRequest `
        -Uri "https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-30-2.zip" `
        -OutFile $asioZip -UseBasicParsing
    Expand-Archive -Path $asioZip -DestinationPath "$depsDir\asio-extracted" -Force
    $asioSrc = "$depsDir\asio-extracted\asio-asio-1-30-2\asio\include"
    New-Item -ItemType Directory -Force -Path $asioDir | Out-Null
    Copy-Item "$asioSrc\*" $asioDir -Recurse -Force
    Remove-Item $asioZip -Force
    Set-Content -Path $asioMarker -Value $asioVersion -Encoding ASCII
    # Reemplazar asio/ssl.hpp y asio/ssl/ con stubs vacíos.
    # Crow puede incluir asio/ssl.hpp aunque CROW_ENABLE_SSL=0 esté definido,
    # porque algunos headers de Crow lo incluyen antes de que la macro sea visible.
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
# Crow usa #ifdef CROW_ENABLE_SSL para activar SSL. La forma correcta de
# desactivarlo es NO definir la macro — definirla con valor 0 la activa igualmente.

# Sin CROW_ENABLE_SSL definido, Crow/Asio no generan referencias a OpenSSL.
# No se necesita linkar contra libssl ni libcrypto.
$opensslLinkBlock = ""

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
set(GGML_AVX512         OFF CACHE BOOL "" FORCE)

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
)

if(S2_VULKAN)
    target_compile_definitions(s2 PRIVATE GGML_USE_VULKAN)
elseif(S2_CUDA)
    target_compile_definitions(s2 PRIVATE GGML_USE_CUDA)
elseif(S2_METAL)
    target_compile_definitions(s2 PRIVATE GGML_USE_METAL)
endif()

if(WIN32)
    target_link_libraries(s2 PRIVATE ws2_32 mswsock crypt32)
$opensslLinkBlock
    target_compile_definitions(s2 PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _WIN32_WINNT=0x0A00
        ASIO_STANDALONE)
    if(MSVC)
        # /FI fuerza un include al inicio de cada TU — garantiza que
        # ASIO_STANDALONE se define ANTES de cualquier #include en el codigo fuente.
        # CROW_ENABLE_SSL no se define — con #ifdef, definirlo con valor 0
        # activa el bloque SSL igualmente. La ausencia de la macro lo desactiva.
        target_compile_options(s2 PRIVATE
            /W3 /wd4996 /wd4267 /wd4244 /wd4566 /MP /utf-8 /EHsc
            /DASIO_STANDALONE
            /DNOMINMAX
            /DWIN32_LEAN_AND_MEAN
            /arch:AVX2
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
