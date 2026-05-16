# patch-cmake.ps1
# Estrategia: descargar Crow y Asio como ZIPs de headers puros.
# NO se usa FetchContent para Crow — su CMakeLists.txt intenta configurar SSL
# y buscar OpenSSL aunque se le diga CROW_ENABLE_SSL=OFF.
# Bajando solo los headers evitamos todo ese problema.

Write-Host "=== Parcheando CMake files para Windows ==="

# ── 0. Directorios de trabajo ─────────────────────────────────────────────────
$depsDir = "build\_deps"
New-Item -ItemType Directory -Force -Path $depsDir | Out-Null

# ── 1. Descargar Crow headers ─────────────────────────────────────────────────
$crowDir = "$depsDir\crow-include"
if (-Not (Test-Path "$crowDir\crow\crow.h")) {
    Write-Host "Descargando Crow v1.2.0 headers..."
    $crowZip = "$depsDir\crow.zip"
    Invoke-WebRequest -Uri "https://github.com/CrowCpp/Crow/releases/download/v1.2.0/crow-v1.2.0.zip" `
                      -OutFile $crowZip -UseBasicParsing
    Expand-Archive -Path $crowZip -DestinationPath "$depsDir\crow-extracted" -Force
    # El ZIP de release de Crow tiene los headers en include/
    $crowIncludeSrc = Get-ChildItem "$depsDir\crow-extracted" -Recurse -Filter "crow.h" |
                      Select-Object -First 1 -ExpandProperty DirectoryName
    if (-Not $crowIncludeSrc) {
        # Fallback: buscar en cualquier subdirectorio
        $crowIncludeSrc = "$depsDir\crow-extracted"
    }
    New-Item -ItemType Directory -Force -Path $crowDir | Out-Null
    Copy-Item "$crowIncludeSrc\*" $crowDir -Recurse -Force
    Remove-Item $crowZip -Force
    Write-Host "OK: Crow headers en $crowDir"
} else {
    Write-Host "OK: Crow headers ya presentes (cache)"
}

# ── 2. Descargar Asio headers ─────────────────────────────────────────────────
$asioDir = "$depsDir\asio-include"
if (-Not (Test-Path "$asioDir\asio.hpp")) {
    Write-Host "Descargando Asio 1.30.2 headers..."
    $asioZip = "$depsDir\asio.zip"
    Invoke-WebRequest -Uri "https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-30-2.zip" `
                      -OutFile $asioZip -UseBasicParsing
    Expand-Archive -Path $asioZip -DestinationPath "$depsDir\asio-extracted" -Force
    # El ZIP de Asio tiene la estructura: asio-asio-1-30-2/asio/include/
    $asioSrc = "$depsDir\asio-extracted\asio-asio-1-30-2\asio\include"
    New-Item -ItemType Directory -Force -Path $asioDir | Out-Null
    Copy-Item "$asioSrc\*" $asioDir -Recurse -Force
    Remove-Item $asioZip -Force
    Write-Host "OK: Asio headers en $asioDir"
} else {
    Write-Host "OK: Asio headers ya presentes (cache)"
}

# Convertir a rutas absolutas para CMakeLists.txt
$crowAbs = (Resolve-Path $crowDir).Path
$asioAbs = (Resolve-Path $asioDir).Path

Write-Host "Crow: $crowAbs"
Write-Host "Asio: $asioAbs"

# ── 3. Parchear ggml-vulkan/CMakeLists.txt ────────────────────────────────────
# Desactiva la deteccion de coopmat/coopmat2 para evitar el race condition
# entre vulkan-shaders-gen y ggml-vulkan.cpp compilando en paralelo.

$vkCmakePath = "ggml\src\ggml-vulkan\CMakeLists.txt"
$vkCmake = Get-Content $vkCmakePath -Raw

if ($vkCmake -match "execute_process") {
    $vkCmake = $vkCmake -replace '(?s)function\(test_shader_extension_support.*?endfunction\(\)', @'
function(test_shader_extension_support EXTENSION_NAME TEST_SHADER_FILE RESULT_VARIABLE)
    message(STATUS "${EXTENSION_NAME} disabled (portability build)")
    set(${RESULT_VARIABLE} OFF PARENT_SCOPE)
endfunction()
'@
    [System.IO.File]::WriteAllText(
        (Resolve-Path $vkCmakePath).Path,
        $vkCmake,
        [System.Text.UTF8Encoding]::new($false)
    )
    Write-Host "OK: ggml-vulkan CMakeLists.txt parcheado"
} else {
    Write-Host "OK: ggml-vulkan CMakeLists.txt ya parcheado"
}

# ── 4. Reescribir CMakeLists.txt raiz ────────────────────────────────────────
# Usa las rutas absolutas de Crow y Asio descargados como headers puros.
# NO se usa FetchContent para Crow — evita por completo la configuracion SSL.

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
# Crow y Asio como headers puros (sin FetchContent, sin CMake propio)
# Las rutas son absolutas, inyectadas por patch-cmake.ps1
# ---------------------------------------------------------------------------

set(CROW_INCLUDE_DIR "$crowAbs")
set(ASIO_INCLUDE_DIR "$asioAbs")

add_library(asio_iface INTERFACE)
target_include_directories(asio_iface INTERFACE "`${ASIO_INCLUDE_DIR}")
target_compile_definitions(asio_iface INTERFACE
    ASIO_STANDALONE
    ASIO_NO_SSL)

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
