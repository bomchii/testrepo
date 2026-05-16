# patch-cmake.ps1
# Reescribe los archivos CMake necesarios para compilar en Windows con MSVC.
# Usa Out-File en vez de .Replace() para evitar problemas con CRLF/LF y
# caracteres especiales de PowerShell.

Write-Host "=== Parcheando CMake files para Windows ==="

# ── 1. ggml/src/ggml-vulkan/CMakeLists.txt ───────────────────────────────────
# Reemplaza la funcion test_shader_extension_support para que siempre devuelva
# OFF, evitando el race condition entre vulkan-shaders-gen y ggml-vulkan.cpp

$vkCmakePath = "ggml\src\ggml-vulkan\CMakeLists.txt"
$vkCmake = Get-Content $vkCmakePath -Raw

# Verificar que el parche es necesario
if ($vkCmake -match "execute_process") {
    # Usar -replace con regex para ser robusto ante diferencias CRLF/LF
    $vkCmake = $vkCmake -replace '(?s)function\(test_shader_extension_support.*?endfunction\(\)', @'
function(test_shader_extension_support EXTENSION_NAME TEST_SHADER_FILE RESULT_VARIABLE)
    message(STATUS "${EXTENSION_NAME} disabled (portability build - avoid race condition)")
    set(${RESULT_VARIABLE} OFF PARENT_SCOPE)
endfunction()
'@
    [System.IO.File]::WriteAllText((Resolve-Path $vkCmakePath), $vkCmake, [System.Text.UTF8Encoding]::new($false))
    Write-Host "OK: ggml-vulkan CMakeLists.txt parcheado (coopmat desactivado)"
} else {
    Write-Host "AVISO: ggml-vulkan CMakeLists.txt ya parcheado o diferente de lo esperado"
}

# ── 2. CMakeLists.txt raiz — reescritura completa ────────────────────────────
# En vez de hacer .Replace() con strings que pueden fallar por CRLF/comillas,
# escribimos el archivo completo con el contenido correcto para Windows.

$newCmake = @'
cmake_minimum_required(VERSION 3.14)
project(s2cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

option(S2_VULKAN  "Build with Vulkan backend"  OFF)
option(S2_CUDA    "Build with CUDA backend"    OFF)
option(S2_METAL   "Build with Metal backend"   OFF)

# ---------------------------------------------------------------------------
# GGML
# ---------------------------------------------------------------------------

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
# Asio (debe ir ANTES de Crow porque Crow lo busca internamente)
# ---------------------------------------------------------------------------

include(FetchContent)

FetchContent_Declare(asio
    GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
    GIT_TAG        asio-1-30-2
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(asio)

# Exponer ASIO_INCLUDE_DIR como cache para que Findasio.cmake de Crow lo encuentre
set(ASIO_INCLUDE_DIR "${asio_SOURCE_DIR}/asio/include" CACHE PATH "" FORCE)

# Target de interfaz para propagar Asio a s2
add_library(asio_iface INTERFACE)
target_include_directories(asio_iface INTERFACE "${asio_SOURCE_DIR}/asio/include")
target_compile_definitions(asio_iface INTERFACE ASIO_STANDALONE)

# ---------------------------------------------------------------------------
# Crow (servidor HTTP, header-only)
# ---------------------------------------------------------------------------

find_package(Crow QUIET)
if(NOT Crow_FOUND)
    FetchContent_Declare(crow
        GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
        GIT_TAG        v1.2.0
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(crow)
endif()
set(CROW_INCLUDE_DIRS "${crow_SOURCE_DIR}/include")
set(CROW_LIBRARIES "")

# ---------------------------------------------------------------------------
# cxxopts
# ---------------------------------------------------------------------------

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

add_executable(s2 ${S2_SOURCES})

target_include_directories(s2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src
    ${CROW_INCLUDE_DIRS}
)

target_link_libraries(s2 PRIVATE
    ggml
    ${CROW_LIBRARIES}
    cxxopts::cxxopts
)

if(S2_VULKAN)
    target_compile_definitions(s2 PRIVATE GGML_USE_VULKAN)
endif()

# Asio standalone para Crow en Windows
if(TARGET asio_iface)
    target_link_libraries(s2 PRIVATE asio_iface)
    target_compile_definitions(s2 PRIVATE ASIO_STANDALONE CROW_ENABLE_SSL=0)
endif()

# Platform-specific
if(WIN32)
    target_link_libraries(s2 PRIVATE ws2_32 mswsock)
    target_compile_definitions(s2 PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _WIN32_WINNT=0x0A00)
    if(MSVC)
        # /W3 sin warnings de funciones deprecadas, /MP compilacion paralela
        target_compile_options(s2 PRIVATE /W3 /wd4996 /wd4267 /wd4244 /MP)
    endif()
elseif(UNIX AND NOT APPLE)
    target_link_libraries(s2 PRIVATE pthread m)
endif()

# Install
install(TARGETS s2 RUNTIME DESTINATION bin)
'@

[System.IO.File]::WriteAllText(
    (Join-Path (Get-Location) "CMakeLists.txt"),
    $newCmake,
    [System.Text.UTF8Encoding]::new($false)
)
Write-Host "OK: CMakeLists.txt raiz reescrito para Windows."
Write-Host "=== Parche completado ==="
