cmake_minimum_required(VERSION 3.14)
project(s2cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

option(S2_VULKAN  "Build with Vulkan backend"  OFF)
option(S2_CUDA    "Build with CUDA backend"    OFF)
option(S2_METAL   "Build with Metal backend"   OFF)

set(S2_GPU_BACKEND_COUNT 0)
foreach(_backend S2_VULKAN S2_CUDA S2_METAL)
    if(${_backend})
        math(EXPR S2_GPU_BACKEND_COUNT "${S2_GPU_BACKEND_COUNT}+1")
    endif()
endforeach()
if(S2_GPU_BACKEND_COUNT GREATER 1)
    message(FATAL_ERROR "Choose only one GPU backend per executable: S2_VULKAN, S2_CUDA, or S2_METAL")
endif()
if(S2_METAL AND NOT APPLE)
    message(FATAL_ERROR "S2_METAL is supported only on macOS/Apple platforms")
endif()

# ---------------------------------------------------------------------------
# GGML
# ---------------------------------------------------------------------------

set(GGML_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Force every backend cache entry, including OFF. This prevents a reused build
# directory from accidentally carrying Vulkan into a CUDA build (or vice versa).
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
# Crow / Asio. Normal local builds can use find_package(Crow). CI/portable
# builds may pass header-only paths, which avoids requiring a package manager.
# ---------------------------------------------------------------------------
set(S2_CROW_INCLUDE_DIR "" CACHE PATH "Path containing crow.h (optional)")
set(S2_ASIO_INCLUDE_DIR "" CACHE PATH "Path containing asio.hpp (required with S2_CROW_INCLUDE_DIR)")
set(S2_HEADER_ONLY_CROW OFF)
if(S2_CROW_INCLUDE_DIR)
    if(NOT S2_ASIO_INCLUDE_DIR)
        message(FATAL_ERROR "S2_ASIO_INCLUDE_DIR is required when S2_CROW_INCLUDE_DIR is set")
    endif()
    if(NOT EXISTS "${S2_CROW_INCLUDE_DIR}/crow.h")
        message(FATAL_ERROR "crow.h not found under S2_CROW_INCLUDE_DIR=${S2_CROW_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS "${S2_ASIO_INCLUDE_DIR}/asio.hpp")
        message(FATAL_ERROR "asio.hpp not found under S2_ASIO_INCLUDE_DIR=${S2_ASIO_INCLUDE_DIR}")
    endif()
    set(S2_HEADER_ONLY_CROW ON)
else()
    find_package(Crow REQUIRED)
    if(NOT TARGET Crow::Crow)
        message(FATAL_ERROR "Crow package was found but does not provide the Crow::Crow target")
    endif()
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
    src/s2_voice.cpp
    src/main.cpp
)

# The Windows CI can generate an embedded tokenizer pair. Keep the root CMake
# consistent with the CI-generated CMake so a later local configure does not
# see tokenizer_data.h in main.cpp without linking tokenizer_data.cpp.
set(S2_TOKENIZER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer_data.h")
set(S2_TOKENIZER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer_data.cpp")
if(EXISTS "${S2_TOKENIZER_HEADER}" AND EXISTS "${S2_TOKENIZER_SOURCE}")
    list(APPEND S2_SOURCES src/tokenizer_data.cpp)
    message(STATUS "tokenizer embedded: src/tokenizer_data.cpp included")
elseif(EXISTS "${S2_TOKENIZER_HEADER}" OR EXISTS "${S2_TOKENIZER_SOURCE}")
    message(FATAL_ERROR "Incomplete embedded tokenizer pair: both src/tokenizer_data.h and src/tokenizer_data.cpp are required")
endif()

add_executable(s2 ${S2_SOURCES})

if(MSVC AND EXISTS "${S2_TOKENIZER_SOURCE}")
    set_source_files_properties(src/tokenizer_data.cpp PROPERTIES COMPILE_OPTIONS "/wd4309;/wd4365;/wd4838")
endif()

target_include_directories(s2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src
)

if(S2_HEADER_ONLY_CROW)
    target_include_directories(s2 PRIVATE ${S2_CROW_INCLUDE_DIR} ${S2_ASIO_INCLUDE_DIR})
    target_compile_definitions(s2 PRIVATE ASIO_STANDALONE)
    target_link_libraries(s2 PRIVATE ggml)
else()
    target_link_libraries(s2 PRIVATE ggml Crow::Crow)
endif()

if(S2_VULKAN)
    target_compile_definitions(s2 PRIVATE GGML_USE_VULKAN)
elseif(S2_CUDA)
    target_compile_definitions(s2 PRIVATE GGML_USE_CUDA)
elseif(S2_METAL)
    target_compile_definitions(s2 PRIVATE GGML_USE_METAL)
endif()

# Platform-specific
if(WIN32)
    # Crow/standalone Asio use Winsock directly in header-only builds. Keep the
    # root CMake usable on Windows too; the CI patch scripts should not be the
    # only place where these system libraries/definitions exist.
    target_link_libraries(s2 PRIVATE ws2_32 mswsock crypt32)
    target_compile_definitions(s2 PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _WIN32_WINNT=0x0A00)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(s2 PRIVATE pthread m)
endif()

# Install
install(TARGETS s2 RUNTIME DESTINATION bin)
