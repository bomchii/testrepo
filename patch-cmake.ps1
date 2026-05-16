# patch-cmake.ps1
# Parchea CMakeLists.txt principal y ggml-vulkan/CMakeLists.txt para Windows

# ── 1. Parchear ggml/src/ggml-vulkan/CMakeLists.txt ─────────────────────────
# La funcion test_shader_extension_support usa glslc para detectar soporte de
# coopmat/coopmat2/integer_dot. glslc en el runner SI los soporta, los activa,
# y ggml-vulkan.cpp espera esos simbolos del .hpp generado por vulkan-shaders-gen.
# El problema es un race condition: ggml-vulkan.cpp compila en paralelo antes
# de que vulkan-shaders-gen termine de generar el .hpp con esas variantes.
# Solucion: reemplazar la funcion de deteccion para que siempre devuelva OFF.

$vkCmake = Get-Content "ggml\src\ggml-vulkan\CMakeLists.txt" -Raw -Encoding UTF8

$oldFunc = 'function(test_shader_extension_support EXTENSION_NAME TEST_SHADER_FILE RESULT_VARIABLE)
    execute_process(
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o - -fshader-stage=compute --target-env=vulkan1.3 "${TEST_SHADER_FILE}"
        OUTPUT_VARIABLE glslc_output
        ERROR_VARIABLE glslc_error
    )

    if (${glslc_error} MATCHES ".*extension not supported: ${EXTENSION_NAME}.*")
        message(STATUS "${EXTENSION_NAME} not supported by glslc")
        set(${RESULT_VARIABLE} OFF PARENT_SCOPE)
    else()
        message(STATUS "${EXTENSION_NAME} supported by glslc")
        set(${RESULT_VARIABLE} ON PARENT_SCOPE)
        add_compile_definitions(${RESULT_VARIABLE})

        # Ensure the extension support is forwarded to vulkan-shaders-gen
        list(APPEND VULKAN_SHADER_GEN_CMAKE_ARGS -D${RESULT_VARIABLE}=ON)
        set(VULKAN_SHADER_GEN_CMAKE_ARGS "${VULKAN_SHADER_GEN_CMAKE_ARGS}" PARENT_SCOPE)
    endif()
endfunction()'

$newFunc = 'function(test_shader_extension_support EXTENSION_NAME TEST_SHADER_FILE RESULT_VARIABLE)
    # Forzado a OFF para evitar race condition entre vulkan-shaders-gen y ggml-vulkan.cpp
    message(STATUS "${EXTENSION_NAME} disabled (portability build)")
    set(${RESULT_VARIABLE} OFF PARENT_SCOPE)
endfunction()'

$vkCmake = $vkCmake.Replace($oldFunc, $newFunc)
Set-Content "ggml\src\ggml-vulkan\CMakeLists.txt" -Value $vkCmake -Encoding UTF8
Write-Host "ggml-vulkan/CMakeLists.txt parcheado."

# ── 2. Parchear CMakeLists.txt raiz ─────────────────────────────────────────
$content = Get-Content CMakeLists.txt -Raw -Encoding UTF8

$oldCrow = 'find_package(Crow REQUIRED)'

$newCrow = 'include(FetchContent)

# Asio PRIMERO: Crow lo busca internamente con find_package(asio REQUIRED)
FetchContent_Declare(asio
    GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
    GIT_TAG        asio-1-30-2
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(asio)
set(ASIO_INCLUDE_DIR "${asio_SOURCE_DIR}/asio/include" CACHE PATH "" FORCE)

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

add_library(asio_iface INTERFACE)
target_include_directories(asio_iface INTERFACE "${asio_SOURCE_DIR}/asio/include")
target_compile_definitions(asio_iface INTERFACE ASIO_STANDALONE)'

$content = $content.Replace($oldCrow, $newCrow)

$oldLibs = 'if(UNIX AND NOT APPLE)
    target_link_libraries(s2 PRIVATE pthread m)
endif()'

$newLibs = 'if(TARGET asio_iface)
    target_link_libraries(s2 PRIVATE asio_iface)
    target_compile_definitions(s2 PRIVATE ASIO_STANDALONE CROW_ENABLE_SSL=0)
endif()
if(WIN32)
    target_link_libraries(s2 PRIVATE ws2_32 mswsock)
    target_compile_definitions(s2 PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00)
    if(MSVC)
        target_compile_options(s2 PRIVATE /W3 /wd4996 /MP)
    endif()
elseif(UNIX AND NOT APPLE)
    target_link_libraries(s2 PRIVATE pthread m)
endif()'

$content = $content.Replace($oldLibs, $newLibs)

Set-Content CMakeLists.txt -Value $content -Encoding UTF8
Write-Host "CMakeLists.txt raiz parcheado."
