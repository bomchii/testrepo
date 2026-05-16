# patch-cmake.ps1
# Parchea CMakeLists.txt para compilar en Windows con Crow via FetchContent

$content = Get-Content CMakeLists.txt -Raw -Encoding UTF8

# ── 1. Reemplazar find_package(Crow REQUIRED) ────────────────────────────────
# Orden critico:
#   1. Descargar Asio primero y setear ASIO_INCLUDE_DIR en cache
#   2. Solo despues descargar Crow (su Findasio.cmake lo encontrara)
$oldCrow = 'find_package(Crow REQUIRED)'

$newCrow = 'include(FetchContent)

# Asio PRIMERO: Crow lo busca internamente con find_package(asio REQUIRED)
# Si no esta en cache antes de FetchContent_MakeAvailable(crow), falla.
FetchContent_Declare(asio
    GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
    GIT_TAG        asio-1-30-2
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(asio)
# Setear ASIO_INCLUDE_DIR en cache para que Findasio.cmake de Crow lo encuentre
set(ASIO_INCLUDE_DIR "${asio_SOURCE_DIR}/asio/include" CACHE PATH "" FORCE)

# Crow: ahora si puede encontrar Asio
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

# Libreria de interfaz para propagar Asio a nuestro target
add_library(asio_iface INTERFACE)
target_include_directories(asio_iface INTERFACE "${asio_SOURCE_DIR}/asio/include")
target_compile_definitions(asio_iface INTERFACE ASIO_STANDALONE)'

$content = $content.Replace($oldCrow, $newCrow)

# ── 2. Agregar libs de Windows al target s2 ───────────────────────────────────
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

# ── 3. Guardar ────────────────────────────────────────────────────────────────
Set-Content CMakeLists.txt -Value $content -Encoding UTF8
Write-Host "CMakeLists.txt parcheado correctamente."
