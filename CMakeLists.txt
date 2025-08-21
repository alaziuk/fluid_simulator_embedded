cmake_minimum_required(VERSION 3.20)
project(fluid2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ETL
FetchContent_Declare(
        etl
        GIT_REPOSITORY https://github.com/ETLCPP/etl.git
        GIT_TAG 20.42.2
)
FetchContent_MakeAvailable(etl)

add_library(fluid2d STATIC
        src/fluid2d.cpp)

target_include_directories(fluid2d
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(fluid2d
        PUBLIC etl::etl
)

# set(FLUID2D_BUILD_VIEWER ON CACHE BOOL "Build the SDL2 gravity-arrow test app")

# ---------------- Optional viewer ----------------
# Decide the default: ON if top-level, OFF if included
if (NOT DEFINED FLUID2D_BUILD_VIEWER)
    if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        set(FLUID2D_BUILD_VIEWER_DEFAULT ON)
    else()
        set(FLUID2D_BUILD_VIEWER_DEFAULT OFF)
    endif()
else()
    # If the user already set FLUID2D_BUILD_VIEWER, use that as default
    set(FLUID2D_BUILD_VIEWER_DEFAULT ${FLUID2D_BUILD_VIEWER})
endif()

option(FLUID2D_BUILD_VIEWER "Build the SDL2 gravity-arrow test app" ${FLUID2D_BUILD_VIEWER_DEFAULT})

if (FLUID2D_BUILD_VIEWER)
    # Try CMake's FindSDL2 first
    find_package(SDL2 QUIET)

    # Fallback to pkg-config if needed
    if (NOT SDL2_FOUND)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(SDL2 REQUIRED sdl2)
        add_library(SDL2::SDL2 INTERFACE IMPORTED)
        target_include_directories(SDL2::SDL2 INTERFACE ${SDL2_INCLUDE_DIRS})
        target_link_libraries(SDL2::SDL2 INTERFACE ${SDL2_LDFLAGS})
    endif()

    add_executable(viewer
            apps/viewer/main.cpp   # adjust path if needed
    )

    target_link_libraries(viewer
            PRIVATE fluid2d
            PRIVATE SDL2::SDL2
    )

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(viewer PRIVATE -Wall -Wextra -Wpedantic)
    elseif (MSVC)
        target_compile_options(viewer PRIVATE /W4)
    endif()
endif()  # <-- closes if(FLUID2D_BUILD_VIEWER)
