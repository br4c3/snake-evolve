cmake_minimum_required(VERSION 3.16)

project(snake-evolve LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

find_package(raylib 6.0 QUIET)

if(NOT raylib_FOUND)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_GAMES OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        raylib
        URL https://github.com/raysan5/raylib/archive/refs/tags/6.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    )
    FetchContent_MakeAvailable(raylib)
endif()

add_executable(snake-evolve
    src/main.cpp
    src/snake.cpp
    src/game.cpp
    src/agent.cpp
    src/renderer.cpp
)

target_include_directories(snake-evolve PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(snake-evolve PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

target_link_libraries(snake-evolve PRIVATE raylib)
