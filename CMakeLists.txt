cmake_minimum_required(VERSION 3.16)

project(snake-evolve LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(snake-evolve
    src/main.cpp
    src/snake.cpp
    src/game.cpp
)

target_include_directories(snake-evolve PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(snake-evolve PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)