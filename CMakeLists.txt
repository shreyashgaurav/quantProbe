cmake_minimum_required(VERSION 4.2)
project(quantProbe)

set(CMAKE_CXX_STANDARD 17)

add_executable(quantProbe main.cpp
        src/main.cpp
        src/loader.cpp)
