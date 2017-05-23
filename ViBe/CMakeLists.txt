cmake_minimum_required (VERSION 2.8)

project(ViBe)
find_package(OpenCV REQUIRED)

include_directories (${OpenCV_INCLUDE_DIRS})

add_library(ViBe SHARED ViBe.cpp)
add_executable(ViBe_main ViBe_main.cpp)
target_link_libraries(ViBe_main ViBe ${OpenCV_LIBS})
