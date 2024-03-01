cmake_minimum_required(VERSION 2.8)
project(vo)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++17 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV 4 REQUIRED)
find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

add_executable(main main.cpp)
add_library(util util.cpp)
add_library(stereo_vo stereo_vo.cpp)
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main util)
target_link_libraries(main stereo_vo)
