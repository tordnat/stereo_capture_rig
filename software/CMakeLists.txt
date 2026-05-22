cmake_minimum_required(VERSION 2.8)
project(zed_capture_suite CXX)

option(LINK_SHARED_ZED "Link with the ZED SDK shared library" ON)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 OLD)
    cmake_policy(SET CMP0015 OLD)
endif()

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

set(EXECUTABLE_OUTPUT_PATH ".")

find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

add_definitions(-std=c++14 -O3)

if(LINK_SHARED_ZED)
    set(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    set(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

add_executable(zed_record zed_record.cpp)
target_link_libraries(zed_record ${ZED_LIBS} ${OpenCV_LIBS} stdc++fs pthread)

add_executable(zed_export zed_export.cpp)
target_link_libraries(zed_export ${ZED_LIBS} ${OpenCV_LIBS} stdc++fs pthread)