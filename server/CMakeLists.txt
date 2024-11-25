cmake_minimum_required(VERSION 3.13)
project(server)


set(OpenCV_DIR /Users/gyujinkim/Desktop/Github/monitor-vehicle-api/server/opencv/build)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# TVM 경로 설정
set(TVM_PATH /Users/gyujinkim/Desktop/Ai/TinyEngine3/tvm)
include_directories("${TVM_PATH}/include")
include_directories("${TVM_PATH}/3rdparty/dmlc-core/include")
include_directories("${TVM_PATH}/3rdparty/dlpack/include")
include_directories(${EIGEN3_INCLUDE_DIR})
link_directories("${TVM_PATH}/build")

# GStreamer 라이브러리 찾기
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)

# GStreamer 라이브러리 경로 추가
link_directories(${GSTREAMER_LIBRARY_DIRS})

# C++17 이상을 사용할 수 있도록 설정
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ENV{TVM_LOG_DEBUG} 1)

# src 디렉토리를 포함 경로에 추가
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${GSTREAMER_INCLUDE_DIRS} ${GSTREAMER_APP_INCLUDE_DIRS})

set(SOURCE_SERVER_DIR
    src/server.cpp
    src/response.cpp
    src/hungarian.cpp
    src/kalman_filter.cpp
    src/line.cpp
    src/munkres.cpp
    src/ocr.cpp
    src/track.cpp
    src/tracker.cpp
    src/tvm_wrapper.cpp
)

add_executable(server ${SOURCE_SERVER_DIR} example/example_server.cpp)
target_link_libraries(server
    tvm
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBS}
    ${Boost_LIBRARIES}
)

add_executable(client example/example_client.cpp)
target_link_libraries(client
    ${OpenCV_LIBS}
)



# 시스템 라이브러리 및 TVM 라이브러리 링크
target_include_directories(server PRIVATE ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
