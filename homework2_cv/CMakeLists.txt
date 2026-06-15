cmake_minimum_required(VERSION 3.20)
project(fisheye_motion_tracking VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Auto-detect conda OpenCV (handles MSVC version mismatch) ----
if(NOT OpenCV_DIR AND DEFINED ENV{CONDA_PREFIX})
    file(GLOB _candidates "$ENV{CONDA_PREFIX}/Library/cmake/x64/*/lib/OpenCVConfig.cmake")
    if(_candidates)
        list(GET _candidates 0 _first)
        get_filename_component(OpenCV_DIR "${_first}" DIRECTORY)
        message(STATUS "Auto-detected conda OpenCV: ${OpenCV_DIR}")
    endif()
endif()

# ---- Find OpenCV ----
find_package(OpenCV REQUIRED
    COMPONENTS core imgproc video calib3d highgui imgcodecs
)

# ---- Source files ----
set(SOURCES
    src/main.cpp
    src/fisheye_undistort.cpp
    src/motion_extraction.cpp
    src/tracking.cpp
    src/evaluation.cpp
    src/visualization.cpp
)

set(HEADERS
    src/fisheye_undistort.hpp
    src/motion_extraction.hpp
    src/tracking.hpp
    src/evaluation.hpp
    src/visualization.hpp
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# ---- Platform-specific ----
if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES WIN32_EXECUTABLE NO)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
endif()
