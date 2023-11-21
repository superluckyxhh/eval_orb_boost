cmake_minimum_required(VERSION 3.5)
PROJECT(orb_slam_extractor)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

find_package(OpenCV REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
)

add_subdirectory(src)
add_subdirectory(pybind11)

# link_directories(${OpenCV_LIBRARY_DIRS})
pybind11_add_module(orb_slam2_extractor orb_extractor.cpp src/ORB_extractor.cpp)
target_link_libraries(orb_slam2_extractor PRIVATE
        ${OpenCV_LIBS}
        src
)