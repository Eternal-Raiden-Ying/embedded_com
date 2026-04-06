cmake_minimum_required (VERSION 3.5)
project("run_test")

find_package(OpenCV REQUIRED)

message(STATUS "oPENCV Library status:")
message(STATUS ">version:${OpenCV_VERSION}")
message(STATUS "Include:${OpenCV_INCLUDE_DIRS}")

set(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations -Wno-deprecated-declarations")

include_directories(
    /usr/local/include
    /usr/include/opencv4
)

link_directories(
    /usr/local/lib/
)

file(GLOB SRC_LISTS 
    ${CMAKE_CURRENT_SOURCE_DIR}/run_test.cpp  
)

add_executable(run_test ${SRC_LISTS})

target_link_libraries(run_test
    aidlite
	${OpenCV_LIBS}
    pthread
)
