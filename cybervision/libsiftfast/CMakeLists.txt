cmake_minimum_required(VERSION 3.21)

project(libsiftfast LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_SOURCES
    SIFT/siftgateway.h
    SIFT/siftfast.h
    SIFT/msvc-exports.h
    SIFT/siftgateway.cpp
    SIFT/libsiftfast.cpp
)

find_package(Qt6 REQUIRED COMPONENTS Gui)

add_library(libsiftfast SHARED
    ${PROJECT_SOURCES}
)

target_link_libraries(libsiftfast PRIVATE Qt::Gui)

set_target_properties(libsiftfast PROPERTIES
    OUTPUT_NAME "siftfast${CYBERVISION_SUFFIX}"
)

target_include_directories(libsiftfast
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

if(CYBERVISION_OPENMP)
    target_link_libraries(libsiftfast PUBLIC
        gomp
        pthread
    )
    target_compile_options(libsiftfast PRIVATE
        -fopenmp
    )
endif()

if(CYBERVISION_SSE)
    target_compile_options(libsiftfast PRIVATE
        -msse3
    )
endif()
