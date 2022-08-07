cmake_minimum_required(VERSION 3.16)

project(cybervision VERSION 0.5)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

SET(CYBERVISION_SOURCES
    cybervision/correlation.c
    cybervision/fast_detector.c
    cybervision/configuration.c
    cybervision/system.c
    cybervision/triangulation.c
    cybervision/main.c
)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads REQUIRED)
if(CMAKE_USE_PTHREADS_INIT)
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-pthread")
elseif(CMAKE_USE_WIN32_THREADS_INIT)
    file(GLOB CYBERVISION_SOURCES_WIN32 "cybervision/win32/*.c")
    list(APPEND CYBERVISION_SOURCES ${CYBERVISION_SOURCES_WIN32})
else()
    message(FATAL_ERROR "Unsupported threads library ${CMAKE_THREAD_LIBS_INIT}")
endif()

add_executable(cybervision ${CYBERVISION_SOURCES})

find_package(Qhull CONFIG REQUIRED)
find_package(JPEG REQUIRED)
find_package(TIFF REQUIRED)
find_library(FASTFEAT_LIBRARY fastfeat)

target_link_libraries(cybervision PRIVATE Qhull::qhull_r)
target_include_directories(cybervision PRIVATE ${JPEG_INCLUDE_DIR})
target_link_libraries(cybervision PRIVATE JPEG::JPEG)
target_link_libraries(cybervision PRIVATE TIFF::TIFF)
target_link_libraries(cybervision PRIVATE ${FASTFEAT_LIBRARY})

if(NOT MSVC)
    target_link_libraries(cybervision PUBLIC m)
endif()
