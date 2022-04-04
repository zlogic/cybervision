cmake_minimum_required(VERSION 3.21)

project(cybervision LANGUAGES CXX)

if (NOT DEFINED CYBERVISION_OPENCL)
    set(CYBERVISION_OPENCL ON CACHE BOOL "Enable OpenCL")
endif()
if (NOT DEFINED CYBERVISION_SSE)
    set(CYBERVISION_SSE ON CACHE BOOL "Enable SSE")
endif()
if (NOT DEFINED CYBERVISION_OPENMP)
    set(CYBERVISION_OPENMP ON CACHE BOOL "Enable OpenMP")
endif()
if (NOT DEFINED CYBERVISION_DEMO)
    set(CYBERVISION_DEMO OFF CACHE BOOL "Demo mode")
endif()

if(CYBERVISION_DEMO)
    list(APPEND cybervision_features "Demo")
endif()

if(NOT CYBERVISION_OPENCL)
    list(APPEND cybervision_features "noOpenCL")
endif()

if(NOT CYBERVISION_SSE)
    list(APPEND cybervision_features "noSSE")
endif()

if(NOT CYBERVISION_OPENMP)
    list(APPEND cybervision_features "noSSE")
endif()

if(cybervision_features)
    list(JOIN cybervision_features "_" CYBERVISION_SUFFIX)
    string(PREPEND CYBERVISION_SUFFIX "_")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(cybervision-app)
add_subdirectory(libsiftfast)
