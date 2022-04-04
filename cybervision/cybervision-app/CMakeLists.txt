cmake_minimum_required(VERSION 3.21)

project(cybervision-app LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets 3DCore 3DRender 3DExtras)

set(PROJECT_SOURCES
    main.cpp
    UI/mainwindow.cpp
    Reconstruction/pointmatcher.cpp
    Reconstruction/pointmatch.cpp
    Reconstruction/fundamentalmatrix.cpp
    Reconstruction/pointtriangulator.cpp
    Reconstruction/reconstructor.cpp
    Reconstruction/options.cpp
    Reconstruction/sculptor.cpp
    Reconstruction/surface.cpp
    Reconstruction/imageloader.cpp
    Reconstruction/pointmatcheropencl.cpp
    Reconstruction/crosssection.cpp
    KDTree/kdtreegateway.cpp
    UI/processthread.cpp
    UI/cybervisionviewer.cpp
    UI/crosssectionwindow.cpp
    UI/aboutwindow.cpp
    Reconstruction/pointmatcher.h
    Reconstruction/pointmatch.h
    Reconstruction/fundamentalmatrix.h
    Reconstruction/reconstructor.h
    Reconstruction/pointtriangulator.h
    Reconstruction/options.h
    Reconstruction/sculptor.h
    Reconstruction/surface.h
    Reconstruction/imageloader.h
    Reconstruction/pointmatcheropencl.h
    Reconstruction/crosssection.h
    Reconstruction/config.h
    KDTree/kdtreegateway.h
    UI/mainwindow.h
    UI/processthread.h
    UI/cybervisionviewer.h
    UI/crosssectionwindow.h
    UI/aboutwindow.h
    UI/mainwindow.ui
    UI/crosssectionwindow.ui
    UI/aboutwindow.ui
    ReconstructionResources.qrc
    UI/translations/cybervision-app_ru.ts
)

qt_add_executable(cybervision-app
    MANUAL_FINALIZATION
    ${PROJECT_SOURCES}
)

target_link_libraries(cybervision-app PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets Qt6::3DCore Qt6::3DRender Qt6::3DExtras)
target_link_libraries(cybervision-app PRIVATE libsiftfast)

if(CYBERVISION_OPENCL)
    target_compile_definitions(cybervision-app PRIVATE
        CYBERVISION_OPENCL
    )
    target_link_libraries(cybervision-app PRIVATE
        OpenCL
    )
endif()

if(CYBERVISION_OPENMP)
    target_link_libraries(cybervision-app PUBLIC
        gomp
        pthread
    )
    target_compile_options(cybervision-app PRIVATE
        -fopenmp
    )
endif()

if(CYBERVISION_SSE)
    target_compile_options(cybervision-app PRIVATE
        -msse3
    )
endif()

if(CYBERVISION_DEMO)
    target_compile_definitions(cybervision-app PRIVATE
        CYBERVISION_DEMO
    )
endif()

target_compile_options(cybervision-app PRIVATE
    $<$<CONFIG:Debug>:-Wa,-mbig-obj>
)

set_target_properties(cybervision-app PROPERTIES
    WIN32_EXECUTABLE TRUE
    OUTPUT_NAME "cybervision${CYBERVISION_SUFFIX}"
)

qt_finalize_executable(cybervision-app)
