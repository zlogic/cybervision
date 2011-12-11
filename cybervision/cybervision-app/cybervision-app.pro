include( ../cybervision-options.pri )

QT += core gui opengl
TARGET = cybervision$${CYBERVISION_SUFFIX}
TEMPLATE = app
SOURCES += main.cpp \
	UI/mainwindow.cpp \
	Reconstruction/pointmatcher.cpp \
	Reconstruction/pointmatch.cpp \
	Reconstruction/fundamentalmatrix.cpp \
	Reconstruction/pointtriangulator.cpp \
	Reconstruction/reconstructor.cpp \
	Reconstruction/options.cpp \
	Reconstruction/sculptor.cpp \
	Reconstruction/surface.cpp \
	Reconstruction/imageloader.cpp \
	Reconstruction/pointmatcheropencl.cpp \
	Reconstruction/crosssection.cpp \
	KDTree/kdtreegateway.cpp \
	UI/processthread.cpp \
	UI/cybervisionviewer.cpp \
	UI/crosssectionwindow.cpp

HEADERS += \
	Reconstruction/pointmatcher.h \
	Reconstruction/pointmatch.h \
	Reconstruction/fundamentalmatrix.h \
	Reconstruction/reconstructor.h \
	Reconstruction/pointtriangulator.h \
	Reconstruction/options.h \
	Reconstruction/sculptor.h \
	Reconstruction/surface.h \
	Reconstruction/imageloader.h \
	Reconstruction/pointmatcheropencl.h \
	Reconstruction/crosssection.h \
	KDTree/region.hpp \
	KDTree/node.hpp \
	KDTree/kdtreegateway.h \
	KDTree/kdtree.hpp \
	KDTree/iterator.hpp \
	KDTree/function.hpp \
	KDTree/allocator.hpp \
	KDTree/region.hpp \
	KDTree/node.hpp \
	KDTree/kdtree.hpp \
	KDTree/iterator.hpp \
	KDTree/function.hpp \
	KDTree/allocator.hpp \
	Eigen/SVD \
	Eigen/StdVector \
	Eigen/StdList \
	Eigen/StdDeque \
	Eigen/Sparse \
	Eigen/QtAlignedMalloc \
	Eigen/QR \
	Eigen/LU \
	Eigen/LeastSquares \
	Eigen/Jacobi \
	Eigen/Householder \
	Eigen/Geometry \
	Eigen/Eigenvalues \
	Eigen/Eigen2Support \
	Eigen/Core \
	Eigen/Cholesky \
	Eigen/Array \
	Eigen/Eigen \
	Eigen/Dense \
	UI/mainwindow.h \
	UI/processthread.h \
	UI/cybervisionviewer.h \
	UI/crosssectionwindow.h

FORMS += UI/mainwindow.ui \
    UI/crosssectionwindow.ui

RESOURCES += \
	ReconstructionResources.qrc

OTHER_FILES += \
	Reconstruction/ColladaTemplate.xml \
	Reconstruction/PointMatcherKernel.cl \
	Reconstruction/SceneJSTemplate.js \
	UI/cybervision.rc \
	UI/translations/cybervision-app_ru.ts \
    UI/icons/texture-right.png \
    UI/icons/texture-left.png \
    UI/icons/texture-empty.png \
    UI/icons/plus.png \
    UI/icons/pencil-ruler.png \
    UI/icons/minus.png \
    UI/icons/grid.png \
    UI/icons/document-import.png \
    UI/icons/disk.png \
    UI/icons/cybervision.png \
    UI/icons/cybervision.ico \
    UI/icons/border-draw.png \
    UI/icons/arrow-move.png \
    UI/icons/arrow-circle.png

TRANSLATIONS = UI/translations/cybervision-app_ru.ts
CODECFORTR = UTF-8

INCLUDEPATH += $$PWD/../libsiftfast
DEPENDPATH += $$PWD/../libsiftfast


#CONFIG(release, debug|release): DESTDIR = ../release
#CONFIG(debug, debug|release): DESTDIR = ../debug

equals(CYBERVISION_OPENCL, true){
    DEFINES += CYBERVISION_OPENCL
}

equals(CYBERVISION_DEMO, true){
	DEFINES += CYBERVISION_DEMO
}

win32 {
    RC_FILE = UI/cybervision.rc
}
win32-g++ {
    QMAKE_CXXFLAGS += -fopenmp
    equals(CYBERVISION_SSE, true): QMAKE_CXXFLAGS += -msse3
    LIBS += \
            -static -lgcc_eh \
            -lgomp -lpthread

    equals(CYBERVISION_OPENCL, true){
        LIBS += -lOpenCL -laticalrt -laticalcl
    }

    #QMAKE_CXXFLAGS += -U_WIN32
    #QMAKE_CFLAGS += -U_WIN32
}

win32-msvc* {
    INCLUDEPATH += $$quote(C:/QtSDK/MSVC-Libs/include)

    QMAKE_CXXFLAGS_RELEASE += /O2
    QMAKE_CXXFLAGS += /openmp

    equals(CYBERVISION_SSE, true): QMAKE_CXXFLAGS += /arch:SSE2

    equals(CYBERVISION_OPENCL, true){
        QMAKE_LIBDIR += $$quote(C:/QtSDK/MSVC-Libs/lib/x86)
        LIBS += -lOpenCL
    }
}
unix {
    QMAKE_CXXFLAGS += -fopenmp -msse3
    equals(CYBERVISION_OPENCL, true){
		INCLUDEPATH += /opt/AMDAPP/include
        LIBS += -L/opt/AMDAPP/lib/x86_64 -lOpenCL
    }
}

win32-msvc*:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../libsiftfast/release -lsiftfast$${CYBERVISION_SUFFIX}
else:win32-msvc*:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../libsiftfast/debug -lsiftfast$${CYBERVISION_SUFFIX}
else:win32-g++:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../libsiftfast/release -dynamic -lsiftfast$${CYBERVISION_SUFFIX}
else:win32-g++:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../libsiftfast/debug -dynamic -lsiftfast$${CYBERVISION_SUFFIX}
else:win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../libsiftfast/release -lsiftfast$${CYBERVISION_SUFFIX}
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../libsiftfast/debug -lsiftfast$${CYBERVISION_SUFFIX}
else:symbian: LIBS += -lsiftfast$${CYBERVISION_SUFFIX}
else:unix: LIBS += -L$$OUT_PWD/../libsiftfast -dynamic -lsiftfast$${CYBERVISION_SUFFIX}
