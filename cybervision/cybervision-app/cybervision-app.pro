# -------------------------------------------------
# Project created by QtCreator 2009-10-25T19:45:46
# -------------------------------------------------
QT += core gui opengl
TARGET = cybervision
TEMPLATE = app
SOURCES += main.cpp \
	UI/mainwindow.cpp \
	Reconstruction/pointmatcher.cpp \
	Reconstruction/pointmatch.cpp \
	Reconstruction/fundamentalmatrix.cpp \
	Reconstruction/pointtriangulator.cpp \
	Reconstruction/reconstructor.cpp \
	Reconstruction/options.cpp \
	UI/processthread.cpp \
	UI/cybervisionviewer.cpp \
	Reconstruction/sculptor.cpp \
	Reconstruction/surface.cpp \
	KDTree/kdtreegateway.cpp \
    Reconstruction/imageloader.cpp \
    Reconstruction/pointmatcheropencl.cpp
HEADERS += \
	Reconstruction/pointmatcher.h \
	Reconstruction/pointmatch.h \
	Reconstruction/fundamentalmatrix.h \
	Reconstruction/reconstructor.h \
	Reconstruction/pointtriangulator.h \
	Reconstruction/options.h \
	UI/processthread.h \
	UI/cybervisionviewer.h \
	Reconstruction/sculptor.h \
	Reconstruction/surface.h \
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
    Reconstruction/imageloader.h \
    Reconstruction/pointmatcheropencl.h

FORMS += UI/mainwindow.ui

RESOURCES += \
	ReconstructionResources.qrc

OTHER_FILES += \
	Reconstruction/ColladaTemplate.xml \
	UI/icons/arrow-move.png \
	UI/icons/arrow-circle.png \
	UI/icons/grid.png \
	Reconstruction/PointMatcherKernel.cl \
	UI/icons/texture-right.png \
	UI/icons/texture-left.png \
	UI/icons/texture-empty.png \
	UI/icons/cybervision.png \
	UI/cybervision.rc \
	UI/icons/cybervision.ico

include( ../cybervision-options.pri )

QMAKE_CXXFLAGS += -fopenmp

INCLUDEPATH += $$PWD/../libsiftfast
DEPENDPATH += $$PWD/../libsiftfast

equals(CYBERVISION_OPENCL, true){
	DEFINES += CYBERVISION_OPENCL
}

win32 {
	RC_FILE = UI/cybervision.rc
	LIBS += \
			-static -lgcc_eh \
			-lgomp -lpthread.dll

	equals(CYBERVISION_OPENCL, true){
		LIBS += -lOpenCL -laticalrt -laticalcl
	}

	QMAKE_CXXFLAGS += -U_WIN32
	QMAKE_CFLAGS += -U_WIN32
}
unix {
	equals(CYBERVISION_OPENCL, true){
		LIBS += -L/opt/AMDAPP/lib/x86_64 -lOpenCL
	}
}

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../libsiftfast/release/ -dynamic -lsiftfast
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../libsiftfast/debug/ -dynamic -lsiftfast
else:symbian: LIBS += -lsiftfast
else:unix: LIBS += -L$$OUT_PWD/../libsiftfast -dynamic -lsiftfast
