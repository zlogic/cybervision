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
	SIFT/siftgateway.cpp \
	SIFT/libsiftfast.cpp \
	UI/cybervisionviewer.cpp \
	Reconstruction/sculptor.cpp \
	Reconstruction/surface.cpp \
	KDTree/kdtreegateway.cpp \
    Reconstruction/imageloader.cpp
HEADERS += \
	Reconstruction/pointmatcher.h \
	Reconstruction/pointmatch.h \
	Reconstruction/fundamentalmatrix.h \
	Reconstruction/reconstructor.h \
	Reconstruction/pointtriangulator.h \
	Reconstruction/options.h \
	UI/processthread.h \
	SIFT/siftgateway.h \
	SIFT/siftfast.h \
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
    Reconstruction/imageloader.h

FORMS += UI/mainwindow.ui

QMAKE_CXXFLAGS_RELEASE += -msse3
QMAKE_CXXFLAGS_DEBUG +=
win32 { 
	QMAKE_LIBS += -static \
		-lgomp \
		-lpthread.dll
	QMAKE_CXXFLAGS += -U_WIN32 -fopenmp
	QMAKE_CFLAGS += -U_WIN32
}
unix {
	QMAKE_LIBS += -lgomp \
		-lpthread
	QMAKE_CXXFLAGS += -fopenmp
}

RESOURCES += \
	ReconstructionResources.qrc

OTHER_FILES += \
	Reconstruction/ColladaTemplate.xml \
    UI/icons/arrow-move.png \
    UI/icons/arrow-circle.png
