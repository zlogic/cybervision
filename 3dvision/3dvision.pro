# -------------------------------------------------
# Project created by QtCreator 2009-10-25T19:45:46
# -------------------------------------------------
QT += opengl
TARGET = 3dvision
TEMPLATE = app
SOURCES += main.cpp \
    UI/mainwindow.cpp \
    Reconstruction/reconstructor.cpp \
    Reconstruction/options.cpp \
    UI/processthread.cpp \
    SIFT/siftgateway.cpp \
    SIFT/libsiftfast.cpp \
    UI/cybervisionviewer.cpp \
    Reconstruction/sculptor.cpp \
    Reconstruction/surface.cpp
HEADERS += UI/mainwindow.h \
    Reconstruction/svd.h \
    Reconstruction/reconstructor.h \
    Reconstruction/options.h \
    UI/processthread.h \
    SIFT/siftgateway.h \
    SIFT/siftfast.h \
    UI/cybervisionviewer.h \
    Reconstruction/sculptor.h \
    Reconstruction/surface.h
FORMS += UI/mainwindow.ui
OTHER_FILES += SIFT/optimization_flags.txt \
    SIFT/3dvision_pro.txt

QMAKE_CXXFLAGS_RELEASE += -msse3
QMAKE_CXXFLAGS_DEBUG +=

win32 { 
	QMAKE_LIBS += -static \
		-lgomp \
		-lpthread
	QMAKE_CXXFLAGS += -U_WIN32 -fopenmp
	QMAKE_CFLAGS += -U_WIN32
}
unix { 
    QMAKE_LIBS += -lgomp \
        -lpthread
    QMAKE_CXXFLAGS += -fopenmp
}
