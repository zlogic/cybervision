TARGET = siftfast

TEMPLATE = lib

CONFIG += shared

HEADERS += \
    SIFT/siftgateway.h \
    SIFT/siftfast.h \
    SIFT/msvc-exports.h

SOURCES += \
    SIFT/siftgateway.cpp \
    SIFT/libsiftfast.cpp

include( ../cybervision-options.pri )


#CONFIG(release, debug|release): DESTDIR = ../release
#CONFIG(debug, debug|release): DESTDIR = ../debug

win32-g++ {
    LIBS += \
        -static -lgcc_eh -lgomp -lpthread

	QMAKE_CXXFLAGS += -fopenmp
	equals(CYBERVISION_SSE, true): QMAKE_CXXFLAGS += -msse3
    QMAKE_CXXFLAGS += -U_WIN32
    QMAKE_CFLAGS += -U_WIN32
}
win32-msvc* {
	DEFINES += DLL_EXPORTS

	QMAKE_CXXFLAGS += /openmp
	equals(CYBERVISION_SSE, true): QMAKE_CXXFLAGS += /arch:SSE2

	QMAKE_CXXFLAGS_RELEASE += /O2
}

unix {
    LIBS += -lgomp \
            -lpthread

    QMAKE_CXXFLAGS += -msse3 -fopenmp
}
