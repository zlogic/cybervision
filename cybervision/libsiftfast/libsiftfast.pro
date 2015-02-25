include( ../cybervision-options.pri )

TARGET = siftfast$${CYBERVISION_SUFFIX}

TEMPLATE = lib

CONFIG += shared

HEADERS += \
    SIFT/siftgateway.h \
    SIFT/siftfast.h \
    SIFT/msvc-exports.h

SOURCES += \
    SIFT/siftgateway.cpp \
    SIFT/libsiftfast.cpp

#CONFIG(release, debug|release): DESTDIR = ../release
#CONFIG(debug, debug|release): DESTDIR = ../debug

win32-g++ {
	equals(CYBERVISION_OPENMP,true){
		QMAKE_CXXFLAGS += -fopenmp
		LIBS += -lgomp -lpthread
	}
	equals(CYBERVISION_SSE, true): QMAKE_CXXFLAGS += -msse3
	QMAKE_CXXFLAGS += -mstackrealign
}
win32-msvc* {
	DEFINES += DLL_EXPORTS

	equals(CYBERVISION_OPENMP,true): QMAKE_CXXFLAGS += /openmp
	equals(CYBERVISION_SSE, true): QMAKE_CXXFLAGS += /arch:SSE2

	QMAKE_CXXFLAGS_RELEASE += /O2
}

unix {
	equals(CYBERVISION_OPENMP,true){
		QMAKE_CXXFLAGS += -fopenmp
		LIBS += -lgomp -lpthread
	}
	equals(CYBERVISION_SSE, true): QMAKE_CXXFLAGS += -msse3
}
