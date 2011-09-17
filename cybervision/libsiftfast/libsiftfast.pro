TARGET = siftfast

TEMPLATE = lib

CONFIG += shared

HEADERS += \
    SIFT/siftgateway.h \
    SIFT/siftfast.h

SOURCES += \
    SIFT/siftgateway.cpp \
    SIFT/libsiftfast.cpp

include( ../cybervision-options.pri )

equals(CYBERVISION_SSE, true){
    QMAKE_CXXFLAGS_RELEASE += -msse3
    #QMAKE_CXXFLAGS_DEBUG +=
}

win32 {
        LIBS += \
				-static -lgomp -lpthread.dll

	QMAKE_CXXFLAGS += -U_WIN32 -fopenmp
	QMAKE_CFLAGS += -U_WIN32
}
unix {
        LIBS += -lgomp \
		-lpthread
	QMAKE_CXXFLAGS += -fopenmp
}
