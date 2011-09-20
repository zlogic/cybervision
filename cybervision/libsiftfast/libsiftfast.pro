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

QMAKE_CXXFLAGS += -msse3 -fopenmp

win32 {
	LIBS += \
			-static -lgcc_eh -lgomp -lpthread

	QMAKE_CXXFLAGS += -U_WIN32
	QMAKE_CFLAGS += -U_WIN32
}
unix {
        LIBS += -lgomp \
		-lpthread
}
