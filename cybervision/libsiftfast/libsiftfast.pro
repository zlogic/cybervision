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

win32-g++ {
	equals(CYBERVISION_OPENMP,true){
		QMAKE_CXXFLAGS += -fopenmp
		LIBS += -lgomp -lpthread
	}
	equals(CYBERVISION_SSE, true): QMAKE_CXXFLAGS += -msse3
	QMAKE_CXXFLAGS += -mstackrealign
}

unix {
	equals(CYBERVISION_OPENMP,true){
		QMAKE_CXXFLAGS += -fopenmp
		LIBS += -lgomp -lpthread
	}
	equals(CYBERVISION_SSE, true): QMAKE_CXXFLAGS += -msse3
}
