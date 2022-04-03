include( ../cybervision-options.pri )

QT += core gui widgets 3dcore 3drender 3dextras
TARGET = cybervision$${CYBERVISION_SUFFIX}
TEMPLATE = app

CONFIG += exceptions

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
	UI/crosssectionwindow.cpp \
	UI/aboutwindow.cpp

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
	Reconstruction/config.h \
	KDTree/kdtreegateway.h \
	UI/mainwindow.h \
	UI/processthread.h \
	UI/cybervisionviewer.h \
	UI/crosssectionwindow.h \
	UI/aboutwindow.h

FORMS += UI/mainwindow.ui \
	UI/crosssectionwindow.ui \
	UI/aboutwindow.ui

RESOURCES += \
	ReconstructionResources.qrc

OTHER_FILES += \
	Reconstruction/ColladaTemplate.xml \
	Reconstruction/PointMatcherKernel.cl \
	Reconstruction/SceneJSTemplate.js \
	UI/cybervision.rc \
        UI/translations/cybervision-app_ru.ts

DISTFILES += \
	UI/icons/cybervision.png \
	UI/icons/cybervision.ico \
	UI/icons/cybervision-large.svg \
	UI/icons/arrow-repeat.svg \
	UI/icons/arrows-move.svg \
	UI/icons/back.svg \
	UI/icons/badge-3d.svg \
	UI/icons/collection.svg \
	UI/icons/easel.svg \
	UI/icons/file-earmark-easel.svg \
	UI/icons/file-earmark-image.svg \
	UI/icons/file-earmark-minus.svg \
	UI/icons/file-earmark-plus.svg \
	UI/icons/front.svg \
	UI/icons/grid-3x3.svg \
	UI/icons/info-circle.svg \
	UI/icons/rulers.svg \
	UI/icons/save.svg \
	UI/icons/search.svg \
	UI/icons/sliders.svg \
	UI/icons/x-square.svg

TRANSLATIONS = UI/translations/cybervision-app_ru.ts
CODECFORTR = UTF-8

INCLUDEPATH += $$PWD/../libsiftfast
DEPENDPATH += $$PWD/../libsiftfast

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
	equals(CYBERVISION_OPENMP,true){
		QMAKE_CXXFLAGS += -fopenmp
		LIBS += -lgomp -lpthread
	}

	equals(CYBERVISION_SSE, true): QMAKE_CXXFLAGS += -msse3

	equals(CYBERVISION_OPENCL, true){
		LIBS += -lOpenCL
	}
	QMAKE_CXXFLAGS_DEBUG += -Wa,-mbig-obj
}

unix {
	equals(CYBERVISION_OPENMP,true){
		QMAKE_CXXFLAGS += -fopenmp
		LIBS += -lgomp -lpthread
	}

	equals(CYBERVISION_SSE, true): QMAKE_CXXFLAGS += -msse3

	equals(CYBERVISION_OPENCL, true){
		INCLUDEPATH += /opt/AMDAPP/include
		LIBS += -L/opt/AMDAPP/lib/x86_64 -lOpenCL -lGLU
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
