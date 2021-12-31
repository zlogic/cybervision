include( ../cybervision-options.pri )

QT += core gui widgets opengl
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
	UI/icons/arrow-circle.png \
	UI/icons/application-task.png \
	UI/icons/cybervision-large.png \
	UI/icons/pictures-stack.png \
	UI/icons/magnifier--pencil.png \
	UI/icons/information.png

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
	LIBS += -lglu32 -lopengl32
	equals(CYBERVISION_OPENMP,true){
		QMAKE_CXXFLAGS += -fopenmp
		LIBS += -lgomp -lpthread
	}

	equals(CYBERVISION_SSE, true): QMAKE_CXXFLAGS += -msse3

	equals(CYBERVISION_OPENCL, true){
		LIBS += -lOpenCL
	}
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
