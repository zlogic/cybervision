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
    Reconstruction/sculptor.cpp
HEADERS += UI/mainwindow.h \
    Reconstruction/svd.h \
    Reconstruction/reconstructor.h \
    Reconstruction/options.h \
    UI/processthread.h \
    SIFT/siftgateway.h \
    SIFT/siftfast.h \
    UI/cybervisionviewer.h \
    Reconstruction/sculptor.h
FORMS += UI/mainwindow.ui
OTHER_FILES += SIFT/optimization_flags.txt
