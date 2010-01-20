# -------------------------------------------------
# Project created by QtCreator 2009-10-25T19:45:46
# -------------------------------------------------
QT += opengl
TARGET = 3dvision
TEMPLATE = app
SOURCES += main.cpp \
    UI/mainwindow.cpp \
    SIFT/process.cpp \
    Reconstruction/svd.cpp \
    Reconstruction/reconstructor.cpp \
    Reconstruction/options.cpp \
    UI/processthread.cpp \
    SIFT/siftgateway.cpp \
    SIFT/libsiftfast.cpp
HEADERS += UI/mainwindow.h \
    SIFT/process.h \
    Reconstruction/svd.h \
    Reconstruction/reconstructor.h \
    Reconstruction/options.h \
    UI/processthread.h \
    SIFT/siftgateway.h \
    SIFT/siftfast.h
FORMS += UI/mainwindow.ui
OTHER_FILES += SIFT/optimization_flags.txt
