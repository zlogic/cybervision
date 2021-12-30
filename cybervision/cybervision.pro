TEMPLATE = subdirs

CONFIG += ordered

DEPENDPATH += libsiftfast

OTHER_FILES += \
	cybervision-options.pri

SUBDIRS = \
	libsiftfast \
	cybervision-app

cybervision-app.depends = libsiftfast
