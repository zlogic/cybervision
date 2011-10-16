TEMPLATE = subdirs

CONFIG += ordered

DEPENDPATH += libsiftfast

OTHER_FILES += \
    cybervision-options.pri

#TRANSLATIONS = cybervision-app/UI/translations/cybervision-app_ru.ts

SUBDIRS = \
	libsiftfast \
	cybervision-app

cybervision-app.depends = libsiftfast
