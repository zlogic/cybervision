#include <QtWidgets/QApplication>
#include "UI/mainwindow.h"
#include <QTranslator>

int main(int argc, char *argv[]){
	//Create the app
	QApplication app(argc, argv);

	//Translate the app, if possible
	QString locale = QLocale::system().name();
	QTranslator translator;
	translator.load(QString("cybervision-app_") + locale);
	app.installTranslator(&translator);

	//Run the main window
	MainWindow w;
	w.show();
	return app.exec();
}
