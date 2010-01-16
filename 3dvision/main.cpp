#include <QtGui/QApplication>
#include "UI/mainwindow.h"

#include "Utils/svd.h"

int main(int argc, char *argv[])
{
	test_svd();
    QApplication a(argc, argv);
    MainWindow w;
	w.show();
    return a.exec();
}
