#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui/QMainWindow>
#include <QGraphicsScene>
#include <QThread>

namespace Ui
{
    class MainWindow;
}


class MainWindow;
class ProcessThread :public QThread{
	Q_OBJECT
protected:
	MainWindow *mw;
	QStringList image_filenames;
public:
	ProcessThread();
	void setUi(MainWindow *nw=NULL);

	void extract(QStringList image_filenames);
	void run();//the main thread loop
signals:
	void processStarted(QString statusBarText);
	void processStopped(QStringList image_filenames,QImage result);
};


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

	QGraphicsScene scene;
	ProcessThread thread;
private slots:
	void on_pushButton_clicked();
	void on_pushButton_2_clicked();

	void processStarted(QString statusBarText);
	void processStopped(QStringList image_filenames,QImage result);
};

#endif // MAINWINDOW_H
