#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui/QMainWindow>
#include <QGraphicsScene>
#include <QTextStream>
#include <QDoubleValidator>

#include <UI/processthread.h>

namespace Ui
{
    class MainWindow;
}


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

	QDoubleValidator scaleXYValidator,scaleZValidator,angleValidator;

	//Updates widgets enabled/disabled/visible status
	void updateWidgetStatus();

	//Loads debugging data from file
	void loadDebugPreferences();

private slots:
 void on_addImageButton_clicked();
 void on_deleteImageButton_clicked();
 void on_imageList_itemSelectionChanged();
	void on_actionShowlog_toggled(bool);
	void on_logDockWidget_visibilityChanged(bool visible);
	void on_saveButton_clicked();
	void on_startProcessButton_clicked();
	void on_moveToolButton_toggled(bool checked);
	void on_rotateToolButton_toggled(bool checked);
	void on_gridToolButton_toggled(bool checked);

	//Slots for receiving messages from process thread
	void processStarted();
	void processUpdated(QString logMessage,QString statusBarText=QString());
	void processStopped(QString resultText,cybervision::Surface);
	void on_texture1ToolButton_clicked();
	void on_texture2ToolButton_clicked();
	void on_textureNoneToolButton_clicked();
};

#endif // MAINWINDOW_H
