#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui/QMainWindow>
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

	ProcessThread thread;

	QDoubleValidator scaleXYValidator,scaleZValidator,angleValidator;

	QString startPath;

	bool inspectorOk;

	//Updates widgets enabled/disabled/visible status
	void updateWidgetStatus();

	//Loads debugging data from file
	void loadDebugPreferences();

	//Updates the surface stats
	void updateSurfaceStats();
private slots:
	//Slots for receiving messages from process thread
	void processStarted();
	void processUpdated(QString logMessage,QString statusBarText=QString());
	void processStopped(QString resultText,cybervision::Surface);

	//Slots for receiving messages from OpenGL viewport
	void viewerSelectedPointUpdated(QVector3D);
	void viewerCrosssectionLineChanged(QVector3D start,QVector3D end);

	//UI slots
	void on_addImageButton_clicked();
	void on_deleteImageButton_clicked();
	void on_imageList_itemSelectionChanged();
	void on_actionShow_log_triggered(bool checked);
	void on_logDockWidget_visibilityChanged(bool visible);
	void on_actionShow_statistics_triggered(bool checked);
	void on_inspectorDockWidget_visibilityChanged(bool visible);
	void on_saveButton_clicked();
	void on_loadSurfaceButton_clicked();
	void on_startProcessButton_clicked();
	void on_moveToolButton_toggled(bool checked);
	void on_rotateToolButton_toggled(bool checked);
	void on_gridToolButton_toggled(bool checked);
	void on_texture1ToolButton_clicked();
	void on_texture2ToolButton_clicked();
	void on_textureNoneToolButton_clicked();
	void on_crosssectionButton_clicked(bool checked);
	void on_crosssectionPSpinBox_valueChanged(int arg1);
};

#endif // MAINWINDOW_H
