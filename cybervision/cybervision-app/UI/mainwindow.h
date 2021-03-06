#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtWidgets/QMainWindow>
#include <QTextStream>
#include <QDoubleValidator>

#ifdef CYBERVISION_DEMO
#include <QTimer>
#endif

#include <UI/processthread.h>
#include <UI/crosssectionwindow.h>
#include <UI/aboutwindow.h>

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
	CrossSectionWindow crossSectionWindow;
	AboutWindow aboutWindow;

	ProcessThread thread;

	QDoubleValidator scaleXYValidator,scaleZValidator,angleValidator;

	QString startPath,image1Path,image2Path;

	QString inputImageFilter;

#ifdef CYBERVISION_DEMO
	//True if user is running reconstruction for the first time
	bool demoReconstructionAllowed;
	//Timer to quit after a timeout
	QTimer demoTimer;
	//Timer minutes
	int demoTimerMinutes;
#endif

	//Updates widgets enabled/disabled/visible status
	void updateWidgetStatus();

	//Loads debugging data from file
	void loadDebugPreferences();

	//Updates the surface stats
	void updateSurfaceStats(int lineId=-1);

#ifdef CYBERVISION_DEMO
	//Shows the "Demo" warning
	void showDemoWarning(QString specificWarning=QString()) const;
#endif
private slots:
	//Slots for receiving messages from process thread
	void processStarted();
	void processUpdated(QString logMessage,QString statusBarText=QString());
	void processStopped(QString resultText,cybervision::Surface);

	//Slots for receiving messages from OpenGL viewport
	void viewerSelectedPointUpdated(QVector3D);
	void viewerCrosssectionLineChanged(QVector3D start,QVector3D end,int lineId);

	//Slots for receiving messages from cross-section viewer
	void crosssectionClosed();

#ifdef CYBERVISION_DEMO
	//Timer expired in demo version, application should quit
	void demoTimerExpired() const;
#endif

	//UI slots
	void on_openImage1_clicked();
	void on_openImage2_clicked();
	void on_addImageButton_clicked();
	void on_deleteImageButton_clicked();
	void on_useForImage1Button_clicked();
	void on_useForImage2Button_clicked();
	void on_imageList_itemSelectionChanged();
	void on_angleEdit_textChanged(const QString &arg1);
	void on_actionShow_log_triggered(bool checked);
	void on_actionShow_controls_triggered(bool checked);
	void on_actionShow_cross_section_window_triggered(bool checked);
	void on_actionAbout_triggered();
	void on_logDockWidget_visibilityChanged(bool visible);
	void on_controlsDockWidget_visibilityChanged(bool visible);
	void on_startProcessButton_clicked();
	void on_saveButton_clicked();
	void on_loadSurfaceButton_clicked();
	void on_moveToolButton_toggled(bool checked);
	void on_rotateToolButton_toggled(bool checked);
	void on_gridToolButton_toggled(bool checked);
	void on_texture1ToolButton_clicked();
	void on_texture2ToolButton_clicked();
	void on_textureNoneToolButton_clicked();
	void on_crosssectionButtonPrimary_clicked(bool checked);
	void on_crosssectionButtonSecondary_clicked(bool checked);
};

#endif // MAINWINDOW_H
