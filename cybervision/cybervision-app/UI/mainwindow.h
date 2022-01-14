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
#include <UI/cybervisionviewer.h>

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
	CybervisionViewer surfaceViewport;
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

	//Initialized the 3D viewport
	void initViewport();

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

	//Slots for receiving messages from 3D viewer viewport
	void viewerSelectedPointUpdated(QVector3D);
	void viewerCrosssectionLineChanged(QVector3D start,QVector3D end,int lineId);

	//Slots for receiving messages from cross-section viewer
	void crosssectionClosed();

#ifdef CYBERVISION_DEMO
	//Timer expired in demo version, application should quit
	void demoTimerExpired() const;
#endif

	//Updates widgets enabled/disabled/visible status
	void updateWidgetStatus();

	//UI slots
	void selectImage1();
	void selectImage2();
	void addImages();
	void deleteImage();
	void useForImage1();
	void useForImage2();
	void showAboutWindow();
	void startReconstruction();
	void saveResult();
	void loadSurface();
	void updateMouseMode();
	void setShowGrid(bool checked);
	void updateTextureMode();
	void primaryCrossSectionClicked(bool checked);
	void secondaryCrossSectionClicked(bool checked);
};

#endif // MAINWINDOW_H
