#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>

#ifdef CYBERVISION_DEMO
#include <QMessageBox>
#endif

#include <limits>

#include <Reconstruction/crosssection.h>
#include <Reconstruction/options.h>

MainWindow::MainWindow(QWidget *parent): QMainWindow(parent), ui(new Ui::MainWindow), crossSectionWindow(this){
	ui->setupUi(this);
	thread.setUi(this);
	updateWidgetStatus();
	loadDebugPreferences();
	initViewport();

	connect(&surfaceViewport, SIGNAL(selectedPointUpdated(QVector3D)),this, SLOT(viewerSelectedPointUpdated(QVector3D)),Qt::AutoConnection);
	connect(&surfaceViewport, SIGNAL(crossSectionLineChanged(QVector3D,QVector3D,int)),this, SLOT(viewerCrosssectionLineChanged(QVector3D,QVector3D,int)),Qt::AutoConnection);
	connect(&crossSectionWindow, SIGNAL(closed()),this, SLOT(crosssectionClosed()),Qt::AutoConnection);
	connect(ui->openImage1, SIGNAL(clicked()),this,SLOT(selectImage1()));
	connect(ui->openImage2, SIGNAL(clicked()),this,SLOT(selectImage2()));
	connect(ui->addImageButton, SIGNAL(clicked()),this,SLOT(addImages()));
	connect(ui->deleteImageButton, SIGNAL(clicked()),this,SLOT(deleteImage()));
	connect(ui->useForImage1Button, SIGNAL(clicked()),this,SLOT(useForImage1()));
	connect(ui->useForImage2Button, SIGNAL(clicked()),this,SLOT(useForImage2()));
	connect(ui->imageList, SIGNAL(itemSelectionChanged()),this,SLOT(updateWidgetStatus()));
	connect(ui->angleEdit, SIGNAL(textChanged(QString)),this,SLOT(updateWidgetStatus()));
	connect(ui->actionShow_log, SIGNAL(triggered(bool)),ui->logDockWidget,SLOT(setVisible(bool)));
	connect(ui->actionShow_controls, SIGNAL(triggered(bool)),ui->controlsDockWidget,SLOT(setVisible(bool)));
	connect(ui->actionShow_cross_section_window, SIGNAL(triggered(bool)),&crossSectionWindow,SLOT(setVisible(bool)));
	connect(ui->logDockWidget, SIGNAL(visibilityChanged(bool)),ui->actionShow_log,SLOT(setChecked(bool)));
	connect(ui->controlsDockWidget, SIGNAL(visibilityChanged(bool)),ui->actionShow_controls,SLOT(setChecked(bool)));
	connect(ui->actionAbout, SIGNAL(triggered(bool)),this,SLOT(showAboutWindow()));
	connect(ui->startProcessButton, SIGNAL(clicked()),this,SLOT(startReconstruction()));
	connect(ui->saveButton, SIGNAL(clicked()),this,SLOT(saveResult()));
	connect(ui->loadSurfaceButton, SIGNAL(clicked()),this,SLOT(loadSurface()));
	connect(ui->moveToolButton, SIGNAL(toggled(bool)),this,SLOT(updateMouseMode()));
	connect(ui->rotateToolButton, SIGNAL(toggled(bool)),this,SLOT(updateMouseMode()));
	connect(ui->gridToolButton, SIGNAL(toggled(bool)),this,SLOT(setShowGrid(bool)));
	connect(ui->texture1ToolButton, SIGNAL(clicked()),this,SLOT(updateTextureMode()));
	connect(ui->texture2ToolButton, SIGNAL(clicked()),this,SLOT(updateTextureMode()));
	connect(ui->textureNoneToolButton, SIGNAL(clicked()),this,SLOT(updateTextureMode()));
	connect(ui->crosssectionButtonPrimary, SIGNAL(clicked(bool)),this,SLOT(primaryCrossSectionClicked(bool)));
	connect(ui->crosssectionButtonSecondary, SIGNAL(clicked(bool)),this,SLOT(secondaryCrossSectionClicked(bool)));

	inputImageFilter= tr("Images") + "(*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;"+tr("All files")+"(*.*)";
#ifdef CYBERVISION_DEMO
	demoTimerMinutes= 20;
	demoReconstructionAllowed= true;

	showDemoWarning();
	setWindowTitle(tr("Cybervision (Demo version)"));
	setMaximumSize(1000,600);
	demoTimer.setSingleShot(true);
	demoTimer.start(1000*60*demoTimerMinutes);
	QObject::connect(&demoTimer,SIGNAL(timeout()),this,SLOT(demoTimerExpired()),Qt::AutoConnection);
#endif
}

MainWindow::~MainWindow(){
	delete ui;
}

void MainWindow::initViewport(){
	surfaceViewport.setFlags(Qt::Widget);
	QWidget *viewportContainer= QWidget::createWindowContainer(&surfaceViewport, this);
	ui->viewportLayout->replaceWidget(ui->viewportPlaceholder,viewportContainer);
	viewportContainer->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);
}

void MainWindow::updateWidgetStatus(){
	bool surfaceIsOK;
	{
		QMutexLocker lock(&surfaceViewport.getSurfaceMutex());
		surfaceIsOK= surfaceViewport.getSurface3D().isOk();
	}

	//Image loading buttons
	if(image1Path.isEmpty() || image1Path.isNull()){
		ui->image1NameLabel->setText(tr("<b>No image selected</b>"));
		ui->image1NameLabel->setToolTip("");
	}else{
		ui->image1NameLabel->setText(QFileInfo(image1Path).fileName());
		ui->image1NameLabel->setToolTip(image1Path);
	}
	if(image2Path.isEmpty() || image2Path.isNull()){
		ui->image2NameLabel->setText(tr("<b>No image selected</b>"));
		ui->image2NameLabel->setToolTip("");
	}else{
		ui->image2NameLabel->setText(QFileInfo(image2Path).fileName());
		ui->image2NameLabel->setToolTip(image2Path);
	}

	//Angle error message
	bool angleIsOk= false;
	angleIsOk= ui->angleEdit->text().toDouble(&angleIsOk)!=0.0 && angleIsOk;
	ui->angleHintLabel->setVisible(!angleIsOk);

	//Start process
	if(!image1Path.isEmpty() && !image1Path.isNull() && !image2Path.isEmpty() && !image2Path.isNull())
		ui->startProcessButton->setEnabled(angleIsOk);
	else
		ui->startProcessButton->setEnabled(false);

	ui->deleteImageButton->setEnabled(!ui->imageList->selectedItems().empty());
	ui->useForImage1Button->setEnabled(!ui->imageList->selectedItems().empty());
	ui->useForImage2Button->setEnabled(!ui->imageList->selectedItems().empty());
	ui->saveButton->setEnabled(surfaceIsOK);
	ui->crosssectionButtonPrimary->setEnabled(surfaceIsOK);
	ui->crosssectionButtonSecondary->setEnabled(surfaceIsOK);

	scaleXYValidator.setBottom(0.0);
	scaleZValidator.setBottom(0.0);
	angleValidator.setRange(-360.0,360.0,1000);
	scaleXYValidator.setLocale(QLocale::C);
	scaleZValidator.setLocale(QLocale::C);
	angleValidator.setLocale(QLocale::C);
	ui->scaleXYEdit->setValidator(&scaleXYValidator);
	ui->scaleZEdit->setValidator(&scaleZValidator);
	ui->angleEdit->setValidator(&angleValidator);

	viewerSelectedPointUpdated(surfaceViewport.getSelectedPoint());

	updateSurfaceStats();
}


void MainWindow::updateSurfaceStats(int lineId){
	QMutexLocker lock(&surfaceViewport.getSurfaceMutex());
	qreal minDepth= surfaceViewport.getSurface3D().getMinDepth();
	qreal maxDepth= surfaceViewport.getSurface3D().getMaxDepth();
	qreal medianDepth= surfaceViewport.getSurface3D().getMedianDepth();
	qreal baseDepth= surfaceViewport.getSurface3D().getBaseDepth();
	if(surfaceViewport.getSurface3D().isOk())
		ui->statisticsLabel->setText(QString(tr("Depth range: %1 \xC2\xB5m\nBase depth: %2 \xC2\xB5m\nMedian depth: %3 \xC2\xB5m"))
									 .arg((maxDepth-minDepth)*cybervision::Options::TextUnitScale)
									 .arg(baseDepth*cybervision::Options::TextUnitScale)
									 .arg(medianDepth*cybervision::Options::TextUnitScale));
	else
		ui->statisticsLabel->setText(tr("No surface available"));

	QPair<QVector3D,QVector3D> crossSectionLine= surfaceViewport.getCrossSectionLine(lineId);
	if(lineId<0)
		return;

	//Compute surface roughness, if possible
	cybervision::CrossSection crossSection;
	crossSection.computeCrossSection(surfaceViewport.getSurface3D(),crossSectionLine.first,crossSectionLine.second);
	crossSectionWindow.updateCrossSection(crossSection,lineId);
	if(crossSection.isOk())
		crossSectionWindow.show();
}

#ifdef CYBERVISION_DEMO
void MainWindow::showDemoWarning(QString specificWarning)const{
	QMessageBox msgBox;
	msgBox.setModal(true);
	msgBox.setWindowTitle(tr("Demo version warning"));
	msgBox.setText(tr("This is a functionally limited demo version of Cybervision.")
				   +((specificWarning.isNull()||specificWarning.isEmpty())?"":(QString("\n")+specificWarning)));
	msgBox.setIcon((specificWarning.isNull()||specificWarning.isEmpty())?QMessageBox::Warning:QMessageBox::Critical);
	msgBox.setInformativeText(QString(tr("Upgrade to the full version to gain access to the following features:\n"
										 "- Saving the reconstruction result\n"
										 "- Reconstructing multiple surface without restarting\n"
										 "- Resizing or maximizing windows for a high resolution view\n"
										 "- Using the application for more than %1 minutes")).arg(demoTimerMinutes));
	msgBox.setStandardButtons(QMessageBox::Ok);
	msgBox.setDefaultButton(QMessageBox::Ok);
	msgBox.exec();
}
#endif

void MainWindow::loadDebugPreferences(){
	QStringList arguments= qApp->arguments();
	QString debugParam="-debugfile=";
	QString debugFileName;
	for(QStringList::const_iterator it= arguments.constBegin();it!=arguments.constEnd();it++){
		if(it->startsWith(debugParam,Qt::CaseInsensitive)){
			debugFileName= it->mid(debugParam.length());
			break;
		}
	}

	if(debugFileName.isNull())
		return;

	QFile debugFile(debugFileName);
	if(debugFile.exists()){
		debugFile.open(QFile::ReadOnly);
		QTextStream stream(&debugFile);

		while(!stream.atEnd()){
			QString line= stream.readLine();
			if(!line.isNull() && !line.isEmpty()){
				QRegExp fileRegexp("^file\\s+(.*)$",Qt::CaseInsensitive);
				QRegExp scaleXYRegexp("^ScaleXY\\s+([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)$",Qt::CaseInsensitive);
				QRegExp scaleZRegexp("^ScaleZ\\s+([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)$",Qt::CaseInsensitive);
				QRegExp angleRegexp("^Angle\\s+([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)$",Qt::CaseInsensitive);
				if(fileRegexp.exactMatch(line) && fileRegexp.capturedTexts().size()>=2){
					QString fileName= fileRegexp.capturedTexts().at(1);
					QString name= QFileInfo(fileName).fileName();
					QListWidgetItem* newItem= new QListWidgetItem(name);
					newItem->setData(32,QDir::toNativeSeparators(fileName));
					ui->imageList->addItem(newItem);

					startPath= QFileInfo(fileName).canonicalPath();
				}else if(scaleXYRegexp.exactMatch(line) && scaleXYRegexp.capturedTexts().size()>=2){
					ui->scaleXYEdit->setText(scaleXYRegexp.capturedTexts().at(1));
				}else if(scaleZRegexp.exactMatch(line) && scaleZRegexp.capturedTexts().size()>=2){
					ui->scaleZEdit->setText(scaleZRegexp.capturedTexts().at(1));
				}else if(angleRegexp.exactMatch(line) && angleRegexp.capturedTexts().size()>=2){
					ui->angleEdit->setText(angleRegexp.capturedTexts().at(1));
				}
			}
		}
		debugFile.close();
		updateWidgetStatus();
	}
}

void MainWindow::processStarted(){
	ui->startProcessButton->setEnabled(false);
	ui->saveButton->setEnabled(false);
	ui->loadSurfaceButton->setEnabled(false);
	ui->logTextEdit->clear();
	ui->logTextEdit->setCurrentCharFormat(QTextCharFormat());
}


void MainWindow::processUpdated(QString logMessage,QString statusBarText){
	if(!logMessage.isNull())
		ui->logTextEdit->appendPlainText(logMessage+"\n");
	if(!statusBarText.isNull())
		ui->statusBar->showMessage(statusBarText);

	if(logMessage.isNull() && !statusBarText.isNull())
		ui->logTextEdit->appendPlainText(statusBarText+"\n");
}

void MainWindow::processStopped(QString resultText,cybervision::Surface surface){
	ui->statusBar->clearMessage();
	ui->startProcessButton->setEnabled(true);
	if(!resultText.isNull() && !resultText.isEmpty()){
		ui->saveButton->setEnabled(true);
		ui->logTextEdit->appendHtml("<b>"+resultText+"</b>");
	}
	ui->loadSurfaceButton->setEnabled(true);

	surfaceViewport.setSurface3D(surface);
	updateWidgetStatus();
}

void MainWindow::viewerSelectedPointUpdated(QVector3D point){
	if(point.x()==std::numeric_limits<qreal>::infinity() || point.y()==std::numeric_limits<qreal>::infinity() || point.z()==std::numeric_limits<qreal>::infinity())
		ui->pointCoordinatesLabel->setText(tr("No point selected"));
	else
		ui->pointCoordinatesLabel->setText(QString(tr("x: %1 \xC2\xB5m\ny: %2 \xC2\xB5m\nz: %3 \xC2\xB5m"))
										   .arg(point.x()*cybervision::Options::TextUnitScale)
										   .arg(point.y()*cybervision::Options::TextUnitScale)
										   .arg(point.z()*cybervision::Options::TextUnitScale));
}

void MainWindow::viewerCrosssectionLineChanged(QVector3D start, QVector3D end,int lineId){
	if(lineId==0)
		ui->crosssectionButtonPrimary->setChecked(false);
	else if(lineId==1)
		ui->crosssectionButtonSecondary->setChecked(false);
	updateSurfaceStats(lineId);
}


void MainWindow::crosssectionClosed(){
	ui->actionShow_cross_section_window->setChecked(false);
}


void MainWindow::startReconstruction(){
	QStringList filenames;

	double scaleXY= ui->scaleXYEdit->text().toDouble();
	double scaleZ= ui->scaleZEdit->text().toDouble();
	double angle= ui->angleEdit->text().toDouble();

	filenames<<image1Path<<image2Path;
#ifdef CYBERVISION_DEMO
	if(!demoReconstructionAllowed){
		showDemoWarning(tr("Please restart the application to reconstruct another surface."));
		return;
	}
	demoReconstructionAllowed= false;
#endif
	thread.reconstruct3DShape(filenames,scaleXY,scaleZ,angle,ui->preferScaleFromMetadata->isChecked());
}

void MainWindow::saveResult(){
	QStringList formats;
	formats<<tr("Surface points")+ " (*.txt)";
	formats<<tr("Surface polygons")+ " (*.txt)";
	formats<<tr("PNG image")+ " (*.png)";
	formats<<tr("SceneJS model")+ " (*.js)";
	formats<<tr("COLLADA model")+ " (*.dae)";
	formats<<tr("Cybervision surface")+ " (*.cvs)";
	QString filter;
	for(QStringList::const_iterator it=formats.constBegin();it!=formats.constEnd();it++)
		filter.append(*it+";;");
	QString selectedFilter;
	QString fileName = QFileDialog::getSaveFileName(this,tr("Save the surface"),startPath,filter,&selectedFilter);
#ifdef CYBERVISION_DEMO
	showDemoWarning(tr("Save functionality is disabled."));
#else
	if(!fileName.isNull()){
		QMutexLocker lock(&surfaceViewport.getSurfaceMutex());

		QFileInfo fileInfo(fileName);
		if(selectedFilter==formats[0]){
			if(fileInfo.suffix()!="txt")
				fileName.append(".txt");
			surfaceViewport.getSurface3D().savePoints(fileName);
		}else if(selectedFilter==formats[1]){
			if(fileInfo.suffix()!="txt")
				fileName.append(".txt");
			surfaceViewport.getSurface3D().savePolygons(fileName);
		}else if(selectedFilter==formats[2]){
			if(fileInfo.suffix()!="png")
				fileName.append(".png");
			QPixmap screenshot= surfaceViewport.getScreenshot();
			screenshot.save(fileName,"png");
		}else if(selectedFilter==formats[3]){
			if(fileInfo.suffix()!="js")
				fileName.append(".js");
			surfaceViewport.getSurface3D().saveSceneJS(fileName);
		}else if(selectedFilter==formats[4]){
			if(fileInfo.suffix()!="dae")
				fileName.append(".dae");
			surfaceViewport.getSurface3D().saveCollada(fileName);
		}else if(selectedFilter==formats[5]){
			if(fileInfo.suffix()!="cvs")
				fileName.append(".cvs");
			surfaceViewport.getSurface3D().saveCybervision(fileName);
		}else{
			ui->statusBar->showMessage(tr("Bad save format selected"));
			ui->logTextEdit->appendHtml(QString(tr("<b>Bad save format selected:</b> %1")).arg(selectedFilter));
		}
		startPath= fileInfo.canonicalPath();
	}
#endif
}

void MainWindow::loadSurface(){
	QString filter= tr("Cybervision surface") + "(*.cvs);;"+tr("All files")+"(*.*)";
	QString filename = QFileDialog::getOpenFileName(this,tr("Select surface to load"),startPath,filter,0);
	if(!filename.isNull()){
		cybervision::Surface surface= cybervision::Surface::fromFile(filename);
		if(!surface.isOk())
			ui->logTextEdit->appendHtml("<b>"+QString(tr("Error loading surface from %1")).arg(QDir::toNativeSeparators(filename))+"</b>");

		surfaceViewport.setSurface3D(surface);
		updateWidgetStatus();

		startPath= QFileInfo(filename).canonicalPath();
	}
}

void MainWindow::showAboutWindow(){
	aboutWindow.setVisible(true);
}

void MainWindow::selectImage1(){
	QString filename = QFileDialog::getOpenFileName(this,tr("Select image 1"),startPath,inputImageFilter,0);
	if(filename.isNull()){
		image1Path= "";
	}else{
		filename= QDir::toNativeSeparators(filename);
		image1Path= filename;
		startPath= QFileInfo(filename).canonicalPath();

		if(ui->imageList->findItems(QFileInfo(filename).fileName(),Qt::MatchExactly).empty()){
			QListWidgetItem* newItem= new QListWidgetItem(QFileInfo(filename).fileName());
			newItem->setData(32,QDir::toNativeSeparators(filename));
			ui->imageList->addItem(newItem);
		}
	}
	updateWidgetStatus();
}

void MainWindow::selectImage2(){
	QString filename = QFileDialog::getOpenFileName(this,tr("Select image 2"),startPath,inputImageFilter,0);
	if(filename.isNull()){
		image2Path= "";
	}else{
		filename= QDir::toNativeSeparators(filename);
		image2Path= filename;
		startPath= QFileInfo(filename).canonicalPath();

		if(ui->imageList->findItems(QFileInfo(filename).fileName(),Qt::MatchExactly).empty()){
			QListWidgetItem* newItem= new QListWidgetItem(QFileInfo(filename).fileName());
			newItem->setData(32,QDir::toNativeSeparators(filename));
			ui->imageList->addItem(newItem);
		}
	}
	updateWidgetStatus();
}

void MainWindow::addImages(){
	QStringList filenames = QFileDialog::getOpenFileNames(this,tr("Select images to add"),startPath,inputImageFilter,0);
	for(QStringList::const_iterator it=filenames.constBegin();it!=filenames.constEnd();it++){
		QString name= QFileInfo(*it).fileName();
		QListWidgetItem* newItem= new QListWidgetItem(name);
		newItem->setData(32,QDir::toNativeSeparators(*it));
		ui->imageList->addItem(newItem);

		startPath= QFileInfo(*it).canonicalPath();
	}

	updateWidgetStatus();
}

void MainWindow::deleteImage(){
	QList<QListWidgetItem*> selection= ui->imageList->selectedItems();
	for(QList<QListWidgetItem*>::const_iterator i=selection.constBegin();i!=selection.constEnd();i++){
		int row= ui->imageList->row(*i);
		if(row>=0){
			QListWidgetItem* deletedItem= ui->imageList->takeItem(row);
			if(deletedItem)
				delete deletedItem;
		}
	}
}

void MainWindow::useForImage1(){
	QList<QListWidgetItem*> selectedItems= ui->imageList->selectedItems();
	if(selectedItems.size()>0)
		image1Path= selectedItems.first()->data(32).toString();
	updateWidgetStatus();
}

void MainWindow::useForImage2(){
	QList<QListWidgetItem*> selectedItems= ui->imageList->selectedItems();
	if(selectedItems.size()>0)
		image2Path= selectedItems.first()->data(32).toString();
	updateWidgetStatus();
}

void MainWindow::updateMouseMode(){
	if(ui->moveToolButton->isChecked())
		surfaceViewport.setMouseMode(CybervisionViewer::MOUSE_PANNING);
	else if(ui->rotateToolButton->isChecked())
		surfaceViewport.setMouseMode(CybervisionViewer::MOUSE_ROTATION);
}

void MainWindow::setShowGrid(bool checked){
	surfaceViewport.setShowGrid(checked);
}

void MainWindow::updateTextureMode(){
	if(ui->texture1ToolButton->isChecked())
		surfaceViewport.setTextureMode(CybervisionViewer::TEXTURE_1);
	else if(ui->texture2ToolButton->isChecked())
		surfaceViewport.setTextureMode(CybervisionViewer::TEXTURE_2);
	else if(ui->textureNoneToolButton->isChecked())
		surfaceViewport.setTextureMode(CybervisionViewer::TEXTURE_NONE);
}

void MainWindow::primaryCrossSectionClicked(bool checked){
	if(checked){
		ui->crosssectionButtonSecondary->setChecked(false);
		QMutexLocker lock(&surfaceViewport.getSurfaceMutex());
		if(surfaceViewport.getSurface3D().isOk())
			surfaceViewport.setDrawCrossSectionLine(checked?0:-1);
		else{
			ui->crosssectionButtonPrimary->setChecked(false);
			ui->crosssectionButtonPrimary->setEnabled(false);
		}
	}else
		surfaceViewport.setDrawCrossSectionLine(checked?0:-1);
}

void MainWindow::secondaryCrossSectionClicked(bool checked){
	if(checked){
		ui->crosssectionButtonPrimary->setChecked(false);
		QMutexLocker lock(&surfaceViewport.getSurfaceMutex());
		if(surfaceViewport.getSurface3D().isOk())
			surfaceViewport.setDrawCrossSectionLine(checked?1:-1);
		else{
			ui->crosssectionButtonSecondary->setChecked(false);
			ui->crosssectionButtonSecondary->setEnabled(false);
		}
	}else
		surfaceViewport.setDrawCrossSectionLine(checked?1:-1);
}
#ifdef CYBERVISION_DEMO
void MainWindow::demoTimerExpired() const{
	showDemoWarning(QString(tr("The demo has run for %1 minutes and will now quit.")).arg(demoTimerMinutes));
	exit(0);
}
#endif
