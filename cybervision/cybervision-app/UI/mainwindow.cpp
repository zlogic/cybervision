#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>

#include <limits>

#include "Reconstruction/inspector.h"

MainWindow::MainWindow(QWidget *parent)	: QMainWindow(parent), ui(new Ui::MainWindow){
	inspectorOk= false;
	ui->setupUi(this);
	thread.setUi(this);
	updateWidgetStatus();
	loadDebugPreferences();

	QObject::connect(ui->openGLViewport, SIGNAL(selectedPointUpdated(QVector3D)),this, SLOT(viewerSelectedPointUpdated(QVector3D)),Qt::AutoConnection);
	QObject::connect(ui->openGLViewport, SIGNAL(crossSectionLineChanged(QVector3D,QVector3D)),this, SLOT(viewerCrosssectionLineChanged(QVector3D,QVector3D)),Qt::AutoConnection);
}

MainWindow::~MainWindow(){
	delete ui;
}

void MainWindow::updateWidgetStatus(){
	bool surfaceIsOK;
	{
		QMutexLocker lock(&ui->openGLViewport->getSurfaceMutex());
		surfaceIsOK= ui->openGLViewport->getSurface3D().isOk();
	}

	if(ui->imageList->count()==0){
		ui->selectionHintLabel->setVisible(false);
		ui->deleteImageButton->setEnabled(false);
		ui->startProcessButton->setEnabled(false);
	}else if(ui->imageList->count()>1 && ui->imageList->selectedItems().count()!=2){
		ui->selectionHintLabel->setVisible(true);
		ui->startProcessButton->setEnabled(false);
	}else {
		ui->selectionHintLabel->setVisible(false);
		ui->startProcessButton->setEnabled(true);
	}

	ui->deleteImageButton->setEnabled(!ui->imageList->selectedItems().empty());
	ui->saveButton->setEnabled(surfaceIsOK);
	ui->crosssectionButton->setEnabled(surfaceIsOK);
	ui->crosssectionPSpinBox->setVisible(inspectorOk);
	ui->crosssectionRoughnessParametersLabel->setVisible(inspectorOk);

	scaleXYValidator.setBottom(0.0);
	scaleZValidator.setBottom(0.0);
	angleValidator.setRange(-360.0,360.0,1000);
	ui->scaleXYEdit->setValidator(&scaleXYValidator);
	ui->scaleZEdit->setValidator(&scaleZValidator);
	ui->angleEdit->setValidator(&angleValidator);

	viewerSelectedPointUpdated(ui->openGLViewport->getSelectedPoint());

	updateSurfaceStats();
}


void MainWindow::updateSurfaceStats(){
	QMutexLocker lock(&ui->openGLViewport->getSurfaceMutex());
	qreal minDepth= ui->openGLViewport->getSurface3D().getMinDepth();
	qreal maxDepth= ui->openGLViewport->getSurface3D().getMaxDepth();
	qreal medianDepth= ui->openGLViewport->getSurface3D().getMedianDepth();
	qreal baseDepth= ui->openGLViewport->getSurface3D().getBaseDepth();
	if(ui->openGLViewport->getSurface3D().isOk())
		ui->statisticsLabel->setText(QString(tr("Depth range: %1 m\nBase depth: %2 m\nMedian depth: %3 m")).arg(maxDepth-minDepth).arg(baseDepth).arg(medianDepth));
	else
		ui->statisticsLabel->setText(tr("No surface available"));

	QPair<QVector3D,QVector3D> crossSectionLine= ui->openGLViewport->getCrossSectionLine();
	inspectorOk= crossSectionLine.first!=crossSectionLine.second;

	//Compute surface roughness, if possible
	cybervision::Inspector inspector(ui->openGLViewport->getSurface3D());
	inspector.updateCrossSection(crossSectionLine.first,crossSectionLine.second,ui->crosssectionPSpinBox->value());

	//Set the cross-section image
	QImage img= inspector.renderCrossSection(ui->crosssectionImage->size());

	if(!inspectorOk){
		img= QImage(0,0);
	}
	ui->crosssectionImage->setPixmap(QPixmap::fromImage(img));

	//Set the roughness parameters
	QString heightParamsString,stepParamsString;
	if(inspectorOk){
		heightParamsString= QString(tr("Ra= %1 m\nRz= %2 m\nRmax= %3 m"))
				.arg(inspector.getRoughnessRa())
				.arg(inspector.getRoughnessRz())
				.arg(inspector.getRoughnessRmax());
		stepParamsString= QString(tr("S= %1 m\nSm= %2 m\ntp= %3"))
				.arg(inspector.getRoughnessS())
				.arg(inspector.getRoughnessSm())
				.arg(inspector.getRoughnessTp());

	}else{
		heightParamsString= "";
		stepParamsString= "";
	}

	ui->crosssectionPSpinBox->setVisible(inspectorOk);
	ui->crosssectionRoughnessParametersLabel->setVisible(inspectorOk);
	ui->crosssectionHeightStatsLabel->setText(heightParamsString);
	ui->crosssectionStepStatsLabel->setText(stepParamsString);
}


void MainWindow::loadDebugPreferences(){
	QStringList arguments= qApp->arguments();
	QString debugParam="-debugfile=";
	QString debugFileName;
	for(QStringList::const_iterator it= arguments.begin();it!=arguments.end();it++){
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
					newItem->setData(32,QDir::convertSeparators(fileName));
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

	ui->openGLViewport->setSurface3D(surface);
	updateWidgetStatus();
}

void MainWindow::viewerSelectedPointUpdated(QVector3D point){
	if(point.x()==std::numeric_limits<qreal>::infinity() || point.y()==std::numeric_limits<qreal>::infinity() || point.z()==std::numeric_limits<qreal>::infinity())
		ui->pointCoordinatesLabel->setText(tr("No point selected"));
	else
		ui->pointCoordinatesLabel->setText(QString("x: %1\ny: %2\nz:%3").arg(point.x()).arg(point.y()).arg(point.z()));
}

void MainWindow::viewerCrosssectionLineChanged(QVector3D start, QVector3D end){
	ui->crosssectionButton->setChecked(false);
	updateSurfaceStats();
}

void MainWindow::on_startProcessButton_clicked(){
	QStringList filenames;

	double scaleXY= ui->scaleXYEdit->text().toDouble();
	double scaleZ= ui->scaleZEdit->text().toDouble();
	double angle= ui->angleEdit->text().toDouble();

	QList<QListWidgetItem*> selectedItems= ui->imageList->selectedItems();
	for(QList<QListWidgetItem*>::const_iterator i=selectedItems.begin();i!=selectedItems.end();i++){
		if(*i)
			filenames<<(*i)->data(32).toString();
	}

	thread.reconstruct3DShape(filenames,scaleXY,scaleZ,angle,ui->actionPrefer_scale_from_metadata->isChecked());
}

void MainWindow::on_saveButton_clicked(){
	QStringList formats;
	formats<<tr("Surface points")+ " (*.txt)";
	formats<<tr("Surface polygons")+ " (*.txt)";
	formats<<tr("PNG image")+ " (*.png)";
	formats<<tr("SceneJS model")+ " (*.js)";
	formats<<tr("COLLADA model")+ " (*.dae)";
	formats<<tr("Cybervision surface")+ " (*.cvs)";
	QString filter;
	for(QStringList::const_iterator it=formats.begin();it!=formats.end();it++)
		filter.append(*it+";;");
	QString selectedFilter;
	QString fileName = QFileDialog::getSaveFileName(this,tr("Save the surface"),startPath,filter,&selectedFilter,0);
	if(!fileName.isNull()){
		QMutexLocker lock(&ui->openGLViewport->getSurfaceMutex());

		QFileInfo fileInfo(fileName);
		if(selectedFilter==formats[0]){
			if(fileInfo.suffix()!="txt")
				fileName.append(".txt");
			ui->openGLViewport->getSurface3D().savePoints(fileName);
		}else if(selectedFilter==formats[1]){
			if(fileInfo.suffix()!="txt")
				fileName.append(".txt");
			ui->openGLViewport->getSurface3D().savePolygons(fileName);
		}else if(selectedFilter==formats[2]){
			if(fileInfo.suffix()!="png")
				fileName.append(".png");
			QImage screenshot= ui->openGLViewport->grabFrameBuffer();
			screenshot.save(fileName,"png");
		}else if(selectedFilter==formats[3]){
			if(fileInfo.suffix()!="js")
				fileName.append(".js");
			ui->openGLViewport->getSurface3D().saveSceneJS(fileName);
		}else if(selectedFilter==formats[4]){
			if(fileInfo.suffix()!="dae")
				fileName.append(".dae");
			ui->openGLViewport->getSurface3D().saveCollada(fileName);
		}else if(selectedFilter==formats[5]){
			if(fileInfo.suffix()!="cvs")
				fileName.append(".cvs");
			ui->openGLViewport->getSurface3D().saveCybervision(fileName);
		}else{
			ui->statusBar->showMessage(tr("Bad save format selected"));
			ui->logTextEdit->appendHtml(QString(tr("<b>Bad save format selected:</b> %1")).arg(selectedFilter));
		}
		startPath= fileInfo.canonicalPath();
	}
}

void MainWindow::on_loadSurfaceButton_clicked(){
	QString filter= tr("Cybervision surface") + "(*.cvs);;"+tr("All files")+"(*.*)";
	QString filename = QFileDialog::getOpenFileName(this,tr("Select surface to load"),startPath,filter,0,0);
	if(!filename.isNull()){
		cybervision::Surface surface= cybervision::Surface::fromFile(filename);
		if(!surface.isOk())
			ui->logTextEdit->appendHtml("<b>"+QString(tr("Error loading surface from %1")).arg(QDir::convertSeparators(filename))+"</b>");

		ui->openGLViewport->setSurface3D(surface);
		updateWidgetStatus();

		startPath= QFileInfo(filename).canonicalPath();
	}
}

void MainWindow::on_logDockWidget_visibilityChanged(bool visible){
	ui->actionShow_log->setChecked(visible);
}

void MainWindow::on_actionShow_log_triggered(bool checked){
	ui->logDockWidget->setVisible(checked);
}

void MainWindow::on_inspectorDockWidget_visibilityChanged(bool visible){
	ui->actionShow_statistics->setChecked(visible);
}

void MainWindow::on_actionShow_statistics_triggered(bool checked){
	ui->inspectorDockWidget->setVisible(checked);
}

void MainWindow::on_addImageButton_clicked(){
	QString filter= tr("Images") + "(*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;"+tr("All files")+"(*.*)";
	QStringList filenames = QFileDialog::getOpenFileNames(this,tr("Select images to add"),startPath,filter,0,0);
	for(QStringList::const_iterator it=filenames.begin();it!=filenames.end();it++){
		QString name= QFileInfo(*it).fileName();
		QListWidgetItem* newItem= new QListWidgetItem(name);
		newItem->setData(32,QDir::convertSeparators(*it));
		ui->imageList->addItem(newItem);

		startPath= QFileInfo(*it).canonicalPath();
	}

	updateWidgetStatus();
}

void MainWindow::on_deleteImageButton_clicked(){
	QList<QListWidgetItem*> selection= ui->imageList->selectedItems();
	for(QList<QListWidgetItem*>::const_iterator i=selection.begin();i!=selection.end();i++){
		int row= ui->imageList->row(*i);
		if(row>=0){
			QListWidgetItem* deletedItem= ui->imageList->takeItem(row);
			if(deletedItem)
				delete deletedItem;
		}
	}
}
void MainWindow::on_imageList_itemSelectionChanged(){
	updateWidgetStatus();
}


void MainWindow::on_moveToolButton_toggled(bool checked){
	ui->openGLViewport->setMouseMode(checked?CybervisionViewer::MOUSE_PANNING:CybervisionViewer::MOUSE_ROTATION);
}

void MainWindow::on_rotateToolButton_toggled(bool checked){
	ui->openGLViewport->setMouseMode(!checked?CybervisionViewer::MOUSE_PANNING:CybervisionViewer::MOUSE_ROTATION);
}

void MainWindow::on_gridToolButton_toggled(bool checked){
	ui->openGLViewport->setShowGrid(checked);
}

void MainWindow::on_texture1ToolButton_clicked(){
	ui->openGLViewport->setTextureMode(CybervisionViewer::TEXTURE_1);
}

void MainWindow::on_texture2ToolButton_clicked(){
	ui->openGLViewport->setTextureMode(CybervisionViewer::TEXTURE_2);
}

void MainWindow::on_textureNoneToolButton_clicked(){
	ui->openGLViewport->setTextureMode(CybervisionViewer::TEXTURE_NONE);
}

void MainWindow::on_crosssectionButton_clicked(bool checked){
	if(checked){
		QMutexLocker lock(&ui->openGLViewport->getSurfaceMutex());
		if(ui->openGLViewport->getSurface3D().isOk())
			ui->openGLViewport->setDrawCrossSectionLine(checked);
		else{
			ui->crosssectionButton->setChecked(false);
			ui->crosssectionButton->setEnabled(false);
		}
	}else
		ui->openGLViewport->setDrawCrossSectionLine(checked);
}

void MainWindow::on_crosssectionPSpinBox_valueChanged(int arg1){
	updateSurfaceStats();
}

