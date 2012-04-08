#include "reconstructor.h"

#include <Reconstruction/pointmatcher.h>
#include <Reconstruction/fundamentalmatrix.h>
#include <Reconstruction/pointtriangulator.h>

#include <ctime>
#include <QDir>

namespace cybervision{

Reconstructor::Reconstructor(QObject *parent) : QObject(parent){
	srand(time(NULL));
}

ALIGN_EIGEN_FUNCTION bool Reconstructor::run(const QString& filename1,const QString& filename2,qreal angle){
	Points3D.clear();
	try{
		//Extract and sort matches by distance
		SortedKeypointMatches matches;//Point matches
		{
			PointMatcher matcher(this);
			QObject::connect(&matcher, SIGNAL(sgnLogMessage(QString)),this, SIGNAL(sgnLogMessage(QString)),Qt::DirectConnection);
			QObject::connect(&matcher, SIGNAL(sgnStatusMessage(QString)),this, SIGNAL(sgnStatusMessage(QString)),Qt::DirectConnection);
			bool ok= matcher.extractMatches(filename1,filename2);
			QObject::disconnect(&matcher, SIGNAL(sgnLogMessage(QString)),this, SIGNAL(sgnLogMessage(QString)));
			QObject::disconnect(&matcher, SIGNAL(sgnStatusMessage(QString)),this, SIGNAL(sgnStatusMessage(QString)));
			matches= matcher.getMatches();
			imageSize= matcher.getSize();
			image1= matcher.getImage1();
			image2= matcher.getImage2();
			scaleMetadata= matcher.getScaleMetadata();
			if(!ok || matches.isEmpty()){
				errorString= tr("No matches found");
				return false;
			}
		}
		if(matches.size()<Options::MinMatches){
			errorString= QString(tr("Not enough matches (%1), need at least %2")).arg(matches.size()).arg(Options::MinMatches);
			return false;
		}
		//Compute fundamental matrix
		Eigen::Matrix3d F;
		{
			FundamentalMatrix fundamentalMatrix(this);
			QObject::connect(&fundamentalMatrix, SIGNAL(sgnLogMessage(QString)),this, SIGNAL(sgnLogMessage(QString)),Qt::DirectConnection);
			QObject::connect(&fundamentalMatrix, SIGNAL(sgnStatusMessage(QString)),this, SIGNAL(sgnStatusMessage(QString)),Qt::DirectConnection);
			bool ok= fundamentalMatrix.computeFundamentalMatrix(matches);
			QObject::disconnect(&fundamentalMatrix, SIGNAL(sgnLogMessage(QString)),this, SIGNAL(sgnLogMessage(QString)));
			QObject::disconnect(&fundamentalMatrix, SIGNAL(sgnStatusMessage(QString)),this, SIGNAL(sgnStatusMessage(QString)));
			matches= fundamentalMatrix.getAcceptedMatches();
			F= fundamentalMatrix.getFundamentalMatrix();
			if(!ok || matches.isEmpty()){
				errorString= tr("Error when computing fundamental matrix");
				return false;
			}
			//Save matches if needed
			if(Options::SaveFilteredMatches){
				QObject::connect(&fundamentalMatrix, SIGNAL(sgnLogMessage(QString)),this, SIGNAL(sgnLogMessage(QString)),Qt::DirectConnection);
				fundamentalMatrix.saveAcceptedMatches(QFileInfo(QFileInfo(filename1).absoluteDir(),QFileInfo(filename1).fileName()+" "+QFileInfo(filename1).fileName()+" filtered.txt").absoluteFilePath());
				QObject::disconnect(&fundamentalMatrix, SIGNAL(sgnLogMessage(QString)),this, SIGNAL(sgnLogMessage(QString)));
			}
		}

		//Triangulate points
		{
			PointTriangulator triangulator(this);
			QObject::connect(&triangulator, SIGNAL(sgnLogMessage(QString)),this, SIGNAL(sgnLogMessage(QString)),Qt::DirectConnection);
			QObject::connect(&triangulator, SIGNAL(sgnStatusMessage(QString)),this, SIGNAL(sgnStatusMessage(QString)),Qt::DirectConnection);
			bool ok=false;
			if(Options::triangulationMode==Options::TRIANGULATION_PERSPECTIVE)
				ok= triangulator.triangulatePoints(matches,F,imageSize);
			else if(Options::triangulationMode==Options::TRIANGULATION_PARALLEL)
				ok= triangulator.triangulatePoints(matches,angle,Options::mapPointsToGrid);
			QObject::disconnect(&triangulator, SIGNAL(sgnLogMessage(QString)),this, SIGNAL(sgnLogMessage(QString)));
			QObject::disconnect(&triangulator, SIGNAL(sgnStatusMessage(QString)),this, SIGNAL(sgnStatusMessage(QString)));
			Points3D= triangulator.getPoints3D();
			if(!ok || Points3D.isEmpty()){
				switch(triangulator.getResult()){
				case PointTriangulator::RESULT_POSE_UNDETERMINED:
					errorString= tr("Error when estimating pose");
					break;
				case PointTriangulator::RESULT_TRIANGULATION_ERROR:
					errorString= tr("Error during 3D triangulation");
					break;
				case PointTriangulator::RESULT_OK:
					errorString= tr("Internal logic error: triangulation failed, but result is OK");
					break;
				default:
					errorString= tr("Unknown error");
					break;
				}
				return false;
			}
		}

		//log points to console
		/*
		QString pointsStr="A=[";
		for(QList<QVector3D>::const_iterator it2=Points3D.begin();it2!=Points3D.end();it2++)
			pointsStr.append(QString("%1 %2 %3%4").arg(it2->x()).arg(it2->y()).arg(-it2->z(),0,'g',12).arg(it2!=(Points3D.end()-1)?";":""));
		pointsStr.append("];");

		emit sgnLogMessage(QString(pointsStr));
		*/

		return true;
	} catch(const std::bad_alloc&) {
		errorString= tr("Critical error: out of memory");
		return false;
	} catch(const std::runtime_error& e) {
		errorString= QString(tr("Critical error: %1")).arg(e.what());
		return false;
	} catch(...) {
		errorString= tr("Unknown error");
		return false;
	}
}

//Getters
bool Reconstructor::isOk()const{ return !errorString.isNull()&&!errorString.isEmpty(); }
QString Reconstructor::getErrorString()const{ return errorString; }
QList<QVector3D> Reconstructor::get3DPoints()const{ return Points3D; }
QSize Reconstructor::getImageSize()const{ return imageSize; }
const QImage&  Reconstructor::getImage1()const{ return image1; }
const QImage&  Reconstructor::getImage2()const{ return image2; }
double Reconstructor::getScaleMetadata()const{ return scaleMetadata; }

}
