#include "pointmatcher.h"

#include <Reconstruction/options.h>
#include <Reconstruction/imageloader.h>
#include <KDTree/kdtreegateway.h>
#include <SIFT/siftgateway.h>
#include <Reconstruction/pointmatcheropencl.h>

#define USE_PRECOMPUTED_DATA
#ifdef USE_PRECOMPUTED_DATA
#include <QFileInfo>
#include <QDir>
#include <QTextStream>
#endif

namespace cybervision{
PointMatcher::PointMatcher(QObject *parent) : QObject(parent){ scaleMetadata= -1; }

SortedKeypointMatches PointMatcher::getMatches()const{ return matches; }
QSize PointMatcher::getSize()const{ return imageSize; }
const QImage& PointMatcher::getImage1()const{ return image1; }
const QImage& PointMatcher::getImage2()const{ return image2; }
double PointMatcher::getScaleMetadata()const{ return scaleMetadata; }

bool PointMatcher::extractMatches(const QString& filename1,const QString& filename2){
	matches.clear();
	imageSize= QSize();
	QFileInfo precomputed_file_info;
	if(Options::UsePrecomputedKeypointData){
		QFileInfo file1(filename1), file2(filename2);
		QString precomputed_filename= QString(file1.fileName()+" "+file2.fileName()+".txt");
		precomputed_file_info= QFileInfo(file1.absoluteDir(),precomputed_filename);
		QFile precomputed_file(precomputed_file_info.absoluteFilePath());
		if(precomputed_file.exists()){
			emit sgnLogMessage(tr("Loading precomputed SIFT matches from %1").arg(QDir::convertSeparators(precomputed_file.fileName())));
			precomputed_file.open(QFile::ReadOnly);
			QTextStream in_stream(&precomputed_file);

			while(!in_stream.atEnd()){
				double distance;
				KeypointMatch match;
				in_stream>>distance>>match.a.rx()>>match.a.ry()>>match.b.rx()>>match.b.ry();
				matches.insert(distance,match);
			}
			precomputed_file.close();

			emit sgnLogMessage(QString(tr("Loading images %1 and %2 to obtain image sizes")).arg(filename1).arg(filename2));
			ImageLoader img1Metadata(filename1),img2Metadata(filename2);
			if(img1Metadata.getSize()!=img2Metadata.getSize()){
				emit sgnLogMessage(QString(tr("Images %1 and %2 have different sizes!")).arg(filename1).arg(filename2));
				return false;
			}else
				imageSize= img1Metadata.getSize();

			double scale1= img1Metadata.getScale(), scale2= img2Metadata.getScale();
			if(scale1!=scale2){
				emit sgnLogMessage(QString(tr("Images %1 and %2 have different scales in metadata!")).arg(filename1).arg(filename2));
				return false;
			}
			else{
				scaleMetadata= scale1;
				if(scaleMetadata>0)
					emit sgnLogMessage(QString(tr("Extracted scale %1 from metadata")).arg(scaleMetadata));
				else
					emit sgnLogMessage(QString(tr("No scale in metadata")).arg(scaleMetadata));
			}
			//emit sgnLogMessage(QString(tr("Extracted fields from image 1 metadata:\n%1\n\n")).arg(img1Metadata.getMetadataString()));
			//emit sgnLogMessage(QString(tr("Extracted fields from image 2 metadata:\n%1\n\n")).arg(img2Metadata.getMetadataString()));
			return true;
		}
	}

	emit sgnStatusMessage(tr("Detecting SIFT keypoints..."));
	{
		emit sgnLogMessage(tr("Starting SIFT keypoint detection"));
		emit sgnLogMessage(QString(tr("Loading images %1 and %2")).arg(filename1).arg(filename2));
		ImageLoader img1Metadata(filename1),img2Metadata(filename2);
		if(img1Metadata.getSize()!=img1Metadata.getSize()){
			emit sgnLogMessage(QString(tr("Images %1 and %2 have different sizes!")).arg(filename1).arg(filename2));
			return false;
		}else
			imageSize= img1Metadata.getSize();

		double scale1= img1Metadata.getScale(), scale2= img2Metadata.getScale();
		if(scale1!=scale2){
			emit sgnLogMessage(QString(tr("Images %1 and %2 have different scales in metadata!")).arg(filename1).arg(filename2));
			return false;
		}else{
			scaleMetadata= scale1;
			if(scaleMetadata>0)
				emit sgnLogMessage(QString(tr("Extracted scale %1 from metadata")).arg(scaleMetadata));
			else
				emit sgnLogMessage(QString(tr("No scale in metadata")).arg(scaleMetadata));
		}
		//emit sgnLogMessage(QString(tr("Extracted fields from image 1 metadata:\n%1\n\n")).arg(img1Metadata.getMetadataString()));
		//emit sgnLogMessage(QString(tr("Extracted fields from image 2 metadata:\n%1\n\n")).arg(img2Metadata.getMetadataString()));

		image1= img1Metadata.getImage(),image2= img2Metadata.getImage();
	}
	QList <SIFT::Keypoint> keypoints1,keypoints2;
	{
		SIFT::Extractor extractor(Options::SIFTContrastCorrection);
		emit sgnLogMessage(QString(tr("Extracting keypoints from %1")).arg(filename1));
		keypoints1= extractor.extract(image1);
		emit sgnLogMessage(QString(tr("Extracted %1 keypoints from %2")).arg(keypoints1.size()).arg(filename1));

		emit sgnLogMessage(QString(tr("Extracting keypoints from %1")).arg(filename2));
		keypoints2= extractor.extract(image2);
		emit sgnLogMessage(QString(tr("Extracted %1 keypoints from %2")).arg(keypoints2.size()).arg(filename2));
	}
	emit sgnStatusMessage(tr("Matching SIFT keypoints..."));

	bool OpenCLSucceeded= true;
	if(Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_CPU || Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_GPU){
#ifdef CYBERVISION_OPENCL
		PointMatcherOpenCL clMatcher(128,this);

		QObject::connect(&clMatcher, SIGNAL(sgnLogMessage(QString)),this, SIGNAL(sgnLogMessage(QString)),Qt::DirectConnection);

		if(OpenCLSucceeded){
			emit sgnLogMessage(QString(tr("Matching keypoints from %1 to %2")).arg(filename1).arg(filename2));
			//Match first image with second
			matches= clMatcher.CalcDistances(keypoints1,keypoints2);
			if(matches.empty())
				OpenCLSucceeded= false;

			OpenCLSucceeded&= clMatcher.ShutdownCL();
		}

		QObject::disconnect(&clMatcher, SIGNAL(sgnLogMessage(QString)),this, SIGNAL(sgnLogMessage(QString)));
#else
		OpenCLSucceeded= false;
#endif
	}
	if(Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_SIMPLE || !OpenCLSucceeded){
		double MaxKeypointDistanceSquared= Options::MaxKeypointDistance*Options::MaxKeypointDistance;
		//Simple matching
		emit sgnLogMessage(QString(tr("Matching keypoints from %1 to %2")).arg(filename1).arg(filename2));
		//Match first image with second
		#pragma omp parallel
		for(QList<SIFT::Keypoint>::const_iterator it1= keypoints1.begin();it1!=keypoints1.end();it1++){
			#pragma omp single nowait
			{
				KeypointMatch match;
				float minDistance= std::numeric_limits<float>::infinity();
				for(QList<SIFT::Keypoint>::const_iterator it2= keypoints2.begin();it2!=keypoints2.end();it2++){
					float distance= it1->distance(*it2,MaxKeypointDistanceSquared);
					if(distance<minDistance && distance<Options::MaxKeypointDistance){
						minDistance= distance;
						match.a= QPointF(it1->getX(),it1->getY());
						match.b= QPointF(it2->getX(),it2->getY());
					}
				}
				if(minDistance!=std::numeric_limits<float>::infinity() && minDistance<Options::MaxKeypointDistance){
					#pragma omp critical
					{
						if(!matches.contains(minDistance,match))
							matches.insert(minDistance,match);
					}
				}
			}
		}
		emit sgnLogMessage(QString(tr("Matching keypoints from %1 to %2")).arg(filename2).arg(filename1));
		//Match second image with first
		#pragma omp parallel
		for(QList<SIFT::Keypoint>::const_iterator it2= keypoints2.begin();it2!=keypoints2.end();it2++){
			#pragma omp single nowait
			{
				KeypointMatch match;
				float minDistance= std::numeric_limits<float>::infinity();
				for(QList<SIFT::Keypoint>::const_iterator it1= keypoints1.begin();it1!=keypoints1.end();it1++){
					float distance= it2->distance(*it1,MaxKeypointDistanceSquared);
					if(distance<minDistance && distance<Options::MaxKeypointDistance){
						minDistance= distance;
						match.a= QPointF(it1->getX(),it1->getY());
						match.b= QPointF(it2->getX(),it2->getY());
					}
				}
				if(minDistance!=std::numeric_limits<float>::infinity() && minDistance<Options::MaxKeypointDistance){
					#pragma omp critical
					{
						if(!matches.contains(minDistance,match))
							matches.insert(minDistance,match);
					}
				}
			}
		}
	}else if(Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_KDTREE){
		//KD tree matching
		emit sgnLogMessage(QString(tr("Matching keypoints from %1 to %2 with kd-tree")).arg(filename1).arg(filename2));
		KDTree::KDTreeGateway kdTree(Options::MaxKeypointDistance,Options::bbf_steps);
		cybervision::SortedKeypointMatches current_matches= kdTree.matchKeypoints(keypoints1,keypoints2);
		for(cybervision::SortedKeypointMatches::const_iterator it=current_matches.begin();it!=current_matches.end();it++){
			KeypointMatch m;
			m.a= it.value().a, m.b= it.value().b;
			matches.insert(it.key(),m);
		}

		emit sgnLogMessage(QString(tr("Matching keypoints from %1 to %2 with kd-tree")).arg(filename2).arg(filename1));
		current_matches= kdTree.matchKeypoints(keypoints2,keypoints1);
		for(cybervision::SortedKeypointMatches::const_iterator it=current_matches.begin();it!=current_matches.end();it++){
			KeypointMatch m;
			m.a= it.value().b, m.b= it.value().a;
			if(!matches.contains(it.key(),m))
				matches.insert(it.key(),m);
		}
	}

	emit sgnLogMessage(QString(tr("Found %1 keypoint matches")).arg(matches.size()));

	if(Options::UsePrecomputedKeypointData){
		emit sgnLogMessage(QString(tr("Saving computed SIFT matches to %1")).arg(QDir::convertSeparators(precomputed_file_info.fileName())));
		QFile precomputed_file(precomputed_file_info.absoluteFilePath());
		precomputed_file.open(QFile::WriteOnly);
		QTextStream out_stream(&precomputed_file);

		for(SortedKeypointMatches::const_iterator it=matches.begin();it!=matches.end();it++)
			out_stream<<it.key()<<"\t"<<it.value().a.x()<<"\t"<<it.value().a.y()<<"\t"<<it.value().b.x()<<"\t"<<it.value().b.y()<<"\n";
		precomputed_file.close();
	}

	return !matches.isEmpty();
}

}
