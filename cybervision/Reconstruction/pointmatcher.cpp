#include "pointmatcher.h"

#include <Reconstruction/options.h>
#include <KDTree/kdtreegateway.h>
#include <SIFT/siftgateway.h>


#define USE_PRECOMPUTED_DATA
#ifdef USE_PRECOMPUTED_DATA
#include <QFileInfo>
#include <QDir>
#include <QTextStream>
#endif

namespace cybervision{
	PointMatcher::PointMatcher(QObject *parent) : QObject(parent){ }

	SortedKeypointMatches PointMatcher::getMatches()const{ return matches; }
	QSize PointMatcher::getSize()const{ return imageSize; }

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
				emit sgnLogMessage("Loading precomputed SIFT matches from "+QDir::convertSeparators(precomputed_file.fileName()));
				precomputed_file.open(QFile::ReadOnly);
				QTextStream out_stream(&precomputed_file);

				while(!out_stream.atEnd()){
					double distance;
					KeypointMatch match;
					out_stream>>distance>>match.a.rx()>>match.a.ry()>>match.b.rx()>>match.b.ry();
					matches.insert(distance,match);
				}
				precomputed_file.close();

				emit sgnLogMessage(QString("Loading images %1 and %2 to obtain image sizes").arg(filename1).arg(filename2));
				QImage img1(filename1),img2(filename2);
				if(img1.size()!=img2.size()){//TODO:Error here!
					emit sgnLogMessage(QString("Images %1 and %2 have different sizes!").arg(filename1).arg(filename2));
					return false;
				}else
					imageSize= img1.size();
				return true;
			}
		}

		emit sgnStatusMessage("Detecting SIFT keypoints...");
		emit sgnLogMessage("Starting SIFT keypoint detection");
		emit sgnLogMessage(QString("Loading images %1 and %2").arg(filename1).arg(filename2));
		QImage img1(filename1),img2(filename2);
		if(img1.size()!=img2.size()){//TODO:Error here!
			emit sgnLogMessage(QString("Images %1 and %2 have different sizes!").arg(filename1).arg(filename2));
			return false;
		}else
			imageSize= img1.size();
		QList <SIFT::Keypoint> keypoints1,keypoints2;
		{
			SIFT::Extractor extractor;
			emit sgnLogMessage(QString("Extracting keypoints from %1").arg(filename1));
			keypoints1= extractor.extract(img1);
			emit sgnLogMessage(QString("Extracting keypoints from %2").arg(filename2));
			keypoints2= extractor.extract(img2);
		}


		emit sgnStatusMessage("Matching SIFT keypoints...");
		if(Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_SIMPLE){
			//Simple matching
			emit sgnLogMessage(QString("Matching keypoints from %1 to %2").arg(filename1).arg(filename2));
			//Match first image with second
			#pragma omp parallel
			for(QList<SIFT::Keypoint>::const_iterator it1= keypoints1.begin();it1!=keypoints1.end();it1++){
				#pragma omp single nowait
				{
					KeypointMatch match;
					float minDistance= std::numeric_limits<float>::infinity();
					for(QList<SIFT::Keypoint>::const_iterator it2= keypoints2.begin();it2!=keypoints2.end();it2++){
						float distance= it1->distance(*it2);
						if(distance<minDistance){
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
			emit sgnLogMessage(QString("Matching keypoints from %1 to %2").arg(filename2).arg(filename1));
			//Match second image with first
			#pragma omp parallel
			for(QList<SIFT::Keypoint>::const_iterator it2= keypoints2.begin();it2!=keypoints2.end();it2++){
				#pragma omp single nowait
				{
					KeypointMatch match;
					float minDistance= std::numeric_limits<float>::infinity();
					for(QList<SIFT::Keypoint>::const_iterator it1= keypoints1.begin();it1!=keypoints1.end();it1++){
						float distance= it2->distance(*it1);
						if(distance<minDistance){
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
			emit sgnLogMessage(QString("Matching keypoints from %1 to %2 with kd-tree").arg(filename1).arg(filename2));
			KDTree::KDTreeGateway kdTree(Options::MaxKeypointDistance,Options::bbf_steps);
			cybervision::SortedKeypointMatches current_matches= kdTree.matchKeypoints(keypoints1,keypoints2);
			for(cybervision::SortedKeypointMatches::const_iterator it=current_matches.begin();it!=current_matches.end();it++){
				KeypointMatch m;
				m.a= it.value().a, m.b= it.value().b;
				matches.insert(it.key(),m);
			}

			emit sgnLogMessage(QString("Matching keypoints from %1 to %2 with kd-tree").arg(filename2).arg(filename1));
			current_matches= kdTree.matchKeypoints(keypoints2,keypoints1);
			for(cybervision::SortedKeypointMatches::const_iterator it=current_matches.begin();it!=current_matches.end();it++){
				KeypointMatch m;
				m.a= it.value().b, m.b= it.value().a;
				if(!matches.contains(it.key(),m))
					matches.insert(it.key(),m);
			}
		}

		emit sgnLogMessage(QString("Found %1 keypoint matches").arg(matches.size()));

		if(Options::UsePrecomputedKeypointData){
			emit sgnLogMessage("Saving computed SIFT matches to "+QDir::convertSeparators(precomputed_file_info.fileName()));
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
