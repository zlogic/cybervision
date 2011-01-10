#include "reconstructor.h"

#include <Reconstruction/svd.h>
#include <SIFT/siftgateway.h>
#include <KDTree/kdtreegateway.h>

#include <limits>
#include <ctime>

#include <QImage>

#define USE_PRECOMPUTED_DATA
#ifdef USE_PRECOMPUTED_DATA
#include <QFileInfo>
#include <QDir>
#include <QTextStream>
#endif

namespace cybervision{
	bool Reconstructor::KeypointMatch::operator==(const KeypointMatch&m)const{ return a==m.a && b==m.b; }

	Reconstructor::Reconstructor(){
		srand(time(NULL));
	}

	Reconstructor::SortedKeypointMatches Reconstructor::extractMatches(const QString& filename1,const QString& filename2){
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

				SortedKeypointMatches matches;
				while(!out_stream.atEnd()){
					double distance;
					KeypointMatch match;
					out_stream>>distance>>match.a.rx()>>match.a.ry()>>match.b.rx()>>match.b.ry();
					matches.insert(distance,match);
				}
				precomputed_file.close();

				emit sgnLogMessage(QString("Loading images %1 and %2 to obtain image sizes").arg(filename1).arg(filename2));
				QImage img1(filename1),img2(filename2);
				if(img1.size()!=img2.size())//TODO:Error here!
					emit sgnLogMessage(QString("Images %1 and %2 have different sizes!").arg(filename1).arg(filename2));
				else
					imageSize= img1.size();
				return matches;
			}
		}

		emit sgnStatusMessage("Detecting SIFT keypoints...");
		emit sgnLogMessage("Starting SIFT keypoint detection");
		emit sgnLogMessage(QString("Loading images %1 and %2").arg(filename1).arg(filename2));
		QImage img1(filename1),img2(filename2);
		if(img1.size()!=img2.size())//TODO:Error here!
			emit sgnLogMessage(QString("Images %1 and %2 have different sizes!").arg(filename1).arg(filename2));
		else
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
		SortedKeypointMatches matches;
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

		return matches;
	}



	QList<Reconstructor::StereopairPosition> Reconstructor::computePose(SortedKeypointMatches& matches){
		emit sgnStatusMessage("Estimating pose...");

		//Compute fundamental matrix
		QGenericMatrix<3,3,double> best_F= computeFundamentalMatrix(matches);

		{
			QGenericMatrix<3,3,double> zero; zero.fill(0);
			if(best_F==zero)
				return QList<Reconstructor::StereopairPosition>();
		}

		QList<StereopairPosition> RTList= computeRT(best_F);
		//Output R,T matrices to log

		QString RT_str;
		for(QList<StereopairPosition>::const_iterator i= RTList.begin();i!=RTList.end();i++){
			RT_str.append(QString("R%1T\n").arg(QString(""),40));
			for(int j=0;j<3;j++){
				qreal T_value_i= j==0? i->T(0,0): (j==1?i->T(1,0):i->T(2,0));//Extract i-th value from QVector3D
				QString matrix_row= QString("%1 %2 %3 %4\n").arg(i->R(j,0),6,'g',6).arg(i->R(j,1),6,'g',6).arg(i->R(j,2),6,'g',6).arg(T_value_i,6,'g',6);
				RT_str.append(matrix_row);
			}
		}
		emit sgnLogMessage(QString("Resulting camera poses\n").append(RT_str));
		return RTList;
	}

	QGenericMatrix<1,3,double> Reconstructor::point2vector(const QPointF&p)const{
		QGenericMatrix<1,3,double> vector;
		vector(0,0)=p.x(),vector(1,0)=p.y(),vector(2,0)=1;
		return vector;
	}

	QPointF Reconstructor::vector2point(const QGenericMatrix<1,3,double>&vector)const{
		return QPointF(vector(0,0)/vector(2,0),vector(1,0)/vector(2,0));
	}

	QGenericMatrix<3,3,double> Reconstructor::computeFundamentalMatrix(SortedKeypointMatches& matches){
		//Compute centroid and scaling factor (normalise2dpts)
		QGenericMatrix<3,3,double> T1,T2;
		{
			QPointF centroidA,centroidB;
			double scalingA, scalingB;
			double sumAX=0,sumAY=0,sumBX=0,sumBY=0;
			for(SortedKeypointMatches::const_iterator it1= matches.begin();it1!=matches.end();it1++){
				sumAX+= it1.value().a.x(), sumBX+= it1.value().b.x();
				sumAY+= it1.value().a.y(), sumBY+= it1.value().b.y();
			}
			centroidA.setX(sumAX/matches.size()), centroidB.setX(sumBX/matches.size());
			centroidA.setY(sumAY/matches.size()), centroidB.setY(sumBY/matches.size());
			double distA=0,distB=0;
			for(SortedKeypointMatches::const_iterator it1= matches.begin();it1!=matches.end();it1++){
				double dAX= it1.value().a.x()-centroidA.x(), dAY= it1.value().a.y()-centroidA.y();
				double dBX= it1.value().b.x()-centroidB.x(), dBY= it1.value().b.y()-centroidB.y();
				distA+= sqrt(dAX*dAX+dAY*dAY);
				distB+= sqrt(dBX*dBX+dBY*dBY);
			}
			distA/= matches.size();
			distB/= matches.size();
			scalingA= sqrt(2)/distA;
			scalingB= sqrt(2)/distB;

			//De-normalize2dpts
			T1.fill(0.0);
			T1(0,0)= scalingA, T1(1,1)= scalingA, T1(2,2)= 1.0;
			T1(0,2)= -scalingA*centroidA.x(), T1(1,2)= -scalingA*centroidA.y();
			T2.fill(0.0);
			T2(0,0)= scalingB, T2(1,1)= scalingB, T2(2,2)= 1.0;
			T2(0,2)= -scalingB*centroidB.x(), T2(1,2)= -scalingB*centroidB.y();
		}

		//Do not normalize
		//T1.fill(0.0); T1(0,0)=1, T1(1,1)=1, T1(2,2)=1;
		//T2=T1;

		//Compute fundamental matrix
		QGenericMatrix<3,3,double> F= ransacComputeFundamentalMatrix(matches,T1,T2);

		//De-normalize F
		F= T1.transposed()*F*T2;

		return F;
	}

	QGenericMatrix<3,3,double> Reconstructor::ransacComputeFundamentalMatrix(SortedKeypointMatches& matches,QGenericMatrix<3,3,double> T1,QGenericMatrix<3,3,double> T2){
		emit sgnStatusMessage("Estimating fundamental matrix...");

		//Increase speed with precomputed lists
		QList<float> matches_keys= matches.uniqueKeys();
		int matches_keys_max_i=0;
		for(int i=0;i<matches_keys.size();i++)
			if(matches_keys.at(i)<=Options::ReliableDistance)
				matches_keys_max_i= i;
			else
				break;
		matches_keys_max_i=std::max(matches_keys_max_i,8);


		//Use the RANSAC algorithm to estimate camera poses
		KeypointMatches best_consensus_set;
		double best_error= std::numeric_limits<float>::infinity();
		QGenericMatrix<3,3,double> best_F;

		#pragma omp parallel
		for(int i=0;i<Options::RANSAC_k;i++){
		#pragma omp single nowait
		{
			if(i%(Options::RANSAC_k/10)==0)
				emit sgnLogMessage(QString("RANSAC %1% complete").arg((i*100)/Options::RANSAC_k));

			//Extract RANSAC_n random values
			KeypointMatches consensus_set,consensus_set_normalized;
			SortedKeypointMatches master_consensus_set;

			while(consensus_set.size()<Options::RANSAC_n){
				//Generate random distance
				int rand_number= (matches_keys_max_i*(long long)rand())/RAND_MAX;
				float random_distance= matches_keys.at(std::min(rand_number,matches_keys.size()-1));
				//Find points at generated distance
				QList<KeypointMatch> it1_values= matches.values(random_distance);
				//Find a match on a random position
				int random_pos= (it1_values.size()*(long long)rand())/RAND_MAX;
				QPair<float,KeypointMatch> new_match(
					random_distance,
					it1_values.at(std::min(it1_values.size()-1,random_pos))
				);
				if(!master_consensus_set.contains(new_match.first,new_match.second)){
					master_consensus_set.insert(new_match.first,new_match.second);
					consensus_set.push_back(new_match);

					//Create normalized point
					QPointF x1=	vector2point(T1*point2vector(new_match.second.a));
					QPointF x2=	vector2point(T2*point2vector(new_match.second.b));
					KeypointMatch new_match_normalized_points;
					new_match_normalized_points.a=x1, new_match_normalized_points.b=x2;
					QPair<float,KeypointMatch> new_match_normalized(random_distance,new_match_normalized_points);
					consensus_set_normalized.push_back(new_match_normalized);
				}
			}

			double error=0;
			//Compute E from the random values
			QGenericMatrix<3,3,double> F_normalized=computeFundamentalMatrix8Point(consensus_set_normalized);

			//Expand consensus set
			for(SortedKeypointMatches::const_iterator it1= matches.begin();it1!=matches.end();it1++){
				//Normalize point for error computation

				QPointF x1=	vector2point(T1*point2vector(it1.value().a));
				QPointF x2=	vector2point(T2*point2vector(it1.value().b));
				KeypointMatch match_normalized; match_normalized.a=x1, match_normalized.b=x2;
				//Check error
				double current_error= computeFundamentalMatrixError(F_normalized,match_normalized);

				//Check if match exists in master consensus set
				if(master_consensus_set.contains(it1.key(),it1.value())){
					//Add to global error
					error+= current_error;
					continue;//Match was found in master consensus set, ignore
				}
				if(current_error<Options::RANSAC_t){
					//Add to global error
					error+= current_error;
					consensus_set.push_back(QPair<float,KeypointMatch>(it1.key(),it1.value()));
				}
			}

			#pragma omp critical
			{
				if(consensus_set.size()>Options::RANSAC_d){
					//Error was already computed, normalize it
					error/= consensus_set.size();
					if(consensus_set.size()>best_consensus_set.size() || (consensus_set.size()==best_consensus_set.size() && error<best_error)){
						best_consensus_set= consensus_set;
						best_error= error;
						best_F= F_normalized;
					}
				}
			}
		}
		}

		if(best_consensus_set.empty()){
			emit sgnLogMessage("No RANSAC consensus found");
			QGenericMatrix<3,3,double> zero; zero.fill(0);
			return zero;
		}else{
			emit sgnLogMessage(QString("RANSAC consensus found, error=%1, size=%2").arg(best_error,0,'g',4).arg(best_consensus_set.size()));
		}

		//Update matches
		matches.clear();
		for(KeypointMatches::const_iterator it=best_consensus_set.begin();it!=best_consensus_set.end();it++)
			matches.insert(it->first,it->second);

		return best_F;
	}


	QGenericMatrix<3,3,double> Reconstructor::computeFundamentalMatrix8Point(const KeypointMatches& matches){
		if(matches.size()!=8){
			emit sgnLogMessage(QString("Wrong consensus set size (%1), should be %2").arg(matches.size()).arg(8));
			//TODO:fail here
		}
		//Create the Chi matrix
		QGenericMatrix<9,8,double> chi;
		{
			int i=0;
			for(KeypointMatches::const_iterator it= matches.begin();it!=matches.end();it++,i++){
				qreal x1= it->second.a.x(), x2=it->second.b.x(), y1=it->second.a.y(), y2=it->second.b.y(), z1=1, z2=1;
				//Kronecker product
				chi(i,0)= x1*x2, chi(i,1)= x1*y2, chi(i,2)= x1*z2;
				chi(i,3)= y1*x2, chi(i,4)= y1*y2, chi(i,5)= y1*z2;
				chi(i,6)= z1*x2, chi(i,7)= z1*y2, chi(i,8)= z1*z2;
			}
		}
		//Compute V_chi from the SVD decomposition
		QGenericMatrix<3,3,double> F;
		{
			SVD<9,8,double> svd(chi);
			QGenericMatrix<9,9,double> V_chi= svd.getV();
			//Get and unstack E
			F(0,0)=V_chi(0,8), F(1,0)=V_chi(1,8), F(2,0)=V_chi(2,8);
			F(0,1)=V_chi(3,8), F(1,1)=V_chi(4,8), F(2,1)=V_chi(5,8);
			F(0,2)=V_chi(6,8), F(1,2)=V_chi(7,8), F(2,2)=V_chi(8,8);
		}
		//"Normalize" E
		//(Project into essential space (do we need this?))
		if(false){
			SVD<3,3,double> svd(F);
			QGenericMatrix<3,3,double> Sigma= svd.getSigma();
			Sigma(2,2)= 0.0;
			/*
			Sigma.fill(0.0);
			Sigma(0,0)= 1.0, Sigma(1,1)= 1.0;
			*/

			F= svd.getU()*Sigma*(svd.getV().transposed());
		}

		return F;
	}

	double Reconstructor::computeFundamentalMatrixError(const QGenericMatrix<3,3,double>&F, const KeypointMatch& match) const{
		QGenericMatrix<1,3,double> x1; x1(0,0)=match.a.x(), x1(1,0)=match.a.y(), x1(2,0)=1;
		QGenericMatrix<1,3,double> x2; x2(0,0)=match.b.x(), x2(1,0)=match.b.y(), x2(2,0)=1;

		QGenericMatrix<1,1,double> x2tFx1= x2.transposed()*F*x1;
		QGenericMatrix<1,3,double> Fx1=F*x1,Ftx2=F.transposed()*x2;
		//Calculate distance from epipolar line to points
		return fabs(x2tFx1(0,0)*x2tFx1(0,0)*(1/(Fx1(0,0)*Fx1(0,0)+Fx1(1,0)*Fx1(1,0))+1/(Ftx2(0,0)*Ftx2(0,0)+Ftx2(1,0)*Ftx2(1,0))));
		/*
		//Badly working method 1
		return fabs(x2tEx1(0,0)*x2tEx1(0,0)/(Ex1(0,0)*Ex1(0,0)+Ex1(1,0)*Ex1(1,0)+Etx2(0,0)*Etx2(0,0)+Etx2(1,0)*Etx2(1,0)));
		*/
		/*
		//Badly working method 2
		return fabs(x2tEx1(0,0));
		*/
	}

	//Essential matrix method functions

	QGenericMatrix<3,3,double> Reconstructor::computeRT_rzfunc(double angle) const{
		QGenericMatrix<3,3,double> result;
		result.fill(0.0);
		result(0,1)= angle>=0?1.0:(-1.0);
		result(1,0)= -(angle>=0?1.0:(-1.0));
		result(2,2)= 1.0;
		return result;
	}


	QGenericMatrix<3,3,double> Reconstructor::computeCameraMatrix()const{
		QGenericMatrix<3,3,double> K;
		K.fill(0.0);
		K(0,0)= Options::scaleFocalDistance;//Focal distance X
		K(1,1)= Options::scaleFocalDistance;//Focal distance Y
		K(0,2)= -imageSize.width();//Optical center X
		K(1,2)= -imageSize.height();//Optical center Y
		K(2,2)= 1;
		return K;
	}

	QList<Reconstructor::StereopairPosition> Reconstructor::computeRT(const QGenericMatrix<3,3,double>&Essential_matrix) const{
		QList<Reconstructor::StereopairPosition> RTList;
		QList<double> pi_values;
		pi_values<<M_PI_2<<-M_PI_2;

		QGenericMatrix<3,3,double> camera_K= computeCameraMatrix();
		QGenericMatrix<3,3,double> Essential_matrix_projected=camera_K.transposed()*Essential_matrix*camera_K;

		//Project into essential space
		{
			SVD<3,3,double> svd(Essential_matrix_projected);
			QGenericMatrix<3,3,double> U= svd.getU(),Sigma=svd.getSigma(), V=svd.getV();

			double Sigma_value= 1;//(Sigma(0,0)+Sigma(1,1))/2.0;
			Sigma.fill(0.0);
			Sigma(0,0)= Sigma_value,Sigma(1,1) = Sigma_value;

			Essential_matrix_projected= U*Sigma*(V.transposed());
		}

		SVD<3,3,double> svd(Essential_matrix_projected);
		QGenericMatrix<3,3,double> U= svd.getU(),Sigma=svd.getSigma(), V=svd.getV();

		for(QList<double>::const_iterator i= pi_values.begin();i!=pi_values.end();i++){
			double PI_R= *i;
			QGenericMatrix<3,3,double> R= U*(computeRT_rzfunc(PI_R).transposed())*(V.transposed());
			//if(R(0,0)*(R(1,1)*R(2,2)-R(1,2)*R(2,1))-R(0,1)*(R(1,0)*R(2,2)-R(1,2)*R(2,0))+R(0,2)*(R(1,0)*R(2,1)-R(1,1)*R(2,0))<0)
			//	R=R*(-1.0);//Probably unnecessary
			for(QList<double>::const_iterator j= pi_values.begin();j!=pi_values.end();j++){
				double PI_T=*j;

				QGenericMatrix<1,3,double> T_unhatted;

				/*
				QGenericMatrix<3,3,double> T= U*computeRT_rzfunc(PI_T)*Sigma*(U.transposed());
				T_unhatted(0,0)= (T(2,1)-T(1,2))/2;
				T_unhatted(1,0)= (T(0,2)-T(2,0))/2;
				T_unhatted(2,0)= (T(1,0)-T(0,1))/2;
				*/

				T_unhatted(0,0)=U(0,2)*(PI_T>=0?1:-1);
				T_unhatted(1,0)=U(1,2)*(PI_T>=0?1:-1);
				T_unhatted(2,0)=U(2,2)*(PI_T>=0?1:-1);

				StereopairPosition RT;
				RT.R= R, RT.T= T_unhatted;
				RTList.push_back(RT);
			}
		}

		return RTList;
	}


	QList<QVector3D> Reconstructor::compute3DPoints(const SortedKeypointMatches&matches,const QList<StereopairPosition>& RTList){
		emit sgnStatusMessage("Performing 3D triangulation...");

		double best_min_depth=std::numeric_limits<double>::infinity(),best_max_depth=-std::numeric_limits<double>::infinity();

		StereopairPosition bestPosition=RTList.first();
		bool found=false;
		//Find best R/T pair
		for(QList<StereopairPosition>::const_iterator it1=RTList.begin();it1!=RTList.end();it1++){
			QList<QVector3D> Points3d= computeTriangulatedPoints(matches,it1->R,it1->T,false);

			//Search for maximum depth in order to evaluate and select best configuration
			double max_depth= -std::numeric_limits<double>::infinity(), min_depth=std::numeric_limits<double>::infinity();
			for(QList<QVector3D>::const_iterator it2=Points3d.begin();it2!=Points3d.end();it2++){
				if(it2->z()>max_depth)
					max_depth= it2->z();
				if(it2->z()<min_depth)
					min_depth= it2->z();
			}

			if(min_depth>0 && max_depth-min_depth>best_max_depth-best_min_depth){
				best_min_depth= min_depth;
				best_max_depth= max_depth;
				bestPosition= *it1;
				found=true;
			}

			emit sgnLogMessage(QString("Minimum depth is %1, maximum depth is %2").arg(min_depth).arg(max_depth));
		}

		if(best_min_depth<0)
			emit sgnLogMessage(QString("Warning: minimum depth is %1, less than zero!").arg(best_min_depth));

		if(best_min_depth==std::numeric_limits<double>::infinity())
			emit sgnLogMessage(QString("Warning: minimum depth is less than zero!"));

		return computeTriangulatedPoints(matches,bestPosition.R,bestPosition.T,true);
	}

	QList<QVector3D> Reconstructor::computeTriangulatedPoints(const SortedKeypointMatches&matches,const QGenericMatrix<3,3,double>&R,const QGenericMatrix<1,3,double>& T,bool normalizeCameras){
		QGenericMatrix<4,3,double> P1,P2;
		P1.fill(0.0);
		for(int i=0;i<3;i++)
			P1(i,i)= 1;

		P2.fill(0.0);
		for(int i=0;i<3;i++)
			for(int j=0;j<3;j++)
				P2(i,j)= R(i,j);

		for(int i=0;i<3;i++)
			P2(i,3)= T(i,0);

		//Search for maximums
		double max_x=0, max_y=0;
		for(SortedKeypointMatches::const_iterator it=matches.begin();it!=matches.end();it++){
			if(fabs(it.value().a.x())>max_x)
				max_x= fabs(it.value().a.x());
			if(fabs(it.value().b.x())>max_x)
				max_x= fabs(it.value().b.x());
			if(fabs(it.value().a.y())>max_y)
				max_y= fabs(it.value().a.y());
			if(fabs(it.value().b.y())>max_y)
				max_y= fabs(it.value().b.y());
		}

		//Camera matrix
		if(normalizeCameras){
			QGenericMatrix<3,3,double> camera_K= computeCameraMatrix();
			P1=camera_K*P1;
			P2=camera_K*P2;
		}

		QList<QVector3D> resultPoints;

		QGenericMatrix<4,4,double> A;

		for(int i=0;i<2;i++)
		for(SortedKeypointMatches::const_iterator it=matches.begin();it!=matches.end();it++){
			QPointF x1= it.value().a, x2= it.value().b;
			for(int i=0;i<4;i++){
				A(0,i)= x1.x()*P1(2,i)-P1(0,i);
				A(1,i)= x1.y()*P1(2,i)-P1(1,i);
				A(2,i)= x2.x()*P2(2,i)-P2(0,i);
				A(3,i)= x2.y()*P2(2,i)-P2(1,i);
			}

			//Normalise rows of A
			/*
			for(int i=0;i<4;i++){
				double row_norm=0;
				for(int j=0;j<4;j++)
					row_norm+= A(i,j)*A(i,j);

				row_norm=sqrt(row_norm);
				for(int j=0;j<4;j++)
					A(i,j)/= row_norm;
			}
			*/

			SVD<4,4,double> svd(A);
			QGenericMatrix<4,4,double> Sigma= svd.getSigma();
			QGenericMatrix<4,4,double> V= svd.getV();
			//Search for min column
			size_t Sigma_min_index=0;
			for(size_t i=1;i<4;i++)
				Sigma_min_index= Sigma(i,i)<Sigma(Sigma_min_index,Sigma_min_index)?i:Sigma_min_index;

			//Sigma_min_index=3;//DZ 02.12.2010 seems Sigma=0 gives the best results. Sigma=1 gives a rougher surface.
			QGenericMatrix<1,4,double> V_col_min;
			for(int i=0;i<4;i++)
				V_col_min(i,0)= V(i,Sigma_min_index);

			QVector3D resultPoint(
					x1.x()*Options::scaleXY,
					x1.y()*Options::scaleXY,
					(Options::scaleXY*Options::scaleZ)*(V_col_min(2,0)-(V_col_min(0,0)+V_col_min(1,0)))/V_col_min(3,0));

			resultPoints.push_back(resultPoint);
		}

		return resultPoints;
	}

	//Fundamental matrix method functions
	QList<QVector3D> Reconstructor::compute3DPoints(const SortedKeypointMatches&matches,QGenericMatrix<3,3,double> F){
		//Project into fundamental space (satisfy rank-2 criteria)
		{
			SVD<3,3,double> svd(F);
			QGenericMatrix<3,3,double> Sigma= svd.getSigma();
			Sigma(2,2)= 0.0;
			F= svd.getU()*Sigma*(svd.getV().transposed());
		}


		QList<QVector3D> resultPoints;

		//For every keypoint match
		for(SortedKeypointMatches::const_iterator match_i=matches.begin();match_i!=matches.end();match_i++){
			QPointF x1= match_i.value().a, x2= match_i.value().b;

			QGenericMatrix<3,3,double> T1,T2;
			T1.fill(0), T2.fill(0);
			T1(0,0)=1, T1(1,1)=1, T1(2,2)=1, T1(0,2)= -x1.x(), T1(1,2)= -x1.y();
			T2(0,0)=1, T2(1,1)=1, T2(2,2)=1, T2(0,2)= -x2.x(), T2(1,2)= -x2.y();

		}

		return resultPoints;
	}


	//Main reconstruction function
	bool Reconstructor::run(const QString& filename1,const QString& filename2){
		Points3D.clear();
		//Extract and sort matches by distance
		SortedKeypointMatches matches= extractMatches(filename1,filename2);
		if(matches.isEmpty()){
			errorString= "No matches found";
			return false;
		}
		if(matches.size()<Options::MinMatches){
			errorString= QString("Not enough matches (%1), need at least %2").arg(matches.size()).arg(Options::MinMatches);
			return false;
		}

		if(Options::reconstructionMode==Options::RECONSTRUCTION_ESSENTIAL_FULL){
			emit sgnLogMessage("Performing reconstruction with essential matrix and complete pose data");
			//Estimate camera poses
			QList<Reconstructor::StereopairPosition> RTList= computePose(matches);
			if(RTList.empty()){
				errorString= "Error when estimating pose";
				return false;
			}

			//Triangulate points
			Points3D= compute3DPoints(matches,RTList);
			if(Points3D.empty()){
				errorString= "Error during 3D triangulation";
				return false;
			}
			//log points to console
			/*
			QString pointsStr="A=[";
			for(QList<QVector3D>::const_iterator it2=Points3D.begin();it2!=Points3D.end();it2++)
				pointsStr.append(QString("%1 %2 %3%4").arg(it2->x()).arg(it2->y()).arg(-it2->z(),0,'g',12).arg(it2!=(Points3D.end()-1)?";":""));
			pointsStr.append("];");

			emit sgnLogMessage(QString(pointsStr));
			*/
		}else if(Options::reconstructionMode==Options::RECONSTRUCTION_FUNDAMENTAL){
			emit sgnLogMessage("Performing reconstruction with fundamental matrix");

			//Compute fundamental matrix
			QGenericMatrix<3,3,double> F= computeFundamentalMatrix(matches);

			//Optimal triangulation of points
			Points3D= compute3DPoints(matches,F);
			if(Points3D.empty()){
				errorString= "Error during 3D triangulation";
				return false;
			}
		}else{
			errorString= QString("Unknown reconstruction mode: %1").arg(Options::reconstructionMode);
			return false;
		}
		return true;
	}

	//Getters
	bool Reconstructor::isOk()const{ return !errorString.isNull()&&!errorString.isEmpty(); }
	QString Reconstructor::getErrorString()const{ return errorString; }
	QList<QVector3D> Reconstructor::get3DPoints()const{ return Points3D; }
}
