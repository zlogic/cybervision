#include "fundamentalmatrix.h"

#include <Reconstruction/options.h>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <limits>
#include <QFile>
#include <QTextStream>

namespace cybervision{

	FundamentalMatrix::FundamentalMatrix(QObject *parent) : QObject(parent){ }

	Eigen::Vector3d FundamentalMatrix::point2vector(const QPointF&p)const{
		Eigen::Vector3d vector;
		vector(0,0)=p.x(),vector(1,0)=p.y(),vector(2,0)=1;
		return vector;
	}

	QPointF FundamentalMatrix::vector2point(const Eigen::Vector3d&vector)const{
		return QPointF(vector(0,0)/vector(2,0),vector(1,0)/vector(2,0));
	}

	SortedKeypointMatches FundamentalMatrix::getAcceptedMatches()const{ return matches; }
	Eigen::Matrix3d FundamentalMatrix::getFundamentalMatrix()const{ return F; }
	Eigen::Matrix3d FundamentalMatrix::getT1()const{ return T1; }
	Eigen::Matrix3d FundamentalMatrix::getT2()const{ return T2; }

	Eigen::Matrix3d FundamentalMatrix::computeFundamentalMatrix(const KeypointMatches& matches){
		if(matches.size()!=8){
			emit sgnLogMessage(QString("Wrong consensus set size (%1), should be %2").arg(matches.size()).arg(8));
			//TODO:fail here
		}
		//Create the Chi matrix
		Eigen::Matrix<double,8,9> chi;
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
		Eigen::Matrix3d F;
		{
			Eigen::JacobiSVD<Eigen::Matrix<double,8,9>,Eigen::FullPivHouseholderQRPreconditioner> svd(chi, Eigen::ComputeFullV);
			Eigen::Matrix<double,9,9> V_chi= svd.matrixV();
			//Get and unstack E
			F(0,0)=V_chi(0,8), F(1,0)=V_chi(1,8), F(2,0)=V_chi(2,8);
			F(0,1)=V_chi(3,8), F(1,1)=V_chi(4,8), F(2,1)=V_chi(5,8);
			F(0,2)=V_chi(6,8), F(1,2)=V_chi(7,8), F(2,2)=V_chi(8,8);
		}

		return F;
	}

	double FundamentalMatrix::computeFundamentalMatrixError(const Eigen::Matrix3d&F, const KeypointMatch& match) const{
		Eigen::Vector3d x1; x1(0,0)=match.a.x(), x1(1,0)=match.a.y(), x1(2,0)=1;
		Eigen::Vector3d x2; x2(0,0)=match.b.x(), x2(1,0)=match.b.y(), x2(2,0)=1;

		Eigen::Matrix<double,1,1> x2tFx1= x2.transpose()*F*x1;
		Eigen::Vector3d Fx1=F*x1,Ftx2=F.transpose()*x2;
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

	Eigen::Matrix3d FundamentalMatrix::computeFundamentalMatrix(){
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
		Eigen::Matrix3d best_F;

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
				Eigen::Matrix3d F_normalized=computeFundamentalMatrix(consensus_set_normalized);

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
			Eigen::Matrix3d zero; zero.fill(0);
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

	bool FundamentalMatrix::computeFundamentalMatrix(const SortedKeypointMatches&matches){
		this->matches= matches;
		//Compute centroid and scaling factor (normalise2dpts)
		{
			QPointF centroidA,centroidB;
			double scalingA, scalingB;
			double sumAX=0,sumAY=0,sumBX=0,sumBY=0;
			for(SortedKeypointMatches::const_iterator it1= this->matches.begin();it1!=this->matches.end();it1++){
				sumAX+= it1.value().a.x(), sumBX+= it1.value().b.x();
				sumAY+= it1.value().a.y(), sumBY+= it1.value().b.y();
			}
			centroidA.setX(sumAX/this->matches.size()), centroidB.setX(sumBX/this->matches.size());
			centroidA.setY(sumAY/this->matches.size()), centroidB.setY(sumBY/this->matches.size());
			double distA=0,distB=0;
			for(SortedKeypointMatches::const_iterator it1= this->matches.begin();it1!=this->matches.end();it1++){
				double dAX= it1.value().a.x()-centroidA.x(), dAY= it1.value().a.y()-centroidA.y();
				double dBX= it1.value().b.x()-centroidB.x(), dBY= it1.value().b.y()-centroidB.y();
				distA+= sqrt(dAX*dAX+dAY*dAY);
				distB+= sqrt(dBX*dBX+dBY*dBY);
			}
			distA/= this->matches.size();
			distB/= this->matches.size();
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
		F= computeFundamentalMatrix();

		//"Normalize" E
		//(Project into essential space (do we need this?))
		if(true){
			Eigen::JacobiSVD<Eigen::Matrix3d,Eigen::FullPivHouseholderQRPreconditioner> svd(F, Eigen::ComputeFullV|Eigen::ComputeFullU);
			Eigen::Vector3d Sigma= svd.singularValues();
			Sigma(2,0)= 0.0;
			F= svd.matrixU()*(Sigma.asDiagonal())*(svd.matrixV().transpose());
		}

		//De-normalize F
		F= T1.transpose()*F*T2;

		{
			Eigen::Matrix3d zero; zero.fill(0);
			if(F==zero)
				return false;
		}
		return true;
	}


	void FundamentalMatrix::saveAcceptedMatches(const QFileInfo &target){
		QFile file(target.absoluteFilePath());
		if (file.open(QFile::WriteOnly|QFile::Text)) {
			QTextStream out(&file);

			for(SortedKeypointMatches::const_iterator i=matches.begin();i!=matches.end();i++){
				out<<QString("%1\t%2\t%3\t%4\n").arg(i.value().a.x()).arg(i.value().a.y()).arg(i.value().b.x()).arg(i.value().b.y());
			}
			file.close();
			emit sgnLogMessage(QString("Saved RANSAC-filtered matches to %1").arg(target.absoluteFilePath()));
		}else{
			emit sgnLogMessage(QString("Error when saving RANSAC-filtered matches to file %1").arg(target.absoluteFilePath()));
		}
	}
}
