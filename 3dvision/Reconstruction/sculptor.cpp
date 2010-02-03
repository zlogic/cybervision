#include <QGLWidget>
#include <QMap>
#include <limits>

#define _USE_MATH_DEFINES
#include <cmath>

#include <Reconstruction/options.h>
#include <Reconstruction/surface.h>
#include "sculptor.h"

bool operator<(const QPoint& a,const QPoint& b){
	if(a.x()<b.x())
		return true;
	else if(a.x()==b.x())
		return a.y()<b.y();
	else
		return false;
}

namespace cybervision{
	Sculptor::Sculptor(const QList<QVector3D>& points){
		if(!points.empty())
			createSurface(points);
	}

	Surface Sculptor::getSurface()const{ return surface; }

	void Sculptor::createSurface(const QList<QVector3D> &points){
		//Get data from the point set
		QVector3D centroid(0,0,0);
		QVector3D min( std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
		QVector3D max(-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity());
		for(QList<QVector3D>::const_iterator it= points.begin();it!=points.end();it++){
			min.setX(qMin(min.x(),it->x()));
			min.setY(qMin(min.y(),it->y()));
			min.setZ(qMin(min.z(),it->z()));
			max.setX(qMax(max.x(),it->x()));
			max.setY(qMax(max.y(),it->y()));
			max.setZ(qMax(max.z(),it->z()));

			centroid.setX(centroid.x()+it->x());
			centroid.setY(centroid.x()+it->y());
			centroid.setZ(centroid.x()+it->z());
		}

		float scale_z=(max.z()-min.z())/10;
		float step_x= (max.x()-min.x())/(qreal)Options::surfaceSteps;
		float step_y= (max.y()-min.y())/(qreal)Options::surfaceSteps;
		float aspectRatio= (max.x()-min.x())/(max.y()-min.y());


		//Compute Z in various grid points
		QMap<QPoint,double> gridData;
		for(int i=0;i<Options::surfaceSteps;i++){
			float x0= i*step_x,x1=(i+1)*step_x;
			for(int j=0;j<Options::surfaceSteps;j++){
				float y0= j*step_y,y1=(j+1)*step_y;
				//Search for points in local range
				QList<QVector3D> selected_points= getPointsInRange(points,QVector3D(x0,y0,0),QVector3D(x1,y1,0),true);

				if(selected_points.empty())
					continue;

				//This is trivial, a good description is in Notes_on_Lidar_interpolation.pdf
				QVector3D center(x0+0.5F*step_x,y0+0.5F*step_y,0);
				double averageZ=0,sumDistance=0;
				for(QList<QVector3D>::const_iterator it= selected_points.begin();it!=selected_points.end();it++){
					float dx= it->x()-center.x(), dy= it->y()-center.y();
					float distance= sqrt(dx*dx+dy*dy);
					averageZ+= it->z()/distance;
					sumDistance+= 1.0/distance;
				}
				averageZ= ((averageZ/sumDistance)-min.z())/scale_z;
				gridData.insert(QPoint(i,(Options::surfaceSteps-1-j)),averageZ/(double)selected_points.size());
			}
		}

		//Complete holes in grid
		/*
		for(int k=0;k<Options::surfaceSteps;k++){
			for(int i=0;i<Options::surfaceSteps-1;i++){
				for(int j=0;j<Options::surfaceSteps-1;j++){
					QPoint p00(i,j);
					if(gridData.contains(p00))
						continue;

					QPoint p10(i+1,j), p01(i,j+1), p11(i+1,j+1);
					double averageZ= 0;
					int averageZCount= 0;
					if(gridData.contains(p10)){
						averageZ+= gridData.value(p10);
						averageZCount++;
					}
					if(gridData.contains(p11)){
						averageZ+= gridData.value(p11);
						averageZCount++;
					}
					if(gridData.contains(p01)){
						averageZ+= gridData.value(p01);
						averageZCount++;
					}

					if(averageZCount>0)
						gridData.insert(p00,averageZ/averageZCount);
				}
			}
		}*/

		//Create quad-polygons
		for(int i=0;i<Options::surfaceSteps-1;i++){
			for(int j=0;j<Options::surfaceSteps-1;j++){
				QPoint p00(i,j), p10(i+1,j), p01(i,j+1), p11(i+1,j+1);
				int x= i-(Options::surfaceSteps/2),y= j-(Options::surfaceSteps/2);
				if(gridData.contains(p00) && gridData.contains(p01) && gridData.contains(p10) && gridData.contains(p11)){
					Surface::Triangle triangle1,triangle2;
					triangle1.a= QVector3D(aspectRatio*x,y,-gridData[p00]);
					triangle1.b= QVector3D(aspectRatio*(x+1),y,-gridData[p10]);
					triangle1.c= QVector3D(aspectRatio*x,y+1,-gridData[p01]);
					triangle1.normal= calcNormal(triangle1.b-triangle1.a,triangle1.c-triangle1.a);
					//if(triangle1.normal.z()<0)
					//	triangle1.normal.setZ(-triangle1.normal.z());
					surface.triangles.push_back(triangle1);

					triangle2.a= QVector3D(aspectRatio*x,y+1,-gridData[p01]);
					triangle2.b= QVector3D(aspectRatio*(x+1),y,-gridData[p10]);
					triangle2.c= QVector3D(aspectRatio*(x+1),y+1,-gridData[p11]);
					triangle2.normal= calcNormal(triangle2.b-triangle2.a,triangle2.c-triangle2.a);
					//if(triangle2.normal.z()<0)
					//	triangle2.normal.setZ(-triangle2.normal.z());

					surface.triangles.push_back(triangle2);
				}
			}
		}
	}

	QList<QVector3D> Sculptor::getPointsInRange(const QList<QVector3D> &points, QVector3D min, QVector3D max, bool ignoreZ)const{
		QList<QVector3D> result;
		for(QList<QVector3D>::const_iterator it= points.begin();it!=points.end();it++)
			if(it->x()>=min.x() && it->y()>=min.y() && (ignoreZ?true:it->z()>=min.z())
				&& it->x()<=max.x() && it->y()<=max.y() && (ignoreZ?true:it->z()<=max.z()))
				result.push_back(*it);

		if(!result.empty())
			return result;

		//Cannot find points inside region, will have to use closest neighbors
		QMap<float,QVector3D> closestNeighbors;

		QVector3D center(min.x()+(max.x()-min.x())/2,min.y()+(max.y()-min.y())/2,min.z()+(max.z()-min.z())/2);

		for(QList<QVector3D>::const_iterator it= points.begin();it!=points.end();it++){
			float dx= it->x()-center.x(), dy= it->y()-center.y(), dz=ignoreZ?0.0F:it->z()-center.z();
			float distance= sqrt(dx*dx+dy*dy+dz*dz);
			closestNeighbors.insertMulti(distance,*it);
			while(closestNeighbors.size()>2)
				closestNeighbors.erase(closestNeighbors.end()-1);
		}

		for(QMap<float,QVector3D>::const_iterator it= closestNeighbors.begin();it!=closestNeighbors.end();it++)
			result.push_back(it.value());
		return result;
	}


	QVector3D  Sculptor::calcNormal(const QVector3D& a, const QVector3D& b)const{
		QVector3D dotProduct=QVector3D::crossProduct(a,b);
		return dotProduct/dotProduct.length();
	}
}
