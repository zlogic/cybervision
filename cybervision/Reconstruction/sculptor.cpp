#include <QGLWidget>
#include <QMap>
#include <QMatrix4x4>
#include <limits>

#define _USE_MATH_DEFINES
#include <cmath>

#include <Reconstruction/options.h>
#include <Reconstruction/surface.h>
#include "sculptor.h"

inline bool operator<(const QPoint& a,const QPoint& b){
	if(a.x()<b.x())
		return true;
	else if(a.x()==b.x())
		return a.y()<b.y();
	else
		return false;
}

inline bool operator<(const QPointF& a,const QPointF& b){
	if(a.x()<b.x())
		return true;
	else if(qFuzzyCompare(a.x(),b.x()))
		return a.y()<b.y();
	else
		return false;
}

inline bool qFuzzyCompare(const QPointF& a,const QPointF& b){
	return qFuzzyCompare(a.x(),b.x())&&qFuzzyCompare(a.y(),b.y());
}

namespace cybervision{
	Sculptor::Sculptor(const QList<QVector3D>& points,QSize imageSize,qreal scaleXY,qreal scaleZ){
		this->scaleXY= scaleXY, this->scaleZ= scaleZ;
		this->imageSize= imageSize;

		if(Options::mapPointsToGrid){
			QList<QVector3D> gridPoints= interpolatePointsToGrid(points);
			if(!gridPoints.empty())
				delaunayTriangulate(gridPoints);
		}else{
			if(!points.empty())
				delaunayTriangulate(points);
		}
	}

	Surface Sculptor::getSurface()const{ return surface; }

	QList<QVector3D> Sculptor::filterPoints(const QList<QVector3D>& points){
		//Get data from the point set
		QVector3D centroid(0,0,0);
		QVector3D min( std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
		QVector3D max(-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity());

		//These values are unwanted
		qreal max_z=points.begin()->z(), min_z=points.begin()->z();
		for(QList<QVector3D>::const_iterator it= points.begin();it!=points.end();it++){
			max_z= qMax(max_z,it->z());
			min_z= qMin(min_z,it->z());
		}

		for(QList<QVector3D>::const_iterator it= points.begin();it!=points.end();it++){
			min.setX(qMin(min.x(),it->x()));
			min.setY(qMin(min.y(),it->y()));
			min.setZ(qMin(min.z(),it->z()));
			max.setX(qMax(max.x(),it->x()));
			max.setY(qMax(max.y(),it->y()));
			max.setZ(qMax(max.z(),it->z()));

			centroid.setX(centroid.x()+it->x());
			centroid.setY(centroid.y()+it->y());
			centroid.setZ(centroid.z()+it->z());
		}

		centroid.setX(centroid.x()/points.count());
		centroid.setY(centroid.y()/points.count());
		centroid.setZ(centroid.z()/points.count());

		qreal scale_x= scaleXY;
		qreal scale_y= scaleXY;
		qreal scale_z= scaleZ;

		QPointF center;
		{
			center.setX((max.x()-min.x())*scaleXY/2.0);
			center.setY((max.y()-min.y())*scaleXY/2.0);
		}

		surface.scale= Options::surfaceSize/(scaleXY*qMax(max.x()-min.x(),max.y()-min.y()));

		QMap<QPointF,qreal> pointsMap;
		for(QList<QVector3D>::const_iterator it= points.begin();it!=points.end();it++){
			QPointF point(it->x(),it->y());
			pointsMap.insertMulti(point,it->z());
		}
		qreal sum=0;
		int count=0;

		QList<QVector3D> filteredPoints;
		for(QMap<QPointF,qreal>::const_iterator it=pointsMap.begin();it!=pointsMap.end();it++){
			sum+= it.value();
			count++;
			if((it+1)==pointsMap.end() || it.key()!=(it+1).key()){
				qreal z= sum/(qreal)count;
				QVector3D scaled_point((it.key().x()-min.x())*scale_x-center.x(),
							-(it.key().y()-min.y())*scale_y+center.y(),
							-(z-min.z())*scale_z
				);
				filteredPoints.push_back(scaled_point);
				sum= 0;
				count= 0;
			}
		}
		return filteredPoints;
	}


	bool Sculptor::filterTriangles(QList<QVector3D>& points,const QList<Surface::Triangle>& triangles){

		bool pointsModified= false;

		for(Surface::PolygonPoint p=0;p<points.size();p++){
			//TODO: fail if we exceed maximum value of int
			bool isPeakCandidate=false;
			QList<double> sorted_heights;
			//Find all triangles containing current point
			for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
				//Search for min and max values of points' neighbors as well as the max height
				bool found=false;
				QVector3D point1, point2;

				if(it->a==p && it->b!=p && it->c!=p){
					point1= points[it->b], point2= points[it->c];
					found=true;
				}else if(it->a!=p && it->b==p && it->c!=p){
					point1= points[it->a], point2= points[it->c];
					found=true;
				}else if(it->a!=p && it->b!=p && it->c==p){
					point1= points[it->a], point2= points[it->b];
					found=true;
				}

				if(found){
					sorted_heights.append(point1.z());
					sorted_heights.append(point2.z());

					qreal distanceXY= (point1.x()-point2.x())*(point1.x()-point2.x()) + (point1.y()-point2.y())*(point1.y()-point2.y());

					if(fabs(points[p].z()-point1.z())>distanceXY*Options::peakSize*surface.scale && fabs(points[p].z()-point2.z())>distanceXY*Options::peakSize*surface.scale){
						isPeakCandidate=true;
					}else{
						isPeakCandidate=false;
					}
				}
			}

			//Compute height median
			qSort(sorted_heights.begin(),sorted_heights.end());
			if(sorted_heights.length()>0){
				double median=sorted_heights.length()%2==1 ?
							  sorted_heights[(sorted_heights.length()-1)/2] :
							  ((sorted_heights[sorted_heights.length()/2-1]+sorted_heights[sorted_heights.length()/2])/2);

				if(isPeakCandidate){
					//this is a peak
					points[p].setZ(median);
					pointsModified=true;
				}
			}
		}

		//Renormalize z-coordinate
		qreal zMin= std::numeric_limits<qreal>::infinity(), zMax= -std::numeric_limits<qreal>::infinity();
		for(QList<QVector3D>::const_iterator it= points.begin();it!=points.end();it++){
			zMin= qMin(zMin,it->z());
			zMax= qMax(zMax,it->z());
		}

		for(QList<QVector3D>::iterator it= points.begin();it!=points.end();it++){
			it->setZ(it->z()-zMin);
			//it->setZ((it->z()-zMin)*10.0/(zMax-zMin));
		}

		return pointsModified;
	}

	Sculptor::CellData::CellData(){
		averageDepth=0;
		min= QVector2D(0,0), max= QVector2D(0,0);
	}
	Sculptor::CellData::CellData(const QList<QVector3D> &points,qreal parentAverageDepth, const QVector2D &min, const QVector2D &max){
		averageDepth=parentAverageDepth;
		this->min= min, this->max= max;
		/*
		for(QList<QVector3D>::const_iterator it=points.begin();it!=points.end();it++){
			if(it->x()>=min.x() && it->x()<=max.x() && it->y()>=min.y() && it->y()<=max.y()){
				averageDepth+= it->z();
				this->points.append(*it);
			}
		}

		averageDepth/= (this->points.size()+1);
		*/
		double sumDepth=0,sumDistance=0;
		QVector2D middle= min+(max-min)/2;
		//Create filter for small peaks
		QVector<qreal> Zp;
		for(QList<QVector3D>::const_iterator it=points.begin();it!=points.end();it++){
			qreal distance= (QVector2D(*it)-middle).length();
			if(distance <= Options::gridCellArea*(max-middle).length())
				Zp<< it->z();
		}
		qSort(Zp);
		qreal median,
				Zmax= !Zp.isEmpty()?Zp.at(Zp.size()-1):0,
				Zmin= !Zp.isEmpty()?Zp.at(0):0;
		if(Zp.isEmpty())
			median= parentAverageDepth;
		else if(Zp.size()%2==0)
			median= (Zp.at(Zp.size()/2-1)+Zp.at(Zp.size()/2))/2.0;
		else
			median= Zp.at(Zp.size()/2-1);

		if(Zp.size()>=3){
			qreal threshold= 2;
			if(abs(Zp.at(0)-median)>threshold*abs(Zp.at(1)-median))
				Zmin= Zp.at(1);
			if(abs(Zp.at(Zp.size()-1)-median)>threshold*abs(Zp.at(Zp.size()-2)-median))
				Zmax= Zp.at(Zp.size()-2);
		}
		Zp.clear();

		for(QList<QVector3D>::const_iterator it=points.begin();it!=points.end();it++){
			if((it->z()<Zmin) || (it->z()>Zmax))
				continue;
			qreal distance= (QVector2D(*it)-middle).length();
			if(distance <= Options::gridCellArea*(max-middle).length()){
				sumDepth+= it->z()/distance;
				sumDistance+= 1/distance;
			}

			if(it->x()>=min.x() && it->x()<=max.x() && it->y()>=min.y() && it->y()<=max.y()){
				this->points.append(*it);
			}
		}
		if(sumDistance>0){
			averageDepth= sumDepth/sumDistance;
		}
	}
	Sculptor::CellData::CellData(const CellData& cellData){ this->operator =(cellData); }
	void Sculptor::CellData::operator =(const CellData& cellData){
		averageDepth= cellData.averageDepth;
		min= cellData.min, max= cellData.max;
		points= cellData.points;
	}

	QList<QVector3D> Sculptor::interpolatePointsToGrid(const QList<QVector3D>& points)const{
		QList<CellData> mappedPoints;

		//Fill the first cell
		{
			qreal minX= points.begin()->x(), minY= points.begin()->y(), maxX= points.begin()->x(), maxY= points.begin()->y();
			for(QList<QVector3D>::const_iterator it=points.begin();it!=points.end();it++){
				minX= qMin(minX,it->x());
				minY= qMin(minY,it->y());
				maxX= qMax(maxX,it->x());
				maxY= qMax(maxY,it->y());
			}
			//minX= 0, minY= 0;
			//maxX= imageSize.width(), maxY= imageSize.height();
			mappedPoints << CellData(points,0,QVector2D(minX,minY),QVector2D(maxX,maxY));
		}

		//Double the resolution on every iteration by splitting each cell into 4 equal parts
		for(int resolution=0;resolution<Options::gridResolution;resolution++){
			QList<CellData> newMappedPoints;

			#pragma omp parallel
			for(QList<CellData>::const_iterator it= mappedPoints.begin();it!=mappedPoints.end();it++){
				#pragma omp single nowait
				{
					QList<CellData> currentCellMappedPoints;
					QVector2D middle((it->min.x()+it->max.x())/2,(it->min.y()+it->max.y())/2);
					currentCellMappedPoints << CellData(points,it->averageDepth,it->min,middle);
					currentCellMappedPoints << CellData(points,it->averageDepth,QVector2D(it->min.x(),middle.y()),QVector2D(middle.x(),it->max.y()));
					currentCellMappedPoints << CellData(points,it->averageDepth,QVector2D(middle.x(),it->min.y()),QVector2D(it->max.x(),middle.y()));
					currentCellMappedPoints << CellData(points,it->averageDepth,middle,it->max);
					#pragma omp critical
					{
						newMappedPoints.append(currentCellMappedPoints);
					}
				}
			}
			mappedPoints= newMappedPoints;
		}

		//Extract interpolated points
		QList<QVector3D> interpolatedPoints;
		for(QList<CellData>::const_iterator it= mappedPoints.begin();it!=mappedPoints.end();it++){
			if(it->points.size()>0)
				for(QList<QVector3D>::const_iterator jt= it->points.begin();jt!=it->points.end();jt++){
					if(Options::gridAddRealPoints)
						interpolatedPoints << *jt;
				}
			interpolatedPoints << QVector3D(it->min.x(),it->min.y(),it->averageDepth);
		}

		return interpolatedPoints;
	}

	Surface::Triangle Sculptor::createTriangle(const QList<QVector3D>& points,const Surface::PolygonPoint& a, const Surface::PolygonPoint& b, const Surface::PolygonPoint& c)const{
		Surface::Triangle triangle;
		triangle.a= a;
		triangle.b= b;
		triangle.c= c;

		//Make sure the triangle has correct winding
		if(QLineF(points[triangle.a].x(),points[triangle.a].y(),points[triangle.b].x(),points[triangle.b].y())
			.angleTo(QLineF(points[triangle.b].x(),points[triangle.b].y(),points[triangle.c].x(),points[triangle.c].y()))<180.0)
			qSwap(triangle.b,triangle.c);
		triangle.normal= calcNormal(points[triangle.b]-points[triangle.a],points[triangle.c]-points[triangle.a]);
		//if(triangle.normal.z()<0)
		//	triangle.normal.setZ(-triangle.normal.z());

		return triangle;
	}

	QVector3D Sculptor::calcNormal(const QList<Surface::Triangle>& triangles,const QList<QVector3D>& points,const Surface::PolygonPoint& point)const{
		Surface::PolygonPoint a= point;
		QVector3D normal;
		int N=0;
		for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
			Surface::PolygonPoint b,c;
			if(it->a==a && (it->b!=a && it->c!=a)){
				b=it->b;
				c=it->c;
			}else if (it->b==a && (it->c!=a && it->b!=a)){
				b=it->c;
				c=it->a;
			}else if (it->c==a && (it->a!=a && it->b!=a)){
				b=it->a;
				c=it->b;
			} else continue;

			//Rotate line a-b 90 degrees
			if(Options::averageNormalsMode== Options::AVERAGE_NORMALS_LINE){
				QVector3D Va= points[a], Vb= points[b], Vc= points[c];
				QVector3D Vab= Vb-Va, Vac= Vc-Va;
				normal+= calcNormal(Vab)+calcNormal(Vac);
				N+= 2;
			}else if(Options::averageNormalsMode== Options::AVERAGE_NORMALS_TRIANGLE){
				normal += it->normal;
				N++;
			}
		}
		return normal/N;
	}

	QVector3D Sculptor::calcNormal(const QVector3D& a, const QVector3D& b)const{
		QVector3D dotProduct=QVector3D::crossProduct(a,b);
		return dotProduct/dotProduct.length();
	}

	QVector3D Sculptor::calcNormal(const QVector3D& vector)const{
		//Rotate vector's projection onto XY plane 90 degrees
		QVector3D projection(vector.x(),vector.y(),0);
		QMatrix4x4 rotationMatrix; rotationMatrix.rotate(90.0,0,0,1);
		QVector3D projection_rotated= rotationMatrix*projection;
		return calcNormal(vector,projection_rotated);
	}


	bool Sculptor::delaunayCircumCircle(const QPointF& p,const QPointF& a,const QPointF& b,const QPointF& c,QVector3D* circle)const{
		double m1,m2,mx1,mx2,my1,my2;
		double dx,dy,rsqr,drsqr;
		double xc, yc, r;

		/* Check for coincident points */
		if(qFuzzyCompare(a.y(),b.y()) && qFuzzyCompare(b.y(),c.y())){
			// Points are coincident
			return false;
		}

		if(qFuzzyCompare(b.y(),a.y())){
			m2= -(c.x()-b.x())/(c.y()-b.y());
			mx2= (b.x()+c.x())/2.0;
			my2= (b.y()+c.y())/2.0;
			xc= (b.x()+a.x())/2.0;
			yc= m2*(xc-mx2)+my2;
		}else if(qFuzzyCompare(c.y(),b.y())){
			m1= -(b.x()-a.x())/(b.y()-a.y());
			mx1= (a.x()+b.x())/2.0;
			my1= (a.y()+b.y())/2.0;
			xc= (c.x()+b.x())/2.0;
			yc= m1*(xc-mx1)+my1;
		}else{
			m1= -(b.x()-a.x())/(b.y()-a.y());
			m2= -(c.x()-b.x())/(c.y()-b.y());
			mx1= (a.x()+b.x())/2.0;
			mx2= (b.x()+c.x())/2.0;
			my1= (a.y()+b.y())/2.0;
			my2= (b.y()+c.y())/2.0;
			xc= (m1*mx1-m2*mx2+my2-my1)/(m1-m2);
			yc= m1*(xc-mx1)+my1;
		}

		dx = b.x()-xc;
		dy = b.y()-yc;
		rsqr = dx*dx+dy*dy;
		r = sqrt(rsqr);

		dx = p.x()-xc;
		dy = p.y()-yc;
		drsqr = dx*dx+dy*dy;

		if(circle)
			*circle= QVector3D(xc,yc,r);

		return drsqr <= rsqr?true:false;

	}

	void Sculptor::delaunayTriangulate(const QList<QVector3D>& unfilteredPoints){
		QList<QVector3D> points=filterPoints(unfilteredPoints);
		QList<QVector3D> pointsNoSuperTriangle=points;


		/*
			Set up the supertriangle
			This is a triangle which encompasses all the sample points.
			The supertriangle coordinates are added to the end of the
			vertex list. The supertriangle is the first triangle in
			the triangle list.
		*/
		typedef QPair<Surface::PolygonPoint,Surface::PolygonPoint> Edge;

		QList<QPair<Surface::Triangle,bool> > triangles;
		Surface::Triangle superTriangle;
		{
			/*
			Find the maximum and minimum vertex bounds.
			This is to allow calculation of the bounding triangle
			*/
			QVector3D min( std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
			QVector3D max(-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity());
			for(QList<QVector3D>::const_iterator it= points.begin();it!=points.end();it++){
				min.setX(qMin(min.x(),it->x()));
				min.setY(qMin(min.y(),it->y()));
				min.setZ(qMin(min.z(),it->z()));
				max.setX(qMax(max.x(),it->x()));
				max.setY(qMax(max.y(),it->y()));
				max.setZ(qMax(max.z(),it->z()));
			}
			float dx= max.x()-min.x();
			float dy= max.y()-min.y();
			float dmax= (dx > dy) ? dx : dy;
			float xmid= (max.x()+min.x())/2.0;
			float ymid= (max.y()+min.x())/2.0;

			points<<QVector3D(xmid-2.0*dmax,ymid-dmax,0);
			superTriangle.a= points.length()-1;
			points<<QVector3D(xmid,ymid+2.0*dmax,0);
			superTriangle.b= points.length()-1;
			points<<QVector3D(xmid+2.0*dmax,ymid-dmax,0);
			superTriangle.c= points.length()-1;

			triangles<<QPair<Surface::Triangle,bool>(superTriangle,false);
		}
		/*
			Include each point one at a time into the existing mesh
		*/
		for(Surface::PolygonPoint it= 0;it<points.length();it++){
			//TODO: fail if we exceed maximum value of int
			QPointF p(points[it].x(),points[it].y());

			QList<Edge> edges;
			/*
				Set up the edge buffer.
				If the point (xp,yp) lies inside the circumcircle then the
				three edges of that triangle are added to the edge buffer
				and that triangle is removed.
			*/
			for(QList<QPair<Surface::Triangle,bool> >::iterator jt=triangles.begin();jt!=triangles.end();){
				if(jt->second){
					jt++;
					continue;
				}
				QPointF a(points[jt->first.a].x(),points[jt->first.a].y());
				QPointF b(points[jt->first.b].x(),points[jt->first.b].y());
				QPointF c(points[jt->first.c].x(),points[jt->first.c].y());
				QVector3D circle;
				bool inside= delaunayCircumCircle(p,a,b,c,&circle);
				if (circle.x() + circle.z() < p.x())
					jt->second= true;
				if (inside){
					edges<<Edge(jt->first.a,jt->first.b);
					edges<<Edge(jt->first.b,jt->first.c);
					edges<<Edge(jt->first.c,jt->first.a);
					jt=triangles.erase(jt);
					continue;
				}
				jt++;
			}
			/*
				Tag multiple edges
				Note: if all triangles are specified anticlockwise then all
				interior edges are opposite pointing in direction.
			*/
			Surface::PolygonPoint vector_inf= -1;
			if(edges.size()>=2)
			for(QList<Edge>::iterator edge1= edges.begin();edge1!=(edges.end()-1);edge1++){
				for (QList<Edge>::iterator edge2= edge1+1;edge2!=edges.end();edge2++){
					if ((edge1->first==edge2->second) && (edge1->second==edge2->first)){
						*edge1= Edge(vector_inf,vector_inf);
						*edge2= Edge(vector_inf,vector_inf);
					}
					/*
					if ((edge1->first==edge2->first) && (edge1->second==edge2->second)){
						*edge1= Edge(vector_inf,vector_inf);
						*edge2= Edge(vector_inf,vector_inf);
					}
					*/
				}
			}

			/*
				Form new triangles for the current point
				Skipping over any tagged edges.
				All edges are arranged in clockwise order.
			*/
			for (QList<Edge>::iterator edge= edges.begin();edge!=edges.end();edge++){
				if(edge->first==vector_inf || edge->second==vector_inf)
					continue;

				Surface::Triangle triangle;
				triangle.a= edge->first;
				triangle.b = edge->second;
				triangle.c = it;
				triangles<<QPair<Surface::Triangle,bool>(triangle,false);
			}
		}

		QList<Surface::Triangle> unfilteredTriangles;
		for(QList<QPair<Surface::Triangle,bool> >::const_iterator it= triangles.begin();it!=triangles.end();it++){
			if((it->first.a!=superTriangle.a) && (it->first.a!=superTriangle.b) &&(it->first.a!=superTriangle.c))
				if((it->first.b!=superTriangle.a) && (it->first.b!=superTriangle.b) &&(it->first.b!=superTriangle.c))
					if((it->first.c!=superTriangle.a) && (it->first.c!=superTriangle.b) &&(it->first.c!=superTriangle.c))
						unfilteredTriangles.push_back(it->first);
		}
		triangles.clear();

		points=pointsNoSuperTriangle;
		pointsNoSuperTriangle.clear();

		//Filter peaks
		//if(!Options::mapPointsToGrid)
			for(int i=0;i<Options::maxPeakFilterPasses;i++)
				if(!filterTriangles(points,unfilteredTriangles)) break;

		//Calculate triangle normals
		for(QList<Surface::Triangle>::const_iterator it= unfilteredTriangles.begin();it!=unfilteredTriangles.end();it++)
			surface.triangles.push_back(createTriangle(points,it->a,it->b,it->c));


		//Calculate point normals
		for(Surface::PolygonPoint i=0;i!=points.size();i++){
			Surface::Point point;
			point.coord= points[i];
			point.normal= calcNormal(surface.triangles,points,i);
			surface.points.append(point);
		}

		//Calculate height median, min & max
		{
			QList<qreal> heights;
			for(QList<QVector3D>::const_iterator it=points.begin();it!=points.end();it++)
				heights<<it->z();
			qSort(heights);

			if(heights.length()>0){
				double median=heights.length()%2==1 ?
							  heights[(heights.length()-1)/2] :
							  ((heights[heights.length()/2-1]+heights[heights.length()/2])/2);
				surface.medianDepth= median;
				surface.minDepth= heights[0];
				surface.maxDepth= heights[heights.length()-1];
			}
		}
	}


}
