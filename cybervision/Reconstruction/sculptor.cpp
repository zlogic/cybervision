#include <QGLWidget>
#include <QMap>
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
	Sculptor::Sculptor(const QList<QVector3D>& points){
		if(!points.empty()){
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
		qreal aspectRatio= (max.x()-min.x())/(max.y()-min.y());
		qreal scale_x= aspectRatio*Options::surfaceSize/(max.x()-min.x());
		qreal scale_y= -Options::surfaceSize/(max.y()-min.y());

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
				QVector3D scaled_point((it.key().x()-min.x())*scale_x-Options::surfaceSize/2,
							(it.key().y()-min.y())*scale_y+Options::surfaceSize/2,
							(z-min.z())*scale_y
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

					if(fabs(points[p].z()-point1.z())>distanceXY*Options::peakSize && fabs(points[p].z()-point2.z())>distanceXY*Options::peakSize){
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
			//it->setZ((it->z()-zMin)*Options::surfaceDepth/(zMax-zMin));
		}

		return pointsModified;
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

	QVector3D Sculptor::calcNormal(const QVector3D& a, const QVector3D& b)const{
		QVector3D dotProduct=QVector3D::crossProduct(a,b);
		return dotProduct/dotProduct.length();
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


		//Filter peaks
		for(int i=0;i<Options::maxPeakFilterPasses;i++)
			if(!filterTriangles(points,unfilteredTriangles)) break;

		surface.points= points;
		for(QList<Surface::Triangle>::const_iterator it= unfilteredTriangles.begin();it!=unfilteredTriangles.end();it++)
			surface.triangles.push_back(createTriangle(points,it->a,it->b,it->c));
	}


}
