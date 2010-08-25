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

		float aspectRatio= (max.x()-min.x())/(max.y()-min.y());
		float scale_x= aspectRatio*Options::surfaceSize/(max.x()-min.x());
		float scale_y= -Options::surfaceSize/(max.y()-min.y());
		float scale_z= -Options::surfaceDepth/(max.z()-min.z());

		QMap<QPointF,float> pointsMap;
		for(QList<QVector3D>::const_iterator it= points.begin();it!=points.end();it++){
			QPointF point(it->x(),it->y());
			pointsMap.insertMulti(point,it->z());
		}
		float sum=0;
		int count=0;

		QList<QVector3D> filteredPoints;
		for(QMap<QPointF,float>::const_iterator it=pointsMap.begin();it!=pointsMap.end();it++){
			sum+= it.value();
			count++;
			if((it+1)==pointsMap.end() || it.key()!=(it+1).key()){
				float z= sum/(float)count;
				QVector3D scaled_point((it.key().x()-min.x())*scale_x-Options::surfaceSize/2,
							(it.key().y()-min.y())*scale_y+Options::surfaceSize/2,
							(z-min.z())*scale_z
				);
				filteredPoints.push_back(scaled_point);
				sum= 0;
				count= 0;
			}
		}
		return filteredPoints;
	}

	Surface::Triangle Sculptor::createTriangle(const QVector3D& a, const QVector3D& b, const QVector3D& c)const{
		Surface::Triangle triangle;
		triangle.a= a;
		triangle.b= b;
		triangle.c= c;
		if(QLineF(triangle.a.x(),triangle.a.y(),triangle.b.x(),triangle.b.y())
			.angleTo(QLineF(triangle.b.x(),triangle.b.y(),triangle.c.x(),triangle.c.y()))<180.0)
			qSwap(triangle.b,triangle.c);
		triangle.normal= calcNormal(triangle.b-triangle.a,triangle.c-triangle.a);
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
		typedef QPair<QVector3D,QVector3D> Edge;

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
			
			superTriangle.a= QVector3D(xmid-2.0*dmax,ymid-dmax,0);
			superTriangle.b= QVector3D(xmid,ymid+2.0*dmax,0);
			superTriangle.c= QVector3D(xmid+2.0*dmax,ymid-dmax,0);
			triangles<<QPair<Surface::Triangle,bool>(superTriangle,false);
			points<<superTriangle.a<<superTriangle.b<<superTriangle.c;
		}
		/*
			Include each point one at a time into the existing mesh
		*/
		for(QList<QVector3D>::const_iterator it= points.begin();it!=points.end();it++){
			QPointF p(it->x(),it->y());

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
				QPointF a(jt->first.a.x(),jt->first.a.y());
				QPointF b(jt->first.b.x(),jt->first.b.y());
				QPointF c(jt->first.c.x(),jt->first.c.y());
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
			QVector3D vector_inf(std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity());
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
				triangle.c = *it;
				triangles<<QPair<Surface::Triangle,bool>(triangle,false);
			}
		}

		for(QList<QPair<Surface::Triangle,bool> >::const_iterator it= triangles.begin();it!=triangles.end();it++){
			if((it->first.a!=superTriangle.a) && (it->first.a!=superTriangle.b) &&(it->first.a!=superTriangle.c))
				if((it->first.b!=superTriangle.a) && (it->first.b!=superTriangle.b) &&(it->first.b!=superTriangle.c))
					if((it->first.c!=superTriangle.a) && (it->first.c!=superTriangle.b) &&(it->first.c!=superTriangle.c))
						surface.triangles.push_back(createTriangle(it->first.a,it->first.b,it->first.c));
		}
	}


}
