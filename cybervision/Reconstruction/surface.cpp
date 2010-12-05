#include <QGLWidget>
#include <QTextStream>

#include "surface.h"
#include "options.h"
#include <cmath>

namespace cybervision{
	Surface::Surface(){

	}

	Surface::Surface(const Surface&sc){
		operator =(sc);
	}

	void Surface::operator =(const Surface&sc){
		this->triangles= sc.triangles;
		this->points= sc.points;
	}

	void Surface::glDraw() const{
		for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
			const QVector3D& pa=points[it->a];
			const QVector3D& pb=points[it->b];
			const QVector3D& pc=points[it->c];

			glBegin(GL_TRIANGLES);
			glColor3f(1.0f, 1.0f, 1.0f);
			//Front side
			glNormal3f(it->normal.x(),it->normal.y(),it->normal.z());
			glVertex3f(pa.x(),pa.y(),pa.z());
			glNormal3f(it->normal.x(),it->normal.y(),it->normal.z());
			glVertex3f(pb.x(),pb.y(),pb.z());
			glNormal3f(it->normal.x(),it->normal.y(),it->normal.z());
			glVertex3f(pc.x(),pc.y(),pc.z());

			//Back side
			glNormal3f(-it->normal.x(),-it->normal.y(),-it->normal.z());
			glVertex3f(pc.x(),pc.y(),pc.z());
			glNormal3f(-it->normal.x(),-it->normal.y(),-it->normal.z());
			glVertex3f(pb.x(),pb.y(),pb.z());
			glNormal3f(-it->normal.x(),-it->normal.y(),-it->normal.z());
			glVertex3f(pa.x(),pa.y(),pa.z());
			glEnd();

			glBegin(GL_POINTS);
			glColor3f(1.0f, 1.0f, 1.0f);
			//glVertex3f(it->a.x(),it->a.y(),it->a.z());
			//glVertex3f(it->a.x(),it->a.y(),it->a.z()+1);

			glEnd();
		}
		/*
		//Draw point cloud
		double radius=.08;
		int num_segments=36;
		for(QList<QVector3D>::const_iterator it= points.begin();it!=points.end();it++){
			glBegin(GL_POLYGON);
			for(int i =0;i<num_segments;i++){
				double angle = i*M_PI*2/(double)num_segments;
				glVertex3f(it->x()+cos(angle)*radius,it->y()+sin(angle)*radius,it->z());
			}
			glEnd();
		}
		*/
	}

	bool Surface::isOk()const{ return !triangles.empty() && !points.empty(); }


	/*
	 Functions for saving image data
	 */
	void Surface::savePoints(QString fileName)const{
		QFile file(fileName);
		file.open(QIODevice::WriteOnly);

		QTextStream stream(&file);
		for(QList<QVector3D>::const_iterator it= points.begin();it!=points.end();it++)
			stream<<it->x()<<"\t"<<it->y()<<"\t"<<it->z()<<"\r\n";

		file.close();
	}
	void Surface::savePolygons(QString fileName)const{
		QFile file(fileName);
		file.open(QIODevice::WriteOnly);

		QTextStream stream(&file);
		for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
			const QVector3D& pa=points[it->a];
			const QVector3D& pb=points[it->b];
			const QVector3D& pc=points[it->c];
			stream<<it->normal.x()<<"\t"<<it->normal.y()<<"\t"<<it->normal.z()<<"\t";
			stream<<pa.x()<<"\t"<<pa.y()<<"\t"<<pa.z()<<"\t";
			stream<<pb.x()<<"\t"<<pb.y()<<"\t"<<pb.z()<<"\t";
			stream<<pc.x()<<"\t"<<pc.y()<<"\t"<<pc.z()<<"\r\n";
		}

		file.close();
	}
	void Surface::saveCollada(QString fileName)const{
		//Read XML template
		QString xmlTemplate;
		{
			QFile templateFile(":/collada/Template.xml");
			templateFile.open(QIODevice::ReadOnly);
			QTextStream stream(&templateFile);
			xmlTemplate= stream.readAll();
		}

		QFile file(fileName);
		file.open(QIODevice::WriteOnly);

		QTextStream stream(&file);

		QString vertexesString,normalsString,trianglesString;

		QString result=xmlTemplate;
		if(Options::colladaFormat==Options::COLLADA_SHARED_POINTS){
			//Output points
			for(QList<QVector3D>::const_iterator it= points.begin();it!=points.end();it++){
				QString currentVertexString;
				currentVertexString.append(QString("%1 %2 %3 ").arg(it->x()).arg(it->y()).arg(it->z()));
				vertexesString.append(currentVertexString);
			}

			//Output normals and polygons
			for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
				QString currentNormalString;
				currentNormalString.append(QString("%1 %2 %3 ").arg(it->normal.x()).arg(it->normal.y()).arg(it->normal.z()));
				QString currentTriangleString;
				currentTriangleString.append(QString("%1 %2 %3 ").arg(it->a).arg(it->b).arg(it->c));

				normalsString.append(currentNormalString+currentNormalString+currentNormalString);
				trianglesString.append(currentTriangleString);
			}

			result.replace("##[points-array-size]##",QString("%1").arg(points.length()*3));
			result.replace("##[points-count]##",QString("%1").arg(points.length()));
			result.replace("##[normals-array-size]##",QString("%1").arg(triangles.length()*9));
			result.replace("##[normals-count]##",QString("%1").arg(triangles.length()));
			result.replace("##[triangles-count]##",QString("%1").arg(triangles.length()));
		}else if(Options::colladaFormat==Options::COLLADA_INDEPENDENT_POLYGONS){
			PolygonPoint i=0;//Triangle indexes
			for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
				const QVector3D& pa=points[it->a];
				const QVector3D& pb=points[it->b];
				const QVector3D& pc=points[it->c];
				QString currentVertexString;
				currentVertexString.append(QString("%1 %2 %3 ").arg(pa.x()).arg(pa.y()).arg(pa.z()));
				currentVertexString.append(QString("%1 %2 %3 ").arg(pb.x()).arg(pb.y()).arg(pb.z()));
				currentVertexString.append(QString("%1 %2 %3 ").arg(pc.x()).arg(pc.y()).arg(pc.z()));
				QString currentNormalString;
				currentNormalString.append(QString("%1 %2 %3 ").arg(it->normal.x()).arg(it->normal.y()).arg(it->normal.z()));
				QString currentTriangleString;
				currentTriangleString.append(QString("%1 %2 %3 ").arg(i).arg(i+1).arg(i+2));

				vertexesString.append(currentVertexString);
				normalsString.append(currentNormalString+currentNormalString+currentNormalString);
				trianglesString.append(currentTriangleString);
				i+=3;
			}

			result.replace("##[points-array-size]##",QString("%1").arg(triangles.length()*9));
			result.replace("##[points-count]##",QString("%1").arg(triangles.length()*3));
			result.replace("##[normals-array-size]##",QString("%1").arg(triangles.length()*9));
			result.replace("##[normals-count]##",QString("%1").arg(triangles.length()*3));
			result.replace("##[triangles-count]##",QString("%1").arg(triangles.length()));
		}

		result.replace("##[points]##",vertexesString);
		result.replace("##[normals]##",normalsString);
		result.replace("##[triangles-indexes]##",trianglesString);

		stream<<result;
		file.close();
	}
}
