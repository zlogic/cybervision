#include <QGLWidget>
#include <QTextStream>

#include "surface.h"

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
			glBegin(GL_TRIANGLES);
			glColor3f(1.0f, 1.0f, 1.0f);
			//Front side
			glNormal3f(it->normal.x(),it->normal.y(),it->normal.z());
			glVertex3f(it->a.x(),it->a.y(),it->a.z());
			glNormal3f(it->normal.x(),it->normal.y(),it->normal.z());
			glVertex3f(it->b.x(),it->b.y(),it->b.z());
			glNormal3f(it->normal.x(),it->normal.y(),it->normal.z());
			glVertex3f(it->c.x(),it->c.y(),it->c.z());

			//Back side
			glNormal3f(-it->normal.x(),-it->normal.y(),-it->normal.z());
			glVertex3f(it->c.x(),it->c.y(),it->c.z());
			glNormal3f(-it->normal.x(),-it->normal.y(),-it->normal.z());
			glVertex3f(it->b.x(),it->b.y(),it->b.z());
			glNormal3f(-it->normal.x(),-it->normal.y(),-it->normal.z());
			glVertex3f(it->a.x(),it->a.y(),it->a.z());
			glEnd();

			glBegin(GL_POINTS);
			glColor3f(1.0f, 1.0f, 1.0f);
			//glVertex3f(it->a.x(),it->a.y(),it->a.z());
			//glVertex3f(it->a.x(),it->a.y(),it->a.z()+1);

			glEnd();
		}
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
			stream<<it->normal.x()<<"\t"<<it->normal.y()<<"\t"<<it->normal.z()<<"\t";
			stream<<it->a.x()<<"\t"<<it->a.y()<<"\t"<<it->a.z()<<"\t";
			stream<<it->b.x()<<"\t"<<it->b.y()<<"\t"<<it->b.z()<<"\t";
			stream<<it->c.x()<<"\t"<<it->c.y()<<"\t"<<it->c.z()<<"\r\n";
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
		int i=0;//Triangle indexes
		for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
			QString currentVertexString;
			currentVertexString.append(QString("%1 %2 %3 ").arg(it->a.x()).arg(it->a.y()).arg(it->a.z()));
			currentVertexString.append(QString("%1 %2 %3 ").arg(it->b.x()).arg(it->b.y()).arg(it->b.z()));
			currentVertexString.append(QString("%1 %2 %3 ").arg(it->c.x()).arg(it->c.y()).arg(it->c.z()));
			QString currentNormalString;
			currentNormalString.append(QString("%1 %2 %3 ").arg(it->normal.x()).arg(it->normal.y()).arg(it->normal.z()));
			QString currentTriangleString;
			currentTriangleString.append(QString("%1 %2 %3 ").arg(i).arg(i+1).arg(i+2));

			vertexesString.append(currentVertexString);
			normalsString.append(currentNormalString+currentNormalString+currentNormalString);
			trianglesString.append(currentTriangleString);
			i+=3;
		}

		QString result=xmlTemplate;
		result.replace("##[points-array-size]##",QString("%1").arg(triangles.length()*9));
		result.replace("##[points-count]##",QString("%1").arg(triangles.length()));
		result.replace("##[normals-array-size]##",QString("%1").arg(triangles.length()*9));
		result.replace("##[normals-count]##",QString("%1").arg(triangles.length()));
		result.replace("##[triangles-count]##",QString("%1").arg(triangles.length()));

		result.replace("##[points]##",vertexesString);
		result.replace("##[normals]##",normalsString);
		result.replace("##[triangles-indexes]##",trianglesString);

		stream<<result;
		file.close();
	}
}
