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
		QFile file(fileName);
		file.open(QIODevice::WriteOnly);

		QTextStream stream(&file);

		for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
			//TODO: write file here
		}

		file.close();
	}
}
