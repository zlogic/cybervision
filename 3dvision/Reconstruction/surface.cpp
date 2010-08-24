#include <QGLWidget>

#include "surface.h"

namespace cybervision{
	Surface::Surface(){

	}

	Surface::Surface(const Surface&sc){
		operator =(sc);
	}

	void Surface::operator =(const Surface&sc){
		this->triangles= sc.triangles;
	}

	void Surface::glDraw() const{
		/*
		glBegin(GL_TRIANGLES);
		for(QList<QVector3D>::const_iterator it= points.begin();it!=points.end();it++){
			glVertex3f(it->x(),it->y(),it->z()*5e6F);
		}
		glEnd();
		*/
		for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
			glBegin(GL_TRIANGLES);
			glColor3f(1.0f, 1.0f, 1.0f);
			glNormal3f(it->normal.x(),it->normal.y(),it->normal.z());
			glVertex3f(it->a.x(),it->a.y(),it->a.z());
			glNormal3f(it->normal.x(),it->normal.y(),it->normal.z());
			glVertex3f(it->b.x(),it->b.y(),it->b.z());
			glNormal3f(it->normal.x(),it->normal.y(),it->normal.z());
			glVertex3f(it->c.x(),it->c.y(),it->c.z());

			glEnd();

			glBegin(GL_POINTS);
			glColor3f(1.0f, 1.0f, 1.0f);
			//glVertex3f(it->a.x(),it->a.y(),it->a.z());
			//glVertex3f(it->a.x(),it->a.y(),it->a.z()+1);

			glEnd();
		}
	}
}
