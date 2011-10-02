#include <QGLWidget>
#include <QTextStream>

#include "surface.h"
#include "options.h"
#include <cmath>

namespace cybervision{
	Surface::Surface(){
		scale= 1.0;
		medianDepth= 0,minDepth= 0,maxDepth= 0;
	}

	Surface::Surface(const Surface&sc){
		operator =(sc);
	}

	void Surface::operator =(const Surface&sc){
		this->triangles= sc.triangles;
		this->points= sc.points;
		this->scale= sc.scale;
		this->medianDepth= sc.medianDepth;
		this->minDepth= sc.minDepth;
		this->maxDepth= sc.maxDepth;
		this->imageSize= sc.imageSize;
		this->image1= sc.image1;
		this->image2= sc.image2;
	}

	void Surface::glDraw() const{

		GLfloat backup_mat_specular[4];
		GLfloat backup_mat_shininess[1];

		if(cybervision::Options::renderShiny){
			glGetMaterialfv(GL_FRONT, GL_SPECULAR, backup_mat_specular);
			glGetMaterialfv(GL_FRONT, GL_SHININESS, backup_mat_shininess);
			static GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
			static GLfloat mat_shininess[] = { 50.0 };
			glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
			glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
		}

		for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
			const Point& pa=points[it->a];
			const Point& pb=points[it->b];
			const Point& pc=points[it->c];

			glBegin(GL_TRIANGLES);
			glColor3f(1.0f, 1.0f, 1.0f);

			if(Options::renderNormalsMode== Options::RENDER_NORMALS_TRIANGLE){
				//Use triangle normals
				//Front side
				glNormal3f(it->normal.x(),it->normal.y(),it->normal.z());
				glTexCoord2d(pa.uv.x(),pa.uv.y());
				glVertex3f(pa.coord.x()*scale,pa.coord.y()*scale,pa.coord.z()*scale);
				glNormal3f(it->normal.x(),it->normal.y(),it->normal.z());
				glTexCoord2d(pb.uv.x(),pb.uv.y());
				glVertex3f(pb.coord.x()*scale,pb.coord.y()*scale,pb.coord.z()*scale);
				glNormal3f(it->normal.x(),it->normal.y(),it->normal.z());
				glTexCoord2d(pc.uv.x(),pc.uv.y());
				glVertex3f(pc.coord.x()*scale,pc.coord.y()*scale,pc.coord.z()*scale);

				//Back side
				glNormal3f(-it->normal.x(),-it->normal.y(),-it->normal.z());
				glTexCoord2d(pc.uv.x(),pc.uv.y());
				glVertex3f(pc.coord.x()*scale,pc.coord.y()*scale,pc.coord.z()*scale);
				glNormal3f(-it->normal.x(),-it->normal.y(),-it->normal.z());
				glTexCoord2d(pb.uv.x(),pb.uv.y());
				glVertex3f(pb.coord.x()*scale,pb.coord.y()*scale,pb.coord.z()*scale);
				glNormal3f(-it->normal.x(),-it->normal.y(),-it->normal.z());
				glTexCoord2d(pa.uv.x(),pa.uv.y());
				glVertex3f(pa.coord.x()*scale,pa.coord.y()*scale,pa.coord.z()*scale);
			}else if(Options::renderNormalsMode== Options::RENDER_NORMALS_POINT){
				//Use point normals
				//Front side
				glNormal3f(pa.normal.x(),pa.normal.y(),pa.normal.z());
				glTexCoord2d(pa.uv.x(),pa.uv.y());
				glVertex3f(pa.coord.x()*scale,pa.coord.y()*scale,pa.coord.z()*scale);
				glNormal3f(pb.normal.x(),pb.normal.y(),pb.normal.z());
				glTexCoord2d(pb.uv.x(),pb.uv.y());
				glVertex3f(pb.coord.x()*scale,pb.coord.y()*scale,pb.coord.z()*scale);
				glNormal3f(pc.normal.x(),pc.normal.y(),pc.normal.z());
				glTexCoord2d(pc.uv.x(),pc.uv.y());
				glVertex3f(pc.coord.x()*scale,pc.coord.y()*scale,pc.coord.z()*scale);

				//Back side
				glNormal3f(-pc.normal.x(),-pc.normal.y(),-pc.normal.z());
				glTexCoord2d(pc.uv.x(),pc.uv.y());
				glVertex3f(pc.coord.x()*scale,pc.coord.y()*scale,pc.coord.z()*scale);
				glNormal3f(-pb.normal.x(),-pb.normal.y(),-pb.normal.z());
				glTexCoord2d(pb.uv.x(),pb.uv.y());
				glVertex3f(pb.coord.x()*scale,pb.coord.y()*scale,pb.coord.z()*scale);
				glNormal3f(-pa.normal.x(),-pa.normal.y(),-pa.normal.z());
				glTexCoord2d(pa.uv.x(),pa.uv.y());
				glVertex3f(pa.coord.x()*scale,pa.coord.y()*scale,pa.coord.z()*scale);
			}
			//glBegin(GL_POINTS);
			//glColor3f(0.0f, 0.0f, 0.0f);
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
				glVertex3f(it->x()*scale+cos(angle)*radius,it->y()*scale+sin(angle)*radius,it->z()*scale);
			}
			glEnd();
		}
		*/

		if(cybervision::Options::renderShiny){
			glMaterialfv(GL_FRONT, GL_SPECULAR, backup_mat_specular);
			glMaterialfv(GL_FRONT, GL_SHININESS, backup_mat_shininess);
		}
	}

	bool Surface::isOk()const{ return !triangles.empty() && !points.empty() && scale>0.0; }

	qreal Surface::getMedianDepth()const{ return medianDepth; }
	qreal Surface::getMinDepth()const{ return minDepth; }
	qreal Surface::getMaxDepth()const{ return maxDepth; }
	QRectF cybervision::Surface::getImageSize() const{ return imageSize; }

	qreal Surface::getScale()const{ return scale; }


	const QImage& Surface::getTexture1()const{ return image1; }
	const QImage& Surface::getTexture2()const{ return image2; }
	void Surface::setTextures(const QImage& image1,const QImage& image2){
		this->image1= image1, this->image2= image2;
	}


	/*
	 Functions for saving image data
	 */
	void Surface::savePoints(QString fileName)const{
		QFile file(fileName);
		file.open(QIODevice::WriteOnly);

		QTextStream stream(&file);
		for(QList<Point>::const_iterator it= points.begin();it!=points.end();it++)
			stream<<it->coord.x()<<"\t"<<it->coord.y()<<"\t"<<it->coord.z()<<"\r\n";

		file.close();
	}
	void Surface::savePolygons(QString fileName)const{
		QFile file(fileName);
		file.open(QIODevice::WriteOnly);

		QTextStream stream(&file);
		for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
			const QVector3D& pa=points[it->a].coord;
			const QVector3D& pb=points[it->b].coord;
			const QVector3D& pc=points[it->c].coord;
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

		QString vertexesString,normalsString,textureCoordinatesString,trianglesString;

		QString result=xmlTemplate;
		if(Options::colladaFormat==Options::COLLADA_SHARED_POINTS){
			//Output points and normals
			for(QList<Point>::const_iterator it= points.begin();it!=points.end();it++){
				QString currentVertexString;
				currentVertexString.append(QString("%1 %2 %3 ").arg(it->coord.x()).arg(it->coord.y()).arg(it->coord.z()));
				QString currentNormalString;
				currentNormalString.append(QString("%1 %2 %3 ").arg(it->normal.x()).arg(it->normal.y()).arg(it->normal.z()));
				QString currentTextureCoordinatesString;
				currentTextureCoordinatesString.append(QString("%1 %2 ").arg(it->uv.x()).arg(it->uv.y()));

				vertexesString.append(currentVertexString);
				normalsString.append(currentNormalString);
				textureCoordinatesString.append(currentTextureCoordinatesString);
			}

			//Output polygons
			for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
				QString currentTriangleString;
				currentTriangleString.append(QString("%1 %2 %3 ").arg(it->a).arg(it->b).arg(it->c));

				trianglesString.append(currentTriangleString);
			}

			result.replace("##[points-array-size]##",QString("%1").arg(points.length()*3));
			result.replace("##[points-count]##",QString("%1").arg(points.length()));
			result.replace("##[normals-array-size]##",QString("%1").arg(points.length()*3));
			result.replace("##[normals-count]##",QString("%1").arg(triangles.length()));
			result.replace("##[texture-coordinates-array-size]##",QString("%1").arg(points.length()*2));
			result.replace("##[texture-coordinates-count]##",QString("%1").arg(points.length()));
			result.replace("##[triangles-count]##",QString("%1").arg(triangles.length()));
		}else if(Options::colladaFormat==Options::COLLADA_INDEPENDENT_POLYGONS){
			PolygonPoint i=0;//Triangle indexes
			for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
				const Point& pa=points[it->a];
				const Point& pb=points[it->b];
				const Point& pc=points[it->c];
				QString currentVertexString;
				currentVertexString.append(QString("%1 %2 %3 ").arg(pa.coord.x()).arg(pa.coord.y()).arg(pa.coord.z()));
				currentVertexString.append(QString("%1 %2 %3 ").arg(pb.coord.x()).arg(pb.coord.y()).arg(pb.coord.z()));
				currentVertexString.append(QString("%1 %2 %3 ").arg(pc.coord.x()).arg(pc.coord.y()).arg(pc.coord.z()));
				QString currentNormalString;
				currentNormalString.append(QString("%1 %2 %3 ").arg(pa.normal.x()).arg(pa.normal.y()).arg(pa.normal.z()));
				currentNormalString.append(QString("%1 %2 %3 ").arg(pb.normal.x()).arg(pb.normal.y()).arg(pb.normal.z()));
				currentNormalString.append(QString("%1 %2 %3 ").arg(pc.normal.x()).arg(pc.normal.y()).arg(pc.normal.z()));
				QString currentTextureCoordinatesString;
				currentTextureCoordinatesString.append(QString("%1 %2 ").arg(pa.uv.x()).arg(pa.uv.y()));
				currentTextureCoordinatesString.append(QString("%1 %2 ").arg(pb.uv.x()).arg(pb.uv.y()));
				currentTextureCoordinatesString.append(QString("%1 %2 ").arg(pc.uv.x()).arg(pc.uv.y()));
				QString currentTriangleString;
				currentTriangleString.append(QString("%1 %2 %3 ").arg(i).arg(i+1).arg(i+2));

				vertexesString.append(currentVertexString);
				normalsString.append(currentNormalString);
				textureCoordinatesString.append(currentTextureCoordinatesString);
				trianglesString.append(currentTriangleString);
				i+=3;
			}

			result.replace("##[points-array-size]##",QString("%1").arg(triangles.length()*9));
			result.replace("##[points-count]##",QString("%1").arg(triangles.length()*3));
			result.replace("##[normals-array-size]##",QString("%1").arg(triangles.length()*9));
			result.replace("##[normals-count]##",QString("%1").arg(triangles.length()*3));
			result.replace("##[texture-coordinates-array-size]##",QString("%1").arg(triangles.length()*6));
			result.replace("##[texture-coordinates-count]##",QString("%1").arg(triangles.length()*3));
			result.replace("##[triangles-count]##",QString("%1").arg(triangles.length()));
		}

		QString fileNameTexture= fileName+".png";

		result.replace("##[points]##",vertexesString);
		result.replace("##[normals]##",normalsString);
		result.replace("##[texture-coordinates]##",textureCoordinatesString);
		result.replace("##[triangles-indexes]##",trianglesString);
		result.replace("##[texture-image-filename]##",QFileInfo(fileNameTexture).fileName());

		if(!image1.save(fileNameTexture,"png"))
			;//TODO:error

		stream<<result;
		file.close();
	}
}
