#include "inspector.h"

#include <QMultiMap>
#include <QPainter>

#include <limits>
#include <cmath>

cybervision::Inspector::Inspector(const cybervision::Surface&surface, QObject *parent):QObject(parent),surface(surface) {
	mA= std::numeric_limits<qreal>::quiet_NaN(), mB= std::numeric_limits<qreal>::quiet_NaN(), mL= std::numeric_limits<qreal>::quiet_NaN();
	Ra= std::numeric_limits<qreal>::quiet_NaN(), Rz= std::numeric_limits<qreal>::quiet_NaN(), Rmax= std::numeric_limits<qreal>::quiet_NaN();
	S= std::numeric_limits<qreal>::quiet_NaN(),  Sm= std::numeric_limits<qreal>::quiet_NaN(), tp= std::numeric_limits<qreal>::quiet_NaN();
}

void cybervision::Inspector::updateCrossSection(const QVector3D &start, const QVector3D &end,int p){
	mA= std::numeric_limits<qreal>::quiet_NaN(), mB= std::numeric_limits<qreal>::quiet_NaN(), mL= std::numeric_limits<qreal>::quiet_NaN();
	Ra= std::numeric_limits<qreal>::quiet_NaN(), Rz= std::numeric_limits<qreal>::quiet_NaN(), Rmax= std::numeric_limits<qreal>::quiet_NaN();
	S= std::numeric_limits<qreal>::quiet_NaN(),  Sm= std::numeric_limits<qreal>::quiet_NaN(), tp= std::numeric_limits<qreal>::quiet_NaN();

	QMultiMap<qreal,qreal> intersections;
	QLineF intersectionLine(start.x(),start.y(),end.x(),end.y());

	for(QList<Surface::Triangle>::const_iterator it= surface.triangles.begin();it!=surface.triangles.end();it++){
		QLineF lineAB(surface.points[it->a].coord.x(),surface.points[it->a].coord.y(),surface.points[it->b].coord.x(),surface.points[it->b].coord.y());
		QLineF lineBC(surface.points[it->b].coord.x(),surface.points[it->b].coord.y(),surface.points[it->c].coord.x(),surface.points[it->c].coord.y());
		QLineF lineCA(surface.points[it->c].coord.x(),surface.points[it->c].coord.y(),surface.points[it->a].coord.x(),surface.points[it->a].coord.y());
		QPointF intersectionPoint;
		if(lineAB.intersect(intersectionLine,&intersectionPoint)==QLineF::BoundedIntersection){
			qreal tLine= qAbs(intersectionLine.dx())>qAbs(intersectionLine.dy())?
						(intersectionPoint.x()-intersectionLine.x1())/(intersectionLine.x2()-intersectionLine.x1()):
						(intersectionPoint.y()-intersectionLine.y1())/(intersectionLine.y2()-intersectionLine.y1());
			qreal tPolygon= qAbs(lineAB.dx())>qAbs(lineAB.dy())?
						(intersectionPoint.x()-lineAB.x1())/(lineAB.x2()-(lineAB.x1())):
						(intersectionPoint.y()-lineAB.y1())/(lineAB.y2()-(lineAB.y1()));
			qreal z= surface.points[it->a].coord.z()*tPolygon+surface.points[it->b].coord.z()*(1-tPolygon);
			intersections.insert(tLine,z);
		}
		if(lineBC.intersect(intersectionLine,&intersectionPoint)==QLineF::BoundedIntersection){
			qreal tLine= qAbs(intersectionLine.dx())>qAbs(intersectionLine.dy())?
						(intersectionPoint.x()-intersectionLine.x1())/(intersectionLine.x2()-intersectionLine.x1()):
						(intersectionPoint.y()-intersectionLine.y1())/(intersectionLine.y2()-intersectionLine.y1());
			qreal tPolygon= qAbs(lineBC.dx())>qAbs(lineBC.dy())?
						(intersectionPoint.x()-lineBC.x1())/(lineBC.x2()-(lineBC.x1())):
						(intersectionPoint.y()-lineBC.y1())/(lineBC.y2()-(lineBC.y1()));
			qreal z= surface.points[it->b].coord.z()*tPolygon+surface.points[it->c].coord.z()*(1-tPolygon);
			intersections.insert(tLine,z);
		}
		if(lineCA.intersect(intersectionLine,&intersectionPoint)==QLineF::BoundedIntersection){
			qreal tLine= qAbs(intersectionLine.dx())>qAbs(intersectionLine.dy())?
						(intersectionPoint.x()-intersectionLine.x1())/(intersectionLine.x2()-intersectionLine.x1()):
						(intersectionPoint.y()-intersectionLine.y1())/(intersectionLine.y2()-intersectionLine.y1());
			qreal tPolygon= qAbs(lineCA.dx())>qAbs(lineCA.dy())?
						(intersectionPoint.x()-lineCA.x1())/(lineCA.x2()-(lineCA.x1())):
						(intersectionPoint.y()-lineCA.y1())/(lineCA.y2()-(lineCA.y1()));
			qreal z= surface.points[it->c].coord.z()*tPolygon+surface.points[it->a].coord.z()*(1-tPolygon);
			intersections.insert(tLine,z);
		}
	}
	qreal sum=0;
	int count=0;
	qreal lineLength= intersectionLine.length();

	for(QMap<qreal,qreal>::const_iterator it=intersections.begin();it!=intersections.end();it++){
		sum+= it.value();
		count++;
		if((it+1)==intersections.end() || !qFuzzyCompare(it.key(),(it+1).key())){
			qreal z= sum/(qreal)count;
			QPointF point(it.key()*lineLength,z);
			crossSection.push_back(point);
			sum= 0;
			count= 0;
		}
	}

	if(crossSection.size()>0)
		computeParams(p);
}

void cybervision::Inspector::computeParams(int p){
	QLineF mLine,mLineNormal;
	{
		//Compute middle-line with least-squares
		//See http://mathworld.wolfram.com/LeastSquaresFitting.html
		qreal xAverage=0,yAverage=0;
		qreal xSquares=0,ySquares=0;
		qreal xy=0;
		for(QList<QPointF>::const_iterator it=crossSection.begin();it!=crossSection.end();it++){
			xAverage+= it->x();
			yAverage+= it->y();
			xSquares+= it->x()*it->x();
			ySquares+= it->y()*it->y();
			xy+= it->x()*it->y();
		}
		xAverage/= (qreal)crossSection.size();
		yAverage/= (qreal)crossSection.size();
		//Compute a and b
		qreal ssXX= xSquares-(qreal)crossSection.size()*xAverage*xAverage;
		qreal ssXY= xy-(qreal)crossSection.size()*xAverage*yAverage;
		mB= ssXY/ssXX;
		mA= yAverage-mB*xAverage;

		qreal minX= std::numeric_limits<qreal>::infinity(),
				maxX= -std::numeric_limits<qreal>::infinity();
		for(QList<QPointF>::const_iterator it=crossSection.begin();it!=crossSection.end();it++){
			minX= qMin(minX,it->x());
			maxX= qMax(maxX,it->x());
		}
		qreal deltaX= (maxX-minX);
		mL= sqrt(deltaX*deltaX*mB*mB+deltaX*deltaX);
		mLine.setPoints(QPointF(minX,mB*minX+mA),QPointF(maxX,mB*maxX+mA));
		mLineNormal= mLine.normalVector();
	}

	//Project points onto m-line
	QList<QPointF> crossSectionProjected;
	{
		QMultiMap<qreal,qreal> crossSectionSorted;
		for(QList<QPointF>::const_iterator it=crossSection.begin();it!=crossSection.end();it++){
			//Create a line going through the point, parallel to m-line
			qreal deltaY= it->y()-it->x()*mB-mA;
			QLineF pointLine= mLine;
			pointLine.translate(0,deltaY);
			//Intersect the line normal with the parallel line
			QPointF intersectionPoint;
			mLineNormal.intersect(pointLine,&intersectionPoint);
			QLineF lineY(mLineNormal.p1(),intersectionPoint),lineX(intersectionPoint,*it);
			qreal pointY= lineY.length()*(qAbs(lineY.angleTo(mLineNormal))<90?1:-1);
			qreal pointX= lineX.length()*(qAbs(lineX.angleTo(pointLine))<90?1:-1);

			crossSectionSorted.insert(pointX,pointY);
		}

		qreal sum=0;
		int count=0;
		for(QMap<qreal,qreal>::const_iterator it=crossSectionSorted.begin();it!=crossSectionSorted.end();it++){
			sum+= it.value();
			count++;
			if((it+1)==crossSectionSorted.end() || !qFuzzyCompare(it.key(),(it+1).key())){
				qreal Y= sum/(qreal)count;
				QPointF point(it.key(),Y);
				crossSectionProjected.push_back(point);
				sum= 0;
				count= 0;
			}
		}
	}

	//Compute height and step parameters
	qreal maxHeight= -std::numeric_limits<qreal>::infinity(),minHeight= std::numeric_limits<qreal>::infinity();//for Rz and tp
	{
		Ra= 0;
		QList<qreal> maxHeights,minHeights;
		bool lastHeightPositive= true;
		qreal peakHeight= 0;
		qreal peakX= std::numeric_limits<qreal>::quiet_NaN(),lastPeakX= std::numeric_limits<qreal>::quiet_NaN();//for S
		qreal lastCrossingX= std::numeric_limits<qreal>::quiet_NaN();
		S= 0;
		Sm= 0;
		int SPeakCount= 0;//for S
		int SCrossingCount= 0;//for Sm
		for(QList<QPointF>::const_iterator it=crossSectionProjected.begin();it!=crossSectionProjected.end();it++){
			qreal pointX=it->x(), pointY=it->y();

			qreal height= pointY;

			Ra+= qAbs(height);

			if(it!=crossSectionProjected.begin()){
				if(lastHeightPositive && (height<0 || (it+1)==crossSectionProjected.end())){
					//Rz
					maxHeights<<qAbs(peakHeight);
					qSort(maxHeights);
					if(maxHeights.size()>5)
						maxHeights.pop_front();
					peakHeight= 0;
					//S
					if(!std::isnan(lastPeakX) && !std::isnan(peakX)){
						SPeakCount++;
						S+= peakX-lastPeakX;
					}
					lastPeakX= peakX;
					peakX= std::numeric_limits<qreal>::quiet_NaN();
					//Sm
					if(!std::isnan(lastCrossingX)){
						Sm+= pointX-lastCrossingX;
						SCrossingCount++;
					}
					lastCrossingX= pointX;
				}
				if(!lastHeightPositive && (height>0 || (it+1)==crossSectionProjected.end())){
					//Rz
					minHeights<<qAbs(peakHeight);
					qSort(minHeights);
					if(minHeights.size()>5)
						minHeights.pop_back();
					peakHeight= 0;
				}
			}
			lastHeightPositive= height>0;
			//S
			if(std::isnan(peakX) && height>0)
				peakX= pointX;
			peakX= (height>peakHeight && height>0)?pointX:peakX;
			//Rz
			peakHeight= qAbs(height)>qAbs(peakHeight)?height:peakHeight;
			//Rmax
			maxHeight= qMax(maxHeight,height);
			minHeight= qMin(minHeight,height);
		}
		//Ra
		Ra/= (qreal)crossSectionProjected.size();
		//Rmax
		Rmax= maxHeight-minHeight;
		//Rz
		if(crossSectionProjected.size()<5 || maxHeights.size()<5 || minHeights.size()<5)
			Rz= std::numeric_limits<qreal>::quiet_NaN();
		else{
			Rz= 0;
			for(int i=0;i<qMin(maxHeights.size(),minHeights.size());i++)
				Rz+= maxHeights[i]+minHeights[i];
			Rz/= 5.0;
		}
		//S
		if(SPeakCount<1)
			S= std::numeric_limits<qreal>::quiet_NaN();
		else
			S/= (qreal)SPeakCount;
		//Sm
		if(SCrossingCount<1)
			Sm= std::numeric_limits<qreal>::quiet_NaN();
		else
			Sm/= (qreal)SCrossingCount;
		//tp
		tp= p;
	}
	//Compute tp
	{
		bool lastHeightPositive= true;
		tp= 0;
		qreal upX= std::numeric_limits<qreal>::quiet_NaN();//Crossing the p-line up (from negative to positive y)
		int tpCount=0;
		for(QList<QPointF>::const_iterator it=crossSectionProjected.begin();it!=crossSectionProjected.end();it++){
			qreal pointX=it->x(), height=it->y();
			if(it!=crossSectionProjected.begin()){
				if(lastHeightPositive && (height<0 || (it+1)==crossSectionProjected.end())){
					if(!std::isnan(upX)){
						tp+= pointX-upX;
						tpCount++;
					}
					upX= std::numeric_limits<qreal>::quiet_NaN();
				}
				if(!lastHeightPositive && (height>0 || (it+1)==crossSectionProjected.end())){
					upX= pointX;
				}
			}
			lastHeightPositive= (height-minHeight)/(maxHeight-minHeight)>((qreal)p/100.0);
		}

		if(tpCount<1)
			tp= std::numeric_limits<qreal>::quiet_NaN();
		else
			tp/= (qreal)tpCount;
		//tp= tpCount;
	}
}

QImage cybervision::Inspector::renderCrossSection(const QSize& imageSize) const{
	//Prepare data
	qreal minX= std::numeric_limits<qreal>::infinity(),
			minY= std::numeric_limits<qreal>::infinity(),
			maxX= -std::numeric_limits<qreal>::infinity(),
			maxY= -std::numeric_limits<qreal>::infinity();
	for(QList<QPointF>::const_iterator it=crossSection.begin();it!=crossSection.end();it++){
		minX= qMin(minX,it->x());
		minY= qMin(minY,it->y());
		maxX= qMax(maxX,it->x());
		maxY= qMax(maxY,it->y());
	}
	//Preapare image
	QImage img(imageSize.width(),imageSize.height(),QImage::Format_ARGB32);
	img.fill(0x00000000);

	//Draw lines
	QPainter painter(&img);
	painter.setRenderHint(QPainter::Antialiasing,true);
	painter.setPen(QColor(0xff,0x99,0x00));
	for(QList<QPointF>::const_iterator it=crossSection.begin();it!=crossSection.end();it++){
		if((it+1)==crossSection.end())
			continue;
		QPoint point1(img.width()*(    it->x()-minX)/(maxX-minX),img.height()*(maxY-    it->y())/(maxY-minY));
		QPoint point2(img.width()*((it+1)->x()-minX)/(maxX-minX),img.height()*(maxY-(it+1)->y())/(maxY-minY));

		point1.setX(qMax(point1.x(),0));
		point1.setY(qMax(point1.y(),0));
		point1.setX(qMin(point1.x(),img.width()-1));
		point1.setY(qMin(point1.y(),img.height()-1));
		point2.setX(qMax(point2.x(),0));
		point2.setY(qMax(point2.y(),0));
		point2.setX(qMin(point2.x(),img.width()-1));
		point2.setY(qMin(point2.y(),img.height()-1));

		painter.drawLine(point1,point2);
	}

	//Draw the m-line
	painter.setPen((QColor(0x66,0x66,0x66)));
	painter.drawLine(QPointF(img.width()*minX/(maxX-minX),img.height()*(maxY-minX*mB-mA)/(maxY-minY)),
					 QPointF(img.width()*maxX/(maxX-minX),img.height()*(maxY-maxX*mB-mA)/(maxY-minY))
	);
	//Display the image
	painter.end();
	return img;
}

QList<QPointF> cybervision::Inspector::getCrossSection() const{	return crossSection; }
qreal cybervision::Inspector::getRoughnessRa() const{ return Ra; }
qreal cybervision::Inspector::getRoughnessRz() const{ return Rz; }
qreal cybervision::Inspector::getRoughnessRmax() const{ return Rmax; }
qreal cybervision::Inspector::getRoughnessS() const{ return S; }
qreal cybervision::Inspector::getRoughnessSm() const{ return Sm; }
qreal cybervision::Inspector::getRoughnessTp() const{ return tp; }
