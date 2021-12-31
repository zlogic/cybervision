#ifndef IMAGELOADER_H
#define IMAGELOADER_H

#include <QObject>
#include <QSize>
#include <QImage>
#include <QMap>

namespace cybervision{
/*
 * Class for loading an image and associated metadata. Called by PointMatcher.
 */
class ImageLoader : public QObject
{
	Q_OBJECT
protected:
	//Static data
	static const qint32 TIFFTAG_META_PHENOM = 34683;
	static const qint32 TIFFTAG_META_QUANTA = 34682;

	QSize imageSize;
	QImage img;
	QString metadata;
	QMap<QString,QString> quantaTags;

	//Parse INI file
	QMap<QString,QString> parseTagString(const QString& metadata)const;

	//Extract tags from image
	QString extractTagStringTIFF(const QString& filename)const;
public:
	explicit ImageLoader(const QString& filename,bool loadFullImage=true,QObject *parent = 0);
	const QSize& getSize() const;
	const QImage& getImage() const;
	const QString& getMetadataString() const;

	//Returns image scale or -1 if scale cannot be extracted
	double getScale()const;
signals:

public slots:

};

}
#endif // IMAGELOADER_H
