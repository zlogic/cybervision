#include "imageloader.h"
#include <QImageReader>
#include <QFile>
#include <QDebug>


namespace cybervision{
ImageLoader::ImageLoader(const QString& filename,bool loadFullImage,QObject *parent) : QObject(parent){
	QImageReader reader(filename);
	imageSize= reader.size();
	if(reader.format().toLower()=="tiff")
		metadata= extractTagStringTIFF(filename);

	if(!metadata.isEmpty())
		quantaTags= parseTagString(metadata);

	//Parse data
	{
		double dDataBarHeight;
		QString strDataBarHeight= quantaTags["PrivateFei.DatabarHeight"];
		bool ok;
		dDataBarHeight= strDataBarHeight.toDouble(&ok);
		if(ok){
			imageSize.setHeight(imageSize.height()-dDataBarHeight);
			reader.setClipRect(QRect(0,0,imageSize.width(),imageSize.height()));
		}

	}

	if(loadFullImage)
		img= reader.read();
}

const QSize& ImageLoader::getSize() const{
	return imageSize;
}

const QImage& ImageLoader::getImage() const{
	return img;
}

const QString& ImageLoader::getMetadataString() const{
	return metadata;
}

QMap<QString,QString> ImageLoader::parseTagString(const QString& metadata)const{
	QMap<QString,QString> newTags;
	QString currentRoot="";


	QRegularExpression rootRegExp("^\\[([^\\]]+)\\]$",QRegularExpression::CaseInsensitiveOption);
	QRegularExpression lineRegexp("^([^ \\t=]+)\\s*=\\s*(.*)$",QRegularExpression::CaseInsensitiveOption);

	QStringList lines= metadata.split(QRegularExpression("(\\r\\n)|(\\n)"),Qt::SkipEmptyParts);
	for(QStringList::const_iterator it=lines.constBegin();it!=lines.constEnd();it++){
		QString line= *it;
		QRegularExpressionMatch rootMatch= rootRegExp.match(line);
		QRegularExpressionMatch lineMatch= lineRegexp.match(line);
		if(rootMatch.hasMatch() && rootMatch.capturedTexts().size()>=2)
			currentRoot= rootMatch.capturedTexts().at(1);
		else if(lineMatch.hasMatch() && lineMatch.capturedTexts().size()>=3)
			newTags.insert(currentRoot+(currentRoot.isEmpty()?"":".")+lineMatch.capturedTexts().at(1),lineMatch.capturedTexts().at(2));
	}

	return newTags;
}

QString ImageLoader::extractTagStringTIFF(const QString& filename) const{
	QString result;
	QFile file(filename);

	file.open(QFile::ReadOnly);
	//QDataStream in_stream(&file);

	QDataStream::ByteOrder endian;
	qint64 IFD= 0;
	{
		//Read endianess
		char buffer[8];
		qint64 read_bytes= file.read(buffer,8);
		if(read_bytes!=8){
			//QString error(tr("File less than 8 bytes"));
			file.close();
			return "";//TODO: log error
		}

		if(buffer[0]==0x49 && buffer[1]==0x49)
			endian= QDataStream::BigEndian;
		else if(buffer[0]==0x4d && buffer[1]==0x4d)
			endian= QDataStream::LittleEndian;
		else{
			//QString error(tr("Bad byte order"));
			file.close();
			return "";//TODO: log error
		}
		//Check if the Answer To Life, the Universe and Everything is present
		int fortyTwo= 0;
		if (endian == QDataStream::BigEndian)
			fortyTwo= (buffer[2]&0xff) | ((buffer[3]&0xff) << 8);
		else if (endian == QDataStream::LittleEndian)
			fortyTwo= (buffer[3]&0xff) | ((buffer[2]&0xff) << 8);
		if (fortyTwo != 42){
			//QString error(tr("Bad version"));
			file.close();
			return "";//TODO: log error
		}

		//Get the first IFD
		if (endian == QDataStream::BigEndian)
			IFD = (buffer[4] & 0xff) | ((buffer[5]&0xff) << 8) | ((buffer[6]&0xff) << 16) | ((buffer[7]&0xff) << 24);
		else if (endian == QDataStream::LittleEndian)
			IFD = (buffer[7] & 0xff) | ((buffer[6]&0xff) << 8) | ((buffer[5]&0xff) << 16) | ((buffer[4]&0xff) << 24);
		if(IFD==0){
			//QString error(tr("No initial IFD"));
			file.close();
			return "";//TODO: log error
		}
	}

	//Process IFDs
	while(IFD!=0){
		char buffer[12];
		file.seek(IFD);
		//Get the number of IFDs
		if(file.read(buffer, 2)!=2){
			//QString error(tr("Unexpected end of file"));
			file.close();
			return "";//TODO: log error
		}
		qint64 numIFDs=0;
		if (endian == QDataStream::BigEndian)
			numIFDs = (buffer[0]&0xff) | ((buffer[1]&0xff)<<8);
		else if (endian == QDataStream::LittleEndian)
			numIFDs = (buffer[1]&0xff) | ((buffer[0]&0xff)<<8);
		if (numIFDs < 1){
			//QString error(tr("Not enough IFDs"));
			file.close();
			return "";//TODO: log error
		}

		//Read IFDs
		for (qint64 i = 0; i < numIFDs; i++) {
			//Read an IFD
			if(file.read(buffer,12)!=12){
				//QString error(tr("Unexpected end of file"));
				file.close();
				return "";//TODO: log error
			}
			quint16 tag_short=0,type_short=0;
			qint64 count=0,offset=0;
			if(endian == QDataStream::BigEndian){
				tag_short= (buffer[0]&0xff) | ((buffer[1]&0xff)<<8);
				type_short= (buffer[2]&0xff) | ((buffer[3]&0xff)<<8);
				count= (buffer[4] & 0xff) | ((buffer[5]&0xff) << 8) | ((buffer[6]&0xff) << 16) | ((buffer[7]&0xff) << 24);
				offset= (buffer[8] & 0xff) | ((buffer[9]&0xff) << 8) | ((buffer[10]&0xff) << 16) | ((buffer[11]&0xff) << 24);
			}
			else if(endian == QDataStream::LittleEndian){
				tag_short= (buffer[1]&0xff) | ((buffer[0]&0xff)<<8);
				type_short= (buffer[2]&0xff) | ((buffer[2]&0xff)<<8);
				count= (buffer[7] & 0xff) | ((buffer[6]&0xff) << 8) | ((buffer[5]&0xff) << 16) | ((buffer[4]&0xff) << 24);
				offset= (buffer[11] & 0xff) | ((buffer[10]&0xff) << 8) | ((buffer[9]&0xff) << 16) | ((buffer[8]&0xff) << 24);
			}

			//Read the value
			qint64 elementSize;
			switch(type_short){
			case 1: elementSize=1;break;//BYTE
			case 2: elementSize=1;break;//ASCII
			case 3: elementSize=2;break;//SHORT
			case 4: elementSize=4;break;//LONG
			case 5: elementSize=8;break;//LONG
			default: elementSize=0; break;
			}

			if(elementSize==0)
				continue;

			qint64 fp= file.pos();

			QVector<char> data(count*elementSize);

			file.seek(offset);
			file.read(data.data(),data.size());


			if(tag_short==TIFFTAG_META_PHENOM || tag_short== TIFFTAG_META_QUANTA)
				result= QString::fromLatin1(data.data(),data.size());

			file.seek(fp);
		}

		//Get the next IFD
		file.read(buffer,4);
		if (endian == QDataStream::BigEndian)
			IFD = (buffer[0] & 0xff) | ((buffer[1]&0xff) << 8) | ((buffer[2]&0xff) << 16) | ((buffer[3]&0xff) << 24);
		else if (endian == QDataStream::LittleEndian)
			IFD = (buffer[3] & 0xff) | ((buffer[2]&0xff) << 8) | ((buffer[1]&0xff) << 16) | ((buffer[0]&0xff) << 24);
	}

	file.close();
	return result;
}
}

double cybervision::ImageLoader::getScale() const{
	if(quantaTags.contains("Scan.PixelWidth")){
		QString pixelWidthStr(quantaTags.value("Scan.PixelWidth"));
		bool ok;
		double result= pixelWidthStr.toDouble(&ok);
		if(ok)
			return result/* *10e5 */;
		else
			return -1;
	} else
		return -1;
}
