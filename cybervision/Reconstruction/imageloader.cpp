#include "imageloader.h"
#include <QImageReader>
#include <QFile>
#include <stdexcept>
#include <QDebug>


namespace cybervision{
ImageLoader::ImageLoader(const QString& filename,bool loadFullImage,QObject *parent) : QObject(parent){
	QImageReader reader(filename);
	imageSize= reader.size();
	if(reader.format().toLower()=="tiff")
		metadata= extractTagStringTIFF(filename);

	if(!metadata.isEmpty())
		tags= parseTagString(metadata);

	//Parse data
	{
		double dDataBarHeight;
		QString strDataBarHeight= tags["PrivateFei.DatabarHeight"];
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
	QMultiMap<QString,QString> newTags;
	QString currentRoot="";


	QRegExp rootRegExp("\\[([^\\]]+)\\]",Qt::CaseInsensitive);
	QRegExp lineRegexp("([^ \\t=]+)\\s*=\\s*(.*)",Qt::CaseInsensitive);

	QStringList lines= metadata.split(QRegExp("(\\r\\n)|(\\n)"),QString::SkipEmptyParts);
	for(QStringList::const_iterator it=lines.begin();it!=lines.end();it++){
		QString line= *it;
		if(rootRegExp.exactMatch(*it) && rootRegExp.capturedTexts().size()>=2)
			currentRoot= rootRegExp.capturedTexts().at(1);
		else if(lineRegexp.exactMatch(*it) && lineRegexp.capturedTexts().size()>=3)
			newTags.insert(currentRoot+(currentRoot.isEmpty()?"":".")+lineRegexp.capturedTexts().at(1),lineRegexp.capturedTexts().at(2));
	}

	return newTags;
}

QString ImageLoader::extractTagStringTIFF(const QString& filename) const{
	QString result;
	QFile file(filename);

	try{
		file.open(QFile::ReadOnly);
		QDataStream in_stream(&file);

		QDataStream::ByteOrder endian;
		qint64 IFD= 0;
		{
			//Read endianess
			char buffer[8];
			qint64 read_bytes= file.read(buffer,8);
			if(read_bytes!=8)
				throw std::runtime_error("File less than 8 bytes");

			if(buffer[0]==0x49 && buffer[1]==0x49)
				endian= QDataStream::BigEndian;
			else if(buffer[0]==0x4d && buffer[1]==0x4d)
				endian= QDataStream::LittleEndian;
			else
				throw std::runtime_error("Bad byte order");
			//Check if the Answer To Life, the Universe and Everything is present
			int fortyTwo= 0;
			if (endian == QDataStream::BigEndian)
				fortyTwo= (buffer[2]&0xff) | ((buffer[3]&0xff) << 8);
			else if (endian == QDataStream::LittleEndian)
				fortyTwo= (buffer[3]&0xff) | ((buffer[2]&0xff) << 8);
			if (fortyTwo != 42)
				throw std::runtime_error("Bad version");

			//Get the first IFD
			if (endian == QDataStream::BigEndian)
				IFD = (buffer[4] & 0xff) | ((buffer[5]&0xff) << 8) | ((buffer[6]&0xff) << 16) | ((buffer[7]&0xff) << 24);
			else if (endian == QDataStream::LittleEndian)
				IFD = (buffer[7] & 0xff) | ((buffer[6]&0xff) << 8) | ((buffer[5]&0xff) << 16) | ((buffer[4]&0xff) << 24);
			if(IFD==0)
				throw std::runtime_error("No initial IFD");
		}

		//Process IFDs
		while(IFD!=0){
			char buffer[12];
			file.seek(IFD);
			//Get the number of IFDs
			if(file.read(buffer, 2)!=2) throw std::runtime_error("Unexpected end of file");
			qint64 numIFDs=0;
			if (endian == QDataStream::BigEndian)
				numIFDs = (buffer[0]&0xff) | ((buffer[1]&0xff)<<8);
			else if (endian == QDataStream::LittleEndian)
				numIFDs = (buffer[1]&0xff) | ((buffer[0]&0xff)<<8);
			if (numIFDs < 1)
				throw std::runtime_error("Not enough IFDs");

			//Read IFDs
			for (qint64 i = 0; i < numIFDs; i++) {
				//Read an IFD
				if(file.read(buffer,12)!=12) throw std::runtime_error("Unexpected end of file");
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
					result= QString::fromAscii(data.data(),data.size());

				file.seek(fp);
			}

			//Get the next IFD
			file.read(buffer,4);
			if (endian == QDataStream::BigEndian)
				IFD = (buffer[0] & 0xff) | ((buffer[1]&0xff) << 8) | ((buffer[2]&0xff) << 16) | ((buffer[3]&0xff) << 24);
			else if (endian == QDataStream::LittleEndian)
				IFD = (buffer[3] & 0xff) | ((buffer[2]&0xff) << 8) | ((buffer[1]&0xff) << 16) | ((buffer[0]&0xff) << 24);
		}
	}catch(...){
		file.close();
		return "";//TODO: log error
	}

	file.close();
	return result;
}
}
