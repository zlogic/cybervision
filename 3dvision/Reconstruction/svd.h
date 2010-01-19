#ifndef SVD_H
#define SVD_H

#include <QGenericMatrix>
#include <QString>

/*
  This is a Qt-friendly implementation of the matrix SVD operation.
  Algorithm borrowed from eigen (http://eigen.tuxfamily.org/)
  and from this article: http://www.haoli.org/nr/bookcpdf/c2-6.pdf
*/

//N= number of columns
//M= number of rows
template<int N,int M,typename T> class SVD
{
protected:
	static const int max_iterations;

	QGenericMatrix<N,M,T> U;
	QGenericMatrix<N,N,T> Sigma;
	QGenericMatrix<N,N,T> V;
	QString error_str;

	void compute(const QGenericMatrix<N,M,T>& source);

	inline T pythag(T a, T b)const;
	inline T sign(T a, T b)const;
	inline T sqr(T)const;
public:
	SVD(const QGenericMatrix<N,M,T>& source);

	//Getters
	const QGenericMatrix <N,M,T>& getU()const;
	const QGenericMatrix <N,N,T>& getSigma()const;
	const QGenericMatrix <N,N,T>& getV()const;

	bool isOK()const;
	QString getErrorString()const;
};

#endif // SVD_H
