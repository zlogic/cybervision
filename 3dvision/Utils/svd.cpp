#include "svd.h"
#include <cmath>

#include <stdexcept>
#include <string>

//Exception class
class SVDError : public std::exception {
public:
	enum Reason{ERROR_NO_CONVERGENCE, ERROR_OTHER};
protected:
	const Reason reason;
	std::string reason_str;
public:
	explicit SVDError (const Reason reason):reason(reason){
		switch(reason){
  case ERROR_NO_CONVERGENCE:
			reason_str= "No convergence";
			break;
  case ERROR_OTHER:
  default:
			reason_str= "Unknown error";
		}
	}

	virtual ~SVDError() throw(){};
	virtual const char* what() const throw(){return reason_str.c_str();}
};

template<int N,int M,typename T> const  int SVD<N,M,T>::max_iterations= 30;

template<int N,int M,typename T> SVD<N,M,T>::SVD(const QGenericMatrix<N,M,T>& source){
	ok= true;
	try{
	compute(source);
	}catch(SVDError& error){
		ok=false;
	}
}

template<int N,int M,typename T> const QGenericMatrix <N,M,T>& SVD<N,M,T>::getU()const{ return U; }
template<int N,int M,typename T> const QGenericMatrix <N,N,T>& SVD<N,M,T>::getSigma()const{ return Sigma; }
template<int N,int M,typename T> const QGenericMatrix <N,N,T>& SVD<N,M,T>::getV()const{ return V; }


template<int N,int M,typename T> void SVD<N,M,T>::compute(const QGenericMatrix<N,M,T>& source){
	U=source;
	ok= false;
	QVector<T> rv1(N), W(N);

	T anorm=0.0;
	//Householder reduction to bidiagonal form
	{
		T scale=0.0, g=0.0;
		for(int i=0;i<N;i++){
			int l=i+1;
			rv1[i]= scale*g;
			g=scale=0.0;
			if(i<M){
				for(int k=i;k<M;k++)
					scale+= fabs(U(k,i));
				if(scale){
					T s=0.0;
					for(int k=i;k<M;k++){
						U(k,i)/= scale;
						s+= U(k,i)*U(k,i);
					}
					T f= U(i,i);
					g= -sign(sqrt(s),f);
					T h= f*g-s;
					U(i,i)= f-g;
					if(i!=(N-1))
						for(int j=l;j<N;j++){
							s=0.0;
							for(int k=i;k<M;k++)
								s+= U(k,i)*U(k,j);
							f=s/h;
							for(int k=i;k<M;k++)
								U(k,j)+= f*U(k,i);
						}
					for(int k=i;k<M;k++)
						U(k,i)*= scale;
				}
			}
			W[i]= scale*g;
			g=scale=0.0;
			if(i<M && i!=(N-1)){
				for(int k=l;k<N;k++)
					scale+= fabs(U(i,k));
				if(scale){
					T s=0.0;
					for(int k=l;k<N;k++){
						U(i,k)/= scale;
						s+= U(i,k)*U(i,k);
					}
					T f= U(i,l);
					g= -sign(sqrt(s),f);
					T h= f*g-s;
					U(i,l)= f-g;
					for(int k=l;k<N;k++)
						rv1[k]= U(i,k)/h;
					if(i!=(M-1))
						for(int j=l;j<M;j++){
							s=0.0;
							for(int k=l;k<N;k++)
								s+= U(j,k)*U(i,k);
							for(int k=l;k<N;k++)
								U(j,k)+= s*rv1[k];
						}
					for(int k=l;k<N;k++)
						U(i,k)*= scale;
				}
			}
			anorm= std::max(anorm,fabs(W[i])+fabs(rv1[i]));
		}
	}
	//Accumulation of right-hand transformations
	{
		T g=0.0;
		int l=N-1;
		for(int i=(N-1);i>=0;i--){
			if(i<(N-1)){
				if(g){
					for(int j=l;j<N;j++)
						V(j,i)=(U(i,j)/U(i,l))/g;
					for(int j=l;j<N;j++){
						T s=0.0;
						for(int k=l;k<N;k++)
							s+= U(i,k)*V(k,j);
						for(int k=l;k<N;k++)
							V(k,j)+= s*V(k,i);
					}
				}
				for(int j=l;j<N;j++)
					V(i,j)= V(j,i)= 0.0;
			}
			V(i,i)= 1.0;
			g= rv1[i];
			l=i;
		}
	}
	//Accumulation of left-hand transformations
	{
		for(int i=std::min(N-1,M-1);i>=0;i--){
			int l=i+1;
			T g=W[i];
			for(int j=l;j<N;j++)
				U(i,j)= 0.0;
			if(g){
				g= 1.0/g;
				for(int j=l;j<N;j++){
					T s= 0.0;
					for(int k=l;k<M;k++)
						s+= U(k,i)*U(k,j);
					T f= (s/U(i,i))*g;
					for(int k=i;k<M;k++)
						U(k,j)+= f*U(k,i);
				}
				for(int j=i;j<M;j++)
					U(j,i)*= g;
			}else
				for(int j=i;j<M;j++)
					U(j,i)= 0.0;
			U(i,i)++;
		}
	}
	//Diagonalization of the bidiagonal form: Loop over singular values, and over allowed iterations.
	{
		for(int k=N-1;k>=0;k--){
			for(int its=0;its<max_iterations;its++){
				bool flag= true;
				int l=k;
				int nm=l-1;
				for(l=k;l>=0;l--){
					//Test for splitting
					//Note that rv1[1] is always zero
					nm= l-1;
					//Zlogic: this is probably some clever "epsilon" comparison
					if((T)(fabs(rv1[l])+anorm) == (T)anorm){
						flag= false;
						break;
					}
					if((T)(fabs(W[nm])+anorm) == (T)anorm)
						break;
				}
				if(flag){
					//Cancellation of rv1[l], if l>1
					T c= 0.0;
					T s= 1.0;
					for(int i=l;i<k;i++){
						T f= s*rv1[i];
						rv1[i]= c*rv1[i];
						if((T)(fabs(f)+anorm) == (T)anorm)
							break;
						T g= W[i];
						T h= pythag(f,g);
						W[i]= h;
						h= 1.0/h;
						c= g*h;
						s= -f*h;
						for(int j=0;j<M;j++){
							T y= U(j,nm);
							T z= U(j,i);
							U(j,nm)= y*c+z*s;
							U(j,i)= z*c-y*s;
						}
					}
				}
				T z= W[k];
				//Convergence
				//Singular value is made nonnegative
				if(l==k){
					if(z<0.0){
						W[k]= -z;
						for(int j=0;j<N;j++)
							V(j,k)= -V(j,k);
					}
					break;
				}
				if(its==max_iterations-1)
					throw SVDError(SVDError::ERROR_NO_CONVERGENCE);
				//Shift from bottom 2-by-2 minor
				T x= W[l];
				nm= k-1;
				T y= W[nm];
				T g= rv1[nm];
				T h= rv1[k];
				T f= ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);;
				g= pythag(f,1.0);
				f=((x-z)*(x+z)+h*((y/(f+sign(g,f)))-h))/x;
				//Next QR transformation
				T c= 1.0,s= 1.0;
				for(int j=l;j<=nm;j++){
					int i=j+1;
					g= rv1[i];
					y= W[i];
					h= s*g;
					g= c*g;
					z= pythag(f,h);
					rv1[j]= z;
					c= f/z;
					s= h/z;
					f= x*c+g*s;
					g= g*c-x*s;
					h= y*s;
					y*= c;
					for(int jj=0;jj<N;jj++){
						x= V(jj,j);
						z= V(jj,i);
						V(jj,j)= x*c+z*s;
						V(jj,i)= z*c-x*s;
					}
					z=pythag(f,h);
					//Rotation can be arbitrary if z = 0
					W[j]= z;
					if(z){
						z= 1.0/z;
						c= f*z;
						s= h*z;
					}
					f= c*g+s*y;
					x= c*y-s*g;
					for(int jj=0;jj<M;jj++){
						y= U(jj,j);
						z= U(jj,i);
						U(jj,j)= y*c+z*s;
						U(jj,i)= z*c-y*s;
					}
				}
				rv1[l]= 0.0;
				rv1[k]= f;
				W[k]= x;
			}
		}
	}
	//Create sigma matrix
	Sigma.fill(0.0);
	for(int i=0;i<N;i++)
		Sigma(i,i)= W[i];
}


template<int N,int M,typename T> inline T SVD<N,M,T>::pythag(T a, T b)const{
	//Computes (a2 + b2)^(1/2) without destructive underflow or overflow
	T absa=fabs(a),absb=fabs(b);
	if (absa > absb) return absa*sqrt(1.0+sqr(absb/absa));
	else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+sqr(absa/absb)));
}
template<int N,int M,typename T> inline T SVD<N,M,T>::sqr(T arg)const{return arg*arg;}
template<int N,int M,typename T> inline T SVD<N,M,T>::sign(T a,T b)const{return b>=0.0?fabs(a):-fabs(a);}

void test_svd(){
	QGenericMatrix<3,3,double> matrix;
	matrix.fill(0);
	/*
	//U=[3 67 46;56 19 98;88 37 16];
	matrix(0,0)= 3; matrix(0,1)= 67; matrix(0,2)= 46;
	matrix(1,0)= 56; matrix(1,1)= 19; matrix(1,2)= 98;
	matrix(2,0)= 88; matrix(2,1)= 37; matrix(2,2)= 16;
	*/
	//U=[1 12 9;1 14 7;8 2 0];
	matrix(0,0)= 1; matrix(0,1)= 12; matrix(0,2)= 9;
	matrix(1,0)= 1; matrix(1,1)= 14; matrix(1,2)= 7;
	matrix(2,0)= 8; matrix(2,1)= 2; matrix(2,2)= 0;
	SVD<3,3,double> mysvd(matrix);
	QGenericMatrix<3,3,double> result= mysvd.getU()*mysvd.getSigma()*(mysvd.getV().transposed());

	QGenericMatrix<3,3,double> resultDelta= result-matrix;
}
