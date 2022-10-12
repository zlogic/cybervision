#include <stdlib.h>
#include <math.h>

#include "linmath.h"

typedef struct {
    size_t rv1_size;
    double *rv1;
} svd_ctx;

svd_internal init_svd()
{
    svd_ctx *ctx = malloc(sizeof(svd_ctx));
    ctx->rv1 = NULL;
    ctx->rv1_size = 0;
    return ctx;
}
void free_svd(svd_internal svd)
{
    svd_ctx *ctx = svd;
    if (ctx == NULL)
        return;
    if (ctx->rv1 != NULL)
        free(ctx->rv1);
    free(ctx);
}

int svdcmp(svd_internal svd, double *a, int rows, int cols, double *w, double *v);

int svdd(svd_internal svd, double *matrix, int rows, int cols, double *s, double *v)
{
    svdcmp(svd, matrix, rows, cols, s, v);
	// Sort singular values
	for (int i=0; i<cols; i++)
    {
		int  i_max = i;
		for (int j=i+1; j<cols; j++)
			if (s[j] > s[i])
				i_max = j;

		if (i_max == i)
            continue;

        double temp = 0;
        temp = s[i];
        s[i] = s[i_max];
        s[i_max] = temp;

        for(int j=0; j<rows; j++)
        {
            temp = matrix[j*cols+i];
            matrix[j*cols+i] = matrix[j*cols+i_max];
            matrix[j*cols+i_max] = temp;
        }

        for(int j=0; j<cols; j++)
        {
            temp = v[j*cols+i];
            v[j*cols+i] = v[j*cols+i_max];
            v[j*cols+i_max] = temp;
        }
	}
    // Transpose V
    for(int i=0; i<cols; i++)
    {
        for(int j=i+1; j<cols; j++)
        {
            double temp = v[j*cols+i];
            v[j*cols+i] = v[i*cols+j];
            v[i*cols+j] = temp;
        }
    }
    return 1;
}

void multiply_matrix_3x3(double a[9], double b[9], double output[9])
{
	for(size_t i=0;i<3;i++)
		for(size_t j=0;j<3;j++)
			output[i*3+j] = a[i*3+0]*b[0*3+j] + a[i*3+1]*b[1*3+j] + a[i*3+2]*b[2*3+j];
}

void multiply_matrix_3tx3(double at[9], double b[9], double output[9])
{
	for(size_t i=0;i<3;i++)
		for(size_t j=0;j<3;j++)
			output[i*3+j] = at[0*3+i]*b[0*3+j] + at[1*3+i]*b[1*3+j] + at[2*3+i]*b[2*3+j];
}

void multiply_f_vector(double f[9], double p[3], double target[3])
{
    target[0] = f[0]*p[0]+f[1]*p[1]+f[2]*p[2];
    target[1] = f[3]*p[0]+f[4]*p[1]+f[5]*p[2];
    target[2] = f[6]*p[0]+f[7]*p[1]+f[8]*p[2];
}

void multiply_ft_vector(double f[9], double p[3], double target[3])
{
    target[0] = f[0]*p[0]+f[3]*p[1]+f[6]*p[2];
    target[1] = f[1]*p[0]+f[4]*p[1]+f[7]*p[2];
    target[2] = f[2]*p[0]+f[5]*p[1]+f[8]*p[2];
}

// TODO: rewrite this algorithm

#define SIGN(a,b) (((b)>0.0?fabs(a):-fabs(a)))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define SQR(a) ((a)*(a))

double pythag(double a, double b) {
	double absa, absb;

	absa = fabs(a);
	absb = fabs(b);

	if (absa > absb)
		return (absa * sqrt(1.0 + SQR(absb/absa)));
	else
		return (absb == 0.0 ? 0.0 : absb * sqrt(1.0 + SQR(absa / absb)));
}

int svdcmp(svd_internal svd, double *a, int nRows, int nCols, double *w, double *v)
{
	int flag, i, its, j, jj, k, l, nm;
	double anorm, c, f, g, h, s, scale, x, y, z, *rv1;
    svd_ctx *ctx = svd;

    if (ctx->rv1_size<nCols)
    {
        size_t new_size = sizeof(double)*nCols;
        ctx->rv1_size = nCols;
        ctx->rv1 = ctx->rv1==NULL ? malloc(new_size) : realloc(ctx->rv1, new_size);
    }
    rv1 = ctx->rv1;

	g = scale = anorm = 0.0;
	for (i = 0; i < nCols; i++) {
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < nRows) {
			for (k = i; k < nRows; k++)
				scale += fabs(a[k*nCols+i]);
			if (scale) {
				for (k = i; k < nRows; k++) {
					a[k*nCols+i] /= scale;
					s += a[k*nCols+i] * a[k*nCols+i];
				}
				f = a[i*nCols+i];
				g = -SIGN(sqrt(s),f);
				h = f * g - s;
				a[i*nCols+i] = f - g;
				for (j = l; j < nCols; j++) {
					for (s = 0.0, k = i; k < nRows; k++)
						s += a[k*nCols+i] * a[k*nCols+j];
					f = s / h;
					for (k = i; k < nRows; k++)
						a[k*nCols+j] += f * a[k*nCols+i];
				}
				for (k = i; k < nRows; k++)
					a[k*nCols+i] *= scale;
			}
		}
		w[i] = scale * g;
		g = s = scale = 0.0;
		if (i < nRows && i != nCols - 1) {
			for (k = l; k < nCols; k++)
				scale += fabs(a[i*nCols+k]);
			if (scale) {
				for (k = l; k < nCols; k++) {
					a[i*nCols+k] /= scale;
					s += a[i*nCols+k] * a[i*nCols+k];
				}
				f = a[i*nCols+l];
				g = -SIGN(sqrt(s),f);
				h = f * g - s;
				a[i*nCols+l] = f - g;
				for (k = l; k < nCols; k++)
					rv1[k] = a[i*nCols+k] / h;
				for (j = l; j < nCols; j++) {
					for (s = 0.0, k = l; k < nCols; k++)
						s += a[j*nCols+k] * a[i*nCols+k];
					for (k = l; k < nCols; k++)
						a[j*nCols+k] += s * rv1[k];
				}
				for (k = l; k < nCols; k++)
					a[i*nCols+k] *= scale;
			}
		}
		anorm = fmax(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}

	for (i = nCols - 1; i >= 0; i--) {
		if (i < nCols - 1) {
			if (g) {
				for (j = l; j < nCols; j++)
					v[j*nCols+i] = (a[i*nCols+j] / a[i*nCols+l]) / g;
				for (j = l; j < nCols; j++) {
					for (s = 0.0, k = l; k < nCols; k++)
						s += a[i*nCols+k] * v[k*nCols+j];
					for (k = l; k < nCols; k++)
						v[k*nCols+j] += s * v[k*nCols+i];
				}
			}
			for (j = l; j < nCols; j++)
				v[i*nCols+j] = v[j*nCols+i] = 0.0;
		}
		v[i*nCols+i] = 1.0;
		g = rv1[i];
		l = i;
	}

	for (i = MIN(nRows,nCols) - 1; i >= 0; i--) {
		l = i + 1;
		g = w[i];
		for (j = l; j < nCols; j++)
			a[i*nCols+j] = 0.0;
		if (g) {
			g = 1.0 / g;
			for (j = l; j < nCols; j++) {
				for (s = 0.0, k = l; k < nRows; k++)
					s += a[k*nCols+i] * a[k*nCols+j];
				f = (s / a[i*nCols+i]) * g;
				for (k = i; k < nRows; k++)
					a[k*nCols+j] += f * a[k*nCols+i];
			}
			for (j = i; j < nRows; j++)
				a[j*nCols+i] *= g;
		} else
			for (j = i; j < nRows; j++)
				a[j*nCols+i] = 0.0;
		++a[i*nCols+i];
	}

	for (k = nCols - 1; k >= 0; k--) {
		for (its = 0; its < 30; its++) {
			flag = 1;
			for (l = k; l >= 0; l--) {
				nm = l - 1;
				if ((fabs(rv1[l]) + anorm) == anorm) {
					flag = 0;
					break;
				}
				if ((fabs(w[nm]) + anorm) == anorm)
					break;
			}
			if (flag) {
				c = 0.0;
				s = 1.0;
				for (i = l; i <= k; i++) {
					f = s * rv1[i];
					rv1[i] = c * rv1[i];
					if ((fabs(f) + anorm) == anorm)
						break;
					g = w[i];
					h = pythag(f, g);
					w[i] = h;
					h = 1.0 / h;
					c = g * h;
					s = -f * h;
					for (j = 0; j < nRows; j++) {
						y = a[j*nCols+nm];
						z = a[j*nCols+i];
						a[j*nCols+nm] = y * c + z * s;
						a[j*nCols+i] = z * c - y * s;
					}
				}
			}
			z = w[k];
			if (l == k) {
				if (z < 0.0) {
					w[k] = -z;
					for (j = 0; j < nCols; j++)
						v[j*nCols+k] = -v[j*nCols+k];
				}
				break;
			}
			if (its == 29)
                return 0;
			x = w[l];
			nm = k - 1;
			y = w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = pythag(f, 1.0);
			f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g,f)))- h)) / x;
			c = s = 1.0;
			for (j = l; j <= nm; j++) {
				i = j + 1;
				g = rv1[i];
				y = w[i];
				h = s * g;
				g = c * g;
				z = pythag(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y *= c;
				for (jj = 0; jj < nCols; jj++) {
					x = v[jj*nCols+j];
					z = v[jj*nCols+i];
					v[jj*nCols+j] = x * c + z * s;
					v[jj*nCols+i] = z * c - x * s;
				}
				z = pythag(f, h);
				w[j] = z;
				if (z) {
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = c * g + s * y;
				x = c * y - s * g;
				for (jj = 0; jj < nRows; jj++) {
					y = a[jj*nCols+j];
					z = a[jj*nCols+i];
					a[jj*nCols+j] = y * c + z * s;
					a[jj*nCols+i] = z * c - y * s;
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = x;
		}
	}

	return 1;
}
