#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "linmath.h"

void multiply_matrix_3x3(float a[9], float b[9], float output[9])
{
    for(size_t i=0;i<3;i++)
        for(size_t j=0;j<3;j++)
            output[i*3+j] = a[i*3+0]*b[0*3+j] + a[i*3+1]*b[1*3+j] + a[i*3+2]*b[2*3+j];
}

void multiply_matrix_3tx3(float at[9], float b[9], float output[9])
{
    for(size_t i=0;i<3;i++)
        for(size_t j=0;j<3;j++)
            output[i*3+j] = at[0*3+i]*b[0*3+j] + at[1*3+i]*b[1*3+j] + at[2*3+i]*b[2*3+j];
}

void multiply_f_vector(float f[9], float p[3], float target[3])
{
    target[0] = f[0]*p[0]+f[1]*p[1]+f[2]*p[2];
    target[1] = f[3]*p[0]+f[4]*p[1]+f[5]*p[2];
    target[2] = f[6]*p[0]+f[7]*p[1]+f[8]*p[2];
}

void multiply_ft_vector(float f[9], float p[3], float target[3])
{
    target[0] = f[0]*p[0]+f[3]*p[1]+f[6]*p[2];
    target[1] = f[1]*p[0]+f[4]*p[1]+f[7]*p[2];
    target[2] = f[2]*p[0]+f[5]*p[1]+f[8]*p[2];
}

typedef struct {
    size_t temp_size;
    double *temp_ata, *temp_q, *temp_r, *temp_h;
    double *previous_q, *q, *r, *temp_pq, *previous_pq;
    unsigned int *rand_seed;
} svd_ctx;

svd_internal init_svd()
{
    svd_ctx *ctx = malloc(sizeof(svd_ctx));
    ctx->temp_ata = NULL;
    ctx->temp_q = NULL;
    ctx->temp_r = NULL;
    ctx->temp_h = NULL;
    ctx->previous_q = NULL;
    ctx->q = NULL;
    ctx->r = NULL;
    ctx->temp_pq = NULL;
    ctx->previous_pq = NULL;
    ctx->temp_size = 0;
    return ctx;
}
void free_svd(svd_internal svd)
{
    svd_ctx *ctx = svd;
    if (ctx == NULL)
        return;
    if (ctx->temp_ata != NULL)
        free(ctx->temp_ata);
    if (ctx->temp_q != NULL)
        free(ctx->temp_q);
    if (ctx->temp_r != NULL)
        free(ctx->temp_r);
    if (ctx->temp_h != NULL)
        free(ctx->temp_h);
    if (ctx->previous_q != NULL)
        free(ctx->previous_q);
    if (ctx->q != NULL)
        free(ctx->q);
    if (ctx->r != NULL)
        free(ctx->r);
    if (ctx->temp_pq != NULL)
        free(ctx->temp_pq);
    if (ctx->previous_pq != NULL)
        free(ctx->previous_pq);
    free(ctx);
}

static inline void multiply_matrix_mxk_kxn(double *a, double *b, double *result, const size_t m, const size_t k, const size_t n)
{
    for(size_t i=0;i<m*n;i++)
        result[i] = 0.0;
    for(size_t i=0;i<m;i++)
        for(size_t j=0;j<n;j++)
            for(size_t l=0;l<k;l++)
                result[i*n+j] += a[i*n+l]*b[l*n+j];
}

static inline void multiply_matrix_kxmt_kxn(double *a, double *b, double *result, const size_t m, const size_t k, const size_t n)
{
    for(size_t i=0;i<m*n;i++)
        result[i] = 0.0;
    for(size_t i=0;i<m;i++)
        for(size_t j=0;j<n;j++)
            for(size_t l=0;l<k;l++)
                result[i*n+j] += a[l*m+i]*b[l*n+j];
}

static inline void normalize_vector(double *vector, size_t n)
{
    double sum = 0.0;
    for (size_t i=0;i<n;i++)
        sum += vector[i]*vector[i];
    sum = sqrt(sum);
    for (size_t i=0;i<n;i++)
        vector[i] = vector[i]/sum;
}

static inline void svd_householder(svd_ctx *ctx, double *a, double *h, size_t n, size_t it)
{
    for(size_t i=0;i<n;i++)
        for(size_t j=0;j<n;j++)
            h[i*n+j] = (i==j?1:0);
    double a_norm = 0.0;
    for(size_t i=it;i<n;i++)
        a_norm += a[i*n+it]*a[i*n+it];
    a_norm = sqrt(a_norm);
    a_norm = 1.0/(a[it*n+it] + (a[it*n+it]>0.0?a_norm:-a_norm));

    double v_dot_v = 1.0;
    for(size_t i=it+1;i<n;i++)
        v_dot_v += (a[i*n+it]*a_norm)*(a[i*n+it]*a_norm);
    for(size_t i=it;i<n;i++)
    {
        for(size_t j=it;j<n;j++)
        {
            double v_ik = i==it?1:(a[i*n+it]*a_norm);
            double vt_kj = j==it?1:(a[j*n+it]*a_norm);
            h[i*n+j] -= (2.0/v_dot_v)*v_ik*vt_kj;
        }
    }
}

static inline void svd_qr(svd_ctx *ctx, double *a, double *q, double *r, const size_t n)
{
    // Based on https://rosettacode.org/wiki/QR_decomposition#Python
    for(size_t i=0;i<n;i++)
    {
        for(size_t j=0;j<n;j++)
        {
            q[i*n+j] = (i==j?1:0);
            r[i*n+j] = a[i*n+j];
        }
    }
    for(size_t i=0;i<n-1;i++)
    {
        svd_householder(ctx, r, ctx->temp_h, n, i);
        multiply_matrix_mxk_kxn(q, ctx->temp_h, ctx->temp_q, n, n, n);
        multiply_matrix_mxk_kxn(ctx->temp_h, r, ctx->temp_r, n, n, n);
        for(size_t j=0;j<n*n;j++)
        {
            q[j] = ctx->temp_q[j];
            r[j] = ctx->temp_r[j];
        }
    }
}

int svd(svd_internal svd, double *matrix, const size_t rows, const size_t cols, double *u, double *s, double *vt)
{
    // SVD using QR decomposition
    svd_ctx *ctx = svd;
    // TODO: if rows<cols, transpose matrix in-place and swap U and S
    if (rows<cols)
        return 0;
    //for (size_t i=0;i<rows*cols;i++)
    //    matrix[i] = i+1;

    for(size_t i=0;i<cols;i++)
        for(size_t j=0;j<cols;j++)
            vt[i] = NAN;
    if (ctx->temp_size<cols)
    {
        size_t new_size = sizeof(double)*cols*cols;
        ctx->temp_size = cols;
        ctx->temp_ata = ctx->temp_ata==NULL? malloc(new_size) : realloc(ctx->temp_ata, new_size);
        ctx->temp_q = ctx->temp_q==NULL? malloc(new_size) : realloc(ctx->temp_q, new_size);
        ctx->temp_r = ctx->temp_r==NULL? malloc(new_size) : realloc(ctx->temp_r, new_size);
        ctx->temp_h = ctx->temp_h==NULL? malloc(new_size) : realloc(ctx->temp_h, new_size);
        ctx->previous_q = ctx->previous_q==NULL? malloc(new_size) : realloc(ctx->previous_q, new_size);
        ctx->q = ctx->q==NULL? malloc(new_size) : realloc(ctx->q, new_size);
        ctx->r = ctx->r==NULL? malloc(new_size) : realloc(ctx->r, new_size);
        ctx->temp_pq = ctx->temp_pq==NULL? malloc(new_size) : realloc(ctx->temp_pq, new_size);
        ctx->previous_pq = ctx->previous_pq==NULL? malloc(new_size) : realloc(ctx->previous_pq, new_size);
    }

    multiply_matrix_kxmt_kxn(matrix, matrix, ctx->temp_ata, cols, rows, cols);
    //for (size_t i=0;i<cols*cols;i++)
    //    ctx->temp_ata_q[i] = ctx->temp_ata[i];
    /*
    printf("AtA=\n[");
    for (size_t j=0;j<cols;j++)
    {
        printf("[");
        for (size_t k=0;k<cols;k++)
            printf("%f%s", ctx->temp_ata[j*cols+k], k<cols-1?",":"");
        printf("]%s", j<cols-1?",":"");
    }
    printf("]\n");
    */
    for(size_t i=0;i<cols;i++)
        for(size_t j=0;j<cols;j++)
            ctx->previous_pq[i*cols+j] = (i==j)?1.0:0;
    int found = 0;
    for(size_t i=0;i<1000;i++)
    {
        svd_qr(ctx, ctx->temp_ata, ctx->q, ctx->r, cols);
        multiply_matrix_mxk_kxn(ctx->r, ctx->q, ctx->temp_ata, cols, cols, cols);
        multiply_matrix_mxk_kxn(ctx->previous_pq, ctx->q, ctx->temp_pq, cols, cols, cols);
        for(size_t j=0;j<cols*cols;j++)
            ctx->previous_pq[j] = ctx->temp_pq[j];
        if (i>0)
        {
            double delta = 0.0;
            for(size_t j=0;j<cols*cols;j++)
                delta += (ctx->previous_q[j]-ctx->q[j])*(ctx->previous_q[j]-ctx->q[j]);
            if (delta<(1.0E-10))
            {
                found = 1;
                break;
            }
        }
        for(size_t i=0;i<cols*cols;i++)
            ctx->previous_q[i] = ctx->q[i];
    }

    if (!found)
        return 0;
    for(size_t i=0;i<cols;i++)
        for(size_t j=0;j<cols;j++)
            vt[i*cols+j] = ctx->previous_pq[i*cols+j];
    for(size_t i=0;i<cols;i++)
    {
        s[i] = ctx->temp_ata[i*cols+i];
        if (s[i]<0)
        {
            s[i] = -s[i];
            for(size_t j=0;j<cols;j++)
                vt[i*cols+j] = -vt[i*cols+j];
        }
        s[i] = sqrt(s[i]);
    }
    multiply_matrix_mxk_kxn(matrix, ctx->q, ctx->temp_q, cols, cols, cols);
    /*
    printf("\naq=\n[");
    for (size_t j=0;j<rows;j++)
    {
        printf("[");
        for (size_t k=0;k<cols;k++)
            printf("%f%s", ctx->temp_q[j*cols+k], k<cols-1?",":"");
        printf("]%s", j<rows-1?",":"");
    }
    */
    for(size_t i=0;i<cols;i++)
        for(size_t j=0;j<cols;j++)
            u[i*cols+j] = ctx->temp_q[i*cols+j]/s[j];

    /*
    printf("\nq=\n[");
    for (size_t j=0;j<cols;j++)
    {
        printf("[");
        for (size_t k=0;k<cols;k++)
            printf("%f%s", ctx->q[j*cols+k], k<cols-1?",":"");
        printf("]%s", j<cols-1?",":"");
    }
    printf("\nr=\n[");
    for (size_t j=0;j<cols;j++)
    {
        printf("[");
        for (size_t k=0;k<cols;k++)
            printf("%f%s", ctx->r[j*cols+k], k<cols-1?",":"");
        printf("]%s", j<cols-1?",":"");
    }
    printf("\nu=\n[");
    for (size_t i=0;i<cols;i++)
    {
        printf("[");
        for (size_t j=0;j<cols;j++)
            printf("%f%s", u[i*cols+j], j<cols-1?",":"");
        printf("]%s", i<cols-1?",":"");
    }
    printf("]\ns=\n[");
    for (size_t i=0;i<cols;i++)
    {
        printf("%f%s", s[i], i<cols-1?",":"");
    }
    printf("]");
    printf("\nvt=\n[");
    for (size_t i=0;i<cols;i++)
    {
        printf("[");
        for (size_t j=0;j<cols;j++)
            printf("%f%s", vt[i*cols+j], j<cols-1?",":"");
        printf("]%s", i<cols-1?",":"");
    }
    printf("]\n");
    double *s_diag = malloc(sizeof(double)*rows*cols);
    for (size_t i=0;i<rows;i++)
        for (size_t j=0;j<cols;j++)
            s_diag[i*cols+j] = (i==j)?s[i]:0;
    printf("s_diag=\n[");
    for (size_t j=0;j<rows;j++)
    {
        printf("[");
        for (size_t k=0;k<cols;k++)
            printf("%f%s", s_diag[j*cols+k], k<cols-1?",":"");
        printf("]%s", j<rows-1?",":"");
    }
    printf("]\n");
    double *us = malloc(sizeof(double)*cols*cols);
    multiply_matrix_mxk_kxn(u, s_diag, us, cols, cols, cols);
    printf("us=\n[");
    for (size_t j=0;j<cols;j++)
    {
        printf("[");
        for (size_t k=0;k<cols;k++)
            printf("%f%s", us[j*cols+k], k<cols-1?",":"");
        printf("]%s", j<cols-1?",":"");
    }
    printf("]\n");
    multiply_matrix_mxk_kxn(us, vt, ctx->temp_ata, cols, cols, cols);
    printf("AtA=\n[");
    for (size_t j=0;j<cols;j++)
    {
        printf("[");
        for (size_t k=0;k<cols;k++)
            printf("%f%s", ctx->temp_ata[j*cols+k], k<cols-1?",":"");
        printf("]%s", j<cols-1?",":"");
    }
    printf("]\n");
    */
    return 1;
}
