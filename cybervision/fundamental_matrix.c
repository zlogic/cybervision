#include <stdlib.h>
#include <pthread.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "fundamental_matrix.h"
#include "triangulation.h"
#include "linmath.h"
#include "configuration.h"
#include "system.h"

#define EXTENDED_INLIERS_GROW_SIZE 1000

typedef struct {
    size_t iteration;
    unsigned int thread_id;

    double best_error;

    int threads_completed;
    pthread_mutex_t lock;
    pthread_t *threads;
} ransac_task_ctx;

static inline double ransac_calculate_error(ransac_match *selected_match, matrix_3x3 f)
{
    // Calculate Sampson distance
    double p1[3] = {selected_match->x1, selected_match->y1, 1.0};
    double p2[3] = {selected_match->x2, selected_match->y2, 1.0};
    double nominator = 0.0;
    double p2tFp1[3];
    p2tFp1[0] = p2[0]*f[0]+p2[1]*f[3]+p2[2]*f[6];
    p2tFp1[1] = p2[0]*f[1]+p2[1]*f[4]+p2[2]*f[7];
    p2tFp1[2] = p2[0]*f[2]+p2[1]*f[5]+p2[2]*f[8];
    nominator = p2tFp1[0]*p1[0]+p2tFp1[1]*p1[1]+p2tFp1[2]*p1[2];
    double Fp1[3];
    double Ftp2[3];
    multiply_f_vector(f, p1, Fp1);
    multiply_ft_vector(f, p2, Ftp2);
    double denominator = Fp1[0]*Fp1[0]+Fp1[1]*Fp1[1]+Ftp2[0]*Ftp2[0]+Ftp2[1]*Ftp2[1];
    return nominator*nominator/denominator;
}

typedef struct {
    svd_internal svd;
    invert_internal invert;
    double *a;
    double *u, *s, *vt;
    double *lm_residuals;
    double *lm_points3d;
    double *lm_jacobian;
    double *lm_lambda_jtj;
    double *lm_j_1;
    double *lm_j_2;
    double *lm_j_3;
    size_t lm_matches_count;
} ransac_memory;

static inline void normalize_points(ransac_match *matches, size_t matches_count, matrix_3x3 m1, matrix_3x3 m2)
{
    // Recenter & rescale points
    double centerX1 = 0.0, centerY1 = 0.0;
    double centerX2 = 0.0, centerY2 = 0.0;
    for(size_t i=0;i<matches_count;i++)
    {
        ransac_match *match = &matches[i];
        centerX1 += (double)match->x1;
        centerY1 += (double)match->y1;
        centerX2 += (double)match->x2;
        centerY2 += (double)match->y2;
    }
    centerX1 /= (double)matches_count;
    centerY1 /= (double)matches_count;
    centerX2 /= (double)matches_count;
    centerY2 /= (double)matches_count;
    double scale1 = 0.0, scale2 = 0.0;
    for(size_t i=0;i<matches_count;i++)
    {
        ransac_match *match = &matches[i];
        double dx1 = (double)match->x1-centerX1;
        double dy1 = (double)match->y1-centerY1;
        double dx2 = (double)match->x2-centerX2;
        double dy2 = (double)match->y2-centerY2;
        scale1 += sqrt(dx1*dx1 + dy1*dy1);
        scale2 += sqrt(dx2*dx2 + dy2*dy2);
    }
    scale1 = sqrt(2.0)/(scale1/(double)matches_count);
    scale2 = sqrt(2.0)/(scale2/(double)matches_count);

    m1[0] = scale1; m1[1] = 0.0; m1[2] = -centerX1*scale1;
    m1[3] = 0.0; m1[3+1] = scale1; m1[3+2] = -centerY1*scale1;
    m1[6] = 0.0; m1[6+1] = 0.0; m1[6+2] = 1.0;

    m2[0] = scale2; m2[1] = 0.0; m2[2] = -centerX2*scale2;
    m2[3] = 0.0; m2[3+1] = scale2; m2[3+2] = -centerY2*scale2;
    m2[6] = 0.0; m2[6+1] = 0.0; m2[6+2] = 1.0;
}

static inline void point_reprojection_error(matrix_4x3 projection_matrix_2, double point3d[4], double x1, double y1, double x2, double y2, double error[4])
{
    // Project point to image 1 (with identity projection matrix)
    error[0] = x1-point3d[0]/point3d[2];
    error[1] = y1-point3d[1]/point3d[2];
    // Project point to image 2
    double point_2[3];
    multiply_p_vector(projection_matrix_2, point3d, point_2);
    error[2] = x2-point_2[0]/point_2[2];
    error[3] = y2-point_2[1]/point_2[2];
}

int optimize_fundamental_matrix(fundamendal_matrix_internal internal_ctx, ransac_match *matches, size_t matches_count, matrix_3x3 fundamental_matrix, matrix_4x3 projection_matrix_2)
{
    ransac_memory *ctx = internal_ctx;
    // Gold standard optimization of fundamental matrix
    const size_t max_iterations = cybervision_lm_max_iterations;
    const double jacobian_h = cybervision_lm_jacobian_h;
    const double lambda_start = cybervision_lm_lambda_start;
    const double lambda_up = cybervision_lm_lambda_up;
    const double lambda_down = cybervision_lm_lambda_down;
    const double lambda_min = cybervision_lm_lambda_min;
    const double lambda_max = cybervision_lm_lambda_max;
    const double rho_epsilon = cybervision_lm_rho_epsilon;
    const double jt_residual_epsilon = cybervision_lm_jt_residual_epsilon;
    const double ratio_epsilon = cybervision_lm_ratio_epsilon;
    const double division_epsilon = cybervision_lm_division_epsilon;
    if (internal_ctx == NULL)
    {
        ctx = malloc(sizeof(ransac_memory));
        ctx->svd = init_svd();
        ctx->invert = init_invert();
        ctx->lm_matches_count = 0;
        ctx->lm_points3d = NULL;
        ctx->lm_residuals = NULL;
        ctx->lm_jacobian = NULL;
        ctx->lm_lambda_jtj = NULL;
        ctx->lm_j_1 = NULL;
        ctx->lm_j_2 = NULL;
        ctx->lm_j_3 = NULL;
    }

    double *p2 = projection_matrix_2;
    if (fundamental_matrix != NULL)
    {
        matrix_3x3 f;
        for (size_t i=0;i<3;i++)
            for (size_t j=0;j<3;j++)
                f[j*3+i] = fundamental_matrix[i*3+j];

        double u[9], s[3], vt[9];
        if (!svdd(ctx->svd, f, 3, 3, u, s, vt))
            return 0;

        // Using e' (epipole in second image) to calculate projection matrix for second image
        double e2[3];
        for(size_t i=0;i<3;i++)
            e2[i] = u[3*2+i];
        double e2_skewsymmetric[9] = {0.0, -e2[2], e2[1], e2[2], 0.0, -e2[0], -e2[1], e2[0], 0.0};
        double e2sf[9];
        multiply_matrix_3x3(e2_skewsymmetric, fundamental_matrix, e2sf);
        for (size_t i=0;i<3;i++)
            for (size_t j=0;j<3;j++)
                p2[4*i+j] = e2sf[3*i+j];
        for (size_t i=0;i<3;i++)
            p2[4*i+3] = e2[i];
    }
    
    if (ctx->lm_matches_count<matches_count)
    {
        size_t j_rows = matches_count*4, j_cols = 4*3+matches_count*3;
        size_t max_dimension = j_rows>j_cols?j_rows:j_cols;
        size_t jacobian_size = sizeof(double)*j_rows*j_cols;
        size_t jacobian_extra_size = sizeof(double)*max_dimension*max_dimension;
        size_t jtj_labmbda_size = sizeof(double)*j_cols*j_cols;
        size_t points3d_size = sizeof(double)*matches_count*4;
        size_t residuals_size = sizeof(double)*j_rows;
        ctx->lm_points3d = ctx->lm_points3d==NULL? malloc(points3d_size) : realloc(ctx->lm_points3d, points3d_size);
        ctx->lm_residuals = ctx->lm_residuals==NULL? malloc(residuals_size) : realloc(ctx->lm_residuals, residuals_size);
        ctx->lm_jacobian = ctx->lm_jacobian==NULL? malloc(jacobian_size) : realloc(ctx->lm_jacobian, jacobian_size);
        ctx->lm_lambda_jtj = ctx->lm_lambda_jtj==NULL? malloc(jtj_labmbda_size) : realloc(ctx->lm_lambda_jtj, jtj_labmbda_size);
        ctx->lm_j_1 = ctx->lm_j_1==NULL? malloc(jacobian_extra_size) : realloc(ctx->lm_j_1, jacobian_extra_size);
        ctx->lm_j_2 = ctx->lm_j_2==NULL? malloc(jacobian_extra_size) : realloc(ctx->lm_j_2, jacobian_extra_size);
        ctx->lm_j_3 = ctx->lm_j_3==NULL? malloc(jacobian_extra_size) : realloc(ctx->lm_j_3, jacobian_extra_size);
    }
    const size_t jacobian_rows = matches_count*4;
    //size_t jacobian_cols = 4*3+matches_count*3;
    const size_t jacobian_cols = 4*3;
    double *points3d = ctx->lm_points3d;
    double *residuals = ctx->lm_residuals;
    double *jacobian = ctx->lm_jacobian;
    double s_previous = 0.0;
    double lambda = lambda_start;
    int completed = 0;
    int result = 0;
    // Levenberg-Marquardt error minimization with matches_count*4 residuals:
    // For each point, a residual for every reprojected coordinate
    for (size_t i=0;i<jacobian_cols*jacobian_cols;i++)
        ctx->lm_lambda_jtj[i] = 0.0;
    for (size_t iter=0;iter<max_iterations;iter++)
    {
        double chi_squared = 0.0;
        // Triangulate points
        for (size_t m_i=0;m_i<matches_count;m_i++)
        {
            double point3d[4];
            ransac_match *match = &matches[m_i];
            if (!triangulation_triangulate_point(ctx->svd, p2, match->x1, match->y1, match->x2, match->y2, point3d))
                goto cleanup;
            for(size_t i=0;i<4;i++)
            {
                point3d[i] /= point3d[3];
                points3d[m_i*4+i] = point3d[i];
            }
            double projection_error[4];
            point_reprojection_error(p2, point3d, match->x1, match->y1, match->x2, match->y2, projection_error);
            for (size_t i=0;i<4;i++)
            {
                residuals[m_i*4+i] = projection_error[i];
                chi_squared += projection_error[i]*projection_error[i];
            }
        }
        // Calculate Jacobian using finite differences (central difference)
        // Jacobian is column-major
        // Modify all parameters of p2
        matrix_4x3 p2_modified;
        for (size_t p2_i=0;p2_i<4*3;p2_i++)
        {
            for(size_t i=0;i<4*3;i++)
                p2_modified[i] = p2[i];
            p2_modified[p2_i] = p2[p2_i]+jacobian_h;
            double projection_error[4];
            for (size_t m_i=0;m_i<matches_count;m_i++)
            {
                double *point3d = &points3d[m_i*4];
                ransac_match *match = &matches[m_i];
                point_reprojection_error(p2_modified, point3d, match->x1, match->y1, match->x2, match->y2, projection_error);
                for (size_t e_i=0;e_i<4;e_i++)
                    jacobian[p2_i*jacobian_rows+m_i*4+e_i] = projection_error[e_i];
            }
            p2_modified[p2_i] = p2[p2_i]-jacobian_h;
            for (size_t m_i=0;m_i<matches_count;m_i++)
            {
                double *point3d = &points3d[m_i*4];
                ransac_match *match = &matches[m_i];
                point_reprojection_error(p2_modified, point3d, match->x1, match->y1, match->x2, match->y2, projection_error);
                for (size_t e_i=0;e_i<4;e_i++)
                {
                    double error_top = jacobian[p2_i*jacobian_rows+m_i*4+e_i];
                    jacobian[p2_i*jacobian_rows+m_i*4+e_i] = (error_top-projection_error[e_i])/(2.0*jacobian_h);
                }
            }
        }
        // Modify all points
        // Warning: Jacobian needs to have more rows than columns, this only works when number of points is > 12
        /*
        for (size_t p_i=0;p_i<3;p_i++)
        {
            for (size_t m_i=0;m_i<matches_count;m_i++)
            {
                const size_t jacobian_column = 4*3+m_i*3+p_i;
                double *point3d = &points3d[m_i*4];
                double point3d_modified[4];
                ransac_match *match = &matches[m_i];
                for (size_t i=0;i<4;i++)
                    point3d_modified[i] = point3d[i];
                point3d_modified[p_i] = point3d[p_i]+jacobian_h;
                // A point only affects its own differential, so most differentials will be zero
                for(size_t i=0;i<jacobian_rows;i++)
                    jacobian[jacobian_column*jacobian_rows+i] = 0.0;
                double projection_error[4];
                point_reprojection_error(p2, point3d_modified, match->x1, match->y1, match->x2, match->y2, projection_error);
                for (size_t e_i=0;e_i<4;e_i++)
                    jacobian[jacobian_column*jacobian_rows+m_i*4+e_i] = projection_error[e_i];
                point3d_modified[p_i] = point3d[p_i]-jacobian_h;
                point_reprojection_error(p2, point3d_modified, match->x1, match->y1, match->x2, match->y2, projection_error);
                for (size_t e_i=0;e_i<4;e_i++)
                {
                    double error_top = jacobian[jacobian_column*jacobian_rows+m_i*4+e_i];
                    jacobian[jacobian_column*jacobian_rows+m_i*4+e_i] = (error_top-projection_error[e_i])/(2.0*jacobian_h);
                }
            }
        }
        */
        // TODO: optimize math or memory usage - most of this is direct conversion from math formulas, can be optimized
        // Calculate JtJ
        double *jt_j_plus_lambda = ctx->lm_j_1;
        double *lambda_jt_j = ctx->lm_lambda_jtj;
        multiplyd(jacobian, jacobian, jt_j_plus_lambda, jacobian_cols, jacobian_cols, jacobian_rows, 1, 0);
        // Add lambda*diag(JtJ)
        for (size_t i=0;i<jacobian_cols;i++)
        {
            lambda_jt_j[i*jacobian_cols+i] = jt_j_plus_lambda[i*jacobian_cols+i]*lambda;
            jt_j_plus_lambda[i*jacobian_cols+i] += lambda_jt_j[i*jacobian_cols+i];
        }
        // Calculate delta (h_lm)
        if(!invertd(ctx->invert, jt_j_plus_lambda, jacobian_cols))
            goto cleanup;
        double *jt_left_pseudoinverse = ctx->lm_j_2; // (JtJ+lambda*diag(JtJ))^-1*Jt
        multiplyd(jt_j_plus_lambda, jacobian, jt_left_pseudoinverse, jacobian_cols, jacobian_rows, jacobian_cols, 0, 1);
        jt_j_plus_lambda = NULL; // lm_j_1 is available
        double *delta = ctx->lm_j_1;
        multiplyd(jt_left_pseudoinverse, residuals, delta, jacobian_cols, 1, jacobian_rows, 0, 0);
        // Check if lambda needs adjustment
        jt_left_pseudoinverse = NULL; // lm_j_2 is available
        double *jt_residual = ctx->lm_j_2; // Jt*residual
        multiplyd(jacobian, residuals, jt_residual, jacobian_cols, 1, jacobian_rows, 1, 0);
        double *lambda_jt_j_delta = ctx->lm_j_3; // lambda*diag(JtJ)*delta
        multiplyd(lambda_jt_j, delta, lambda_jt_j_delta, jacobian_cols, 1, jacobian_cols, 0, 0);
        double max_jt_residual = NAN;
        for (size_t i=0;i<jacobian_cols;i++)
        {
            const double val = jt_residual[i];
            // lambda*diag(JtJ)*delta+Jt*residual
            lambda_jt_j_delta[i] += val;
            // Find max value in Jt*resudual to test for convergence
            if (!isfinite(max_jt_residual) || fabs(val)<max_jt_residual)
                max_jt_residual = fabs(val);
        }
        jt_residual = NULL; // lm_j_2 is available
        double rho_denominator = 0.0; // delta_t*(lambda*diag(JtJ)*delta+Jt*residual)
        multiplyd(delta, lambda_jt_j_delta, &rho_denominator, 1, 1, jacobian_cols, 1, 0);
        // Update projection matrix
        matrix_4x3 p2_new;
        double max_ratio = NAN;
        for (size_t i=0;i<4*3;i++)
        {
            double ratio = fabs(delta[i])/(fabs(p2[i])+division_epsilon);
            p2_new[i] = p2[i]+delta[i];
            if (isfinite(ratio) && (!isfinite(max_ratio) || ratio<max_ratio))
                max_ratio = ratio;
        }
        double chi_squared_new = 0.0;
        // Calculate new chi squared
        for (size_t m_i=0;m_i<matches_count;m_i++)
        {
            double point3d[4];
            ransac_match *match = &matches[m_i];
            if (!triangulation_triangulate_point(ctx->svd, p2_new, match->x1, match->y1, match->x2, match->y2, point3d))
                goto cleanup;
            for(size_t i=0;i<4;i++)
                point3d[i] /= point3d[3];
            double projection_error[4];
            point_reprojection_error(p2_new, point3d, match->x1, match->y1, match->x2, match->y2, projection_error);
            for (size_t i=0;i<4;i++)
                chi_squared_new += projection_error[i]*projection_error[i];
        }
        double rho = (chi_squared-chi_squared_new)/rho_denominator;
        if (rho<rho_epsilon)
        {
            lambda = lambda*lambda_up;
            lambda = lambda<lambda_max? lambda:lambda_max;
        }
        else
        {
            lambda = lambda/lambda_down;
            lambda = lambda>lambda_min? lambda:lambda_min;
            for (size_t i=0;i<4*3;i++)
                p2[i] = p2_new[i];
        }
        // Check for convergence
        if (iter<=2)
            continue;
        if (max_jt_residual<jt_residual_epsilon)
        {
            completed = 1;
            break;
        }
        else if (isfinite(max_ratio) && max_ratio<ratio_epsilon)
        {
            completed = 1;
            break;
        }
    }

    if (!completed)
        goto cleanup;

    // Update fundamental matrix from p2
    if (fundamental_matrix != NULL)
    {
        double t[3];
        for(size_t i=0;i<3;i++)
            t[i] = p2[i*4+3];
        double t_skewsymmetric[9] = {0.0, -t[2], t[1], t[2], 0.0, -t[0], -t[1], t[0], 0.0};
        matrix_3x3 M;
        for(size_t i=0;i<3;i++)
            for(size_t j=0;j<3;j++)
                M[i*3+j] = p2[i*4+j];
        multiply_matrix_3x3(t_skewsymmetric, M, fundamental_matrix);
    }

    result = 1;
cleanup:
    if (internal_ctx == NULL)
    {
        free_svd(ctx->svd);
        free_invert(ctx->invert);
        if (ctx->lm_points3d != NULL)
            free(ctx->lm_points3d);
        if (ctx->lm_residuals != NULL)
            free(ctx->lm_residuals);
        if (ctx->lm_jacobian != NULL)
            free(ctx->lm_jacobian);
        if (ctx->lm_lambda_jtj != NULL)
            free(ctx->lm_lambda_jtj);
        if (ctx->lm_j_1 != NULL)
            free(ctx->lm_j_1);
        if (ctx->lm_j_2 != NULL)
            free(ctx->lm_j_2);
        if (ctx->lm_j_3 != NULL)
            free(ctx->lm_j_3);
        free(ctx);
    }
    return result;
}

static inline int ransac_calculate_model_perspective(ransac_memory *ctx, ransac_match *matches, size_t matches_count, matrix_3x3 f, matrix_4x3 p2)
{
    matrix_3x3 m1;
    matrix_3x3 m2;
    normalize_points(matches, matches_count, m1, m2);
    // Calculate fundamental matrix using the 8-point algorithm
    double *a = ctx->a;
    double *u = ctx->u, *s = ctx->s, *vt = ctx->vt;
    for(size_t i=0;i<matches_count;i++)
    {
        ransac_match *match = &matches[i];
        double x1 = (double)match->x1*m1[0] + (double)match->y1*m1[1] + m1[2];
        double y1 = (double)match->x1*m1[3] + (double)match->y1*m1[4] + m1[5];
        double x2 = (double)match->x2*m2[0] + (double)match->y2*m2[1] + m2[2];
        double y2 = (double)match->x2*m2[3] + (double)match->y2*m2[4] + m2[5];
        a[0*matches_count+i] = x2*x1;
        a[1*matches_count+i] = x2*y1;
        a[2*matches_count+i] = x2;
        a[3*matches_count+i] = y2*x1;
        a[4*matches_count+i] = y2*y1;
        a[5*matches_count+i] = y2;
        a[6*matches_count+i] = x1;
        a[7*matches_count+i] = y1;
        a[8*matches_count+i] = 1.0;
    }
    if (!svdd(ctx->svd, a, matches_count, 9, u, s, vt))
        return 0;

    // Check if matrix rank is too low
    if (fabs(s[7])<cybervision_ransac_rank_epsilon)
        return 0;

   for(size_t i=0;i<9;i++)
        a[i] = vt[9*i+8];

    if (!svdd(ctx->svd, a, 3, 3, u, s, vt))
        return 0;

    matrix_3x3 u_transposed, vt_transposed;
    for(size_t i=0;i<3;i++)
    {
        for(size_t j=0;j<3;j++)
        {
            u_transposed[i*3+j] = u[j*3+i];
            vt_transposed[i*3+j] = vt[j*3+i];
        }
    }
    matrix_3x3 s_matrix = {s[0], 0.0, 0.0, 0.0, s[1], 0.0, 0.0, 0.0, 0.0};
    matrix_3x3 f_temp;
    multiply_matrix_3x3(u_transposed, s_matrix, f_temp);
    multiply_matrix_3x3(f_temp, vt_transposed, f);

    // Scale back to image coordinates
    multiply_matrix_3tx3(m2, f, f_temp);
    multiply_matrix_3x3(f_temp, m1, f);
    return optimize_fundamental_matrix(ctx, matches, matches_count, f, p2);
}

static inline int ransac_calculate_model_affine(ransac_memory *ctx, ransac_match *matches, size_t matches_count, matrix_3x3 f, matrix_4x3 p2)
{
    matrix_3x3 m1;
    matrix_3x3 m2;
    // Calculate fundamental matrix using the 4-point algorithm
    double *a = ctx->a, *u = ctx->u, *s = ctx->s, *vt = ctx->vt;
    double mean_x1 = 0.0, mean_y1 = 0.0;
    double mean_x2 = 0.0, mean_y2 = 0.0;
    for(size_t i=0;i<matches_count;i++)
    {
        ransac_match *match = &matches[i];
        double x1 = match->x1;
        double y1 = match->y1;
        double x2 = match->x2;
        double y2 = match->y2;
        a[0*matches_count+i] = x2;
        a[1*matches_count+i] = y2;
        a[2*matches_count+i] = x1;
        a[3*matches_count+i] = y1;
        mean_x2 += x2;
        mean_y2 += y2;
        mean_x1 += x1;
        mean_y1 += y1;
    }
    mean_x1 /= (double)matches_count;
    mean_y1 /= (double)matches_count;
    mean_x2 /= (double)matches_count;
    mean_y2 /= (double)matches_count;
    for(size_t i=0;i<matches_count;i++)
    {
        a[0*matches_count+i] -= mean_x2;
        a[1*matches_count+i] -= mean_y2;
        a[2*matches_count+i] -= mean_x1;
        a[3*matches_count+i] -= mean_y1;
    }
    if (!svdd(ctx->svd, a, matches_count, 4, u, s, vt))
        return 0;

    // Check if matrix rank is too low
    if (fabs(s[3])<cybervision_ransac_rank_epsilon)
        return 0;

    double vt_lastcol[4];
    for(size_t i=0;i<4;i++)
        vt_lastcol[i] = vt[4*i+3];

    f[0] = 0.0; f[1] = 0.0; f[2] = vt[0];
    f[3] = 0.0; f[4] = 0.0; f[5] = vt[1];
    f[6] = vt[2]; f[7] = vt[3]; f[8] = -(vt_lastcol[0]*mean_x2+vt_lastcol[1]*mean_y2+vt_lastcol[2]*mean_x1+vt_lastcol[3]*mean_y1);

    return 1;
}

void* correlate_ransac_task(void *args)
{
    ransac_task *t = args;
    ransac_task_ctx *ctx = t->internal;
    size_t ransac_n;
    ransac_match *inliers;
    size_t ransac_k;
    double ransac_t;
    size_t ransac_d_early_exit;
    matrix_3x3 fundamental_matrix;
    matrix_4x3 projection_matrix_2;
    size_t extended_inliers_count = 0;
    ransac_match *extended_inliers = NULL;
    size_t extended_inliers_limit = 0;
    ransac_memory ctx_memory = {0};
    int (*ransac_calculate_model)(ransac_memory *ctx, ransac_match *matches, size_t matches_count, matrix_3x3 f, matrix_4x3 projection_matrix_2);
    unsigned int rand_seed;

    if (pthread_mutex_lock(&ctx->lock) != 0)
        goto cleanup;
    rand_seed = (ctx->thread_id++) ^ (unsigned int)time(NULL);
    srand_thread(&rand_seed);
    if (pthread_mutex_unlock(&ctx->lock) != 0)
        goto cleanup;

    if (t->proj_mode == PROJECTION_MODE_PARALLEL)
    {
        ransac_calculate_model = ransac_calculate_model_affine;
        ransac_k = cybervision_ransac_k_affine;
        ransac_n = cybervision_ransac_n_affine;
        ransac_t = cybervision_ransac_t_affine;
        ransac_d_early_exit = cybervision_ransac_d_early_exit_parallel;
    }
    else if (t->proj_mode == PROJECTION_MODE_PERSPECTIVE)
    {
        ransac_calculate_model = ransac_calculate_model_perspective;
        ransac_k = cybervision_ransac_k_perspective;
        ransac_n = cybervision_ransac_n_perspective;
        ransac_t = cybervision_ransac_t_perspective/(t->keypoint_scale*t->keypoint_scale);
        ransac_d_early_exit = cybervision_ransac_d_early_exit_perspective;
    }
    inliers = malloc(sizeof(ransac_match)*ransac_n);
    extended_inliers_limit = ransac_n;
    extended_inliers = malloc(sizeof(ransac_match)*extended_inliers_limit);
    ctx_memory.svd = init_svd();
    ctx_memory.invert = init_invert();
    ctx_memory.a = malloc(sizeof(double)*(ransac_n*9));
    ctx_memory.u = malloc(sizeof(double)*(ransac_n*ransac_n));
    ctx_memory.s = malloc(sizeof(double)*(ransac_n>9?ransac_n:9));
    ctx_memory.vt = malloc(sizeof(double)*9*9);
    ctx_memory.lm_matches_count = 0;
    ctx_memory.lm_points3d = NULL;
    ctx_memory.lm_residuals = NULL;
    ctx_memory.lm_jacobian = NULL;
    ctx_memory.lm_lambda_jtj = NULL;
    ctx_memory.lm_j_1 = NULL;
    ctx_memory.lm_j_2 = NULL;
    ctx_memory.lm_j_3 = NULL;
    
    while (!t->completed)
    {
        size_t iteration;
        if (pthread_mutex_lock(&ctx->lock) != 0)
            goto cleanup;
        iteration = ctx->iteration++;
        t->percent_complete = 100.0F*(float)iteration/ransac_k;
        if (pthread_mutex_unlock(&ctx->lock) != 0)
            goto cleanup;

        if (iteration > ransac_k)
            break;

        if (iteration % cybervision_ransac_check_interval == 0 && t->result_matches_count > ransac_d_early_exit)
            t->completed = 1;

        extended_inliers_count = 0;

        for(size_t i=0;i<ransac_n;i++)
        {
            ransac_match *match;
            ransac_match_bucket *bucket;
            size_t bucket_i, match_i;
            bucket_i = rand_r(&rand_seed)%t->match_buckets_count;

            bucket = &t->match_buckets[bucket_i];
            match_i = rand_r(&rand_seed)%bucket->matches_count;

            match = &bucket->matches[match_i];
            inliers[i] = *match;
        }

        if (!ransac_calculate_model(&ctx_memory, inliers, ransac_n, fundamental_matrix, projection_matrix_2))
            continue;

        double fundamental_matrix_sum = 0.0;
        for(size_t i=0;i<9;i++)
        {
            if (!isfinite(fundamental_matrix[i]))
            {
                fundamental_matrix_sum = NAN;
                break;
            }
            fundamental_matrix_sum += fabs(fundamental_matrix[i]);
        }
        if (fundamental_matrix_sum == 0.0 || !isfinite(fundamental_matrix_sum))
            continue;

        double inliers_error = 0.0;

        for (size_t i=0;i<t->match_buckets_count;i++)
        {
            ransac_match_bucket *bucket = &t->match_buckets[i];
            for (size_t j=0;j<bucket->matches_count;j++)
            {
                int already_exists = 0;
                ransac_match *check_match = &bucket->matches[j];
                for (size_t k=0;k<ransac_n;k++)
                {
                    ransac_match *inlier_match = &inliers[k];
                    if (check_match->x1 == inlier_match->x1 && check_match->x2 == inlier_match->x2 && check_match->y1 == inlier_match->y1 && check_match->y2 == inlier_match->y2)
                    {
                        already_exists = 1;
                        break;
                    }
                }
                if (already_exists)
                    continue;

                double inlier_error = fabs(ransac_calculate_error(check_match, fundamental_matrix));
                if (inlier_error > ransac_t)
                    continue;

                if (extended_inliers_count >= extended_inliers_limit)
                {
                    extended_inliers_limit += EXTENDED_INLIERS_GROW_SIZE;
                    extended_inliers = realloc(extended_inliers, sizeof(ransac_match)*extended_inliers_limit);
                }
                extended_inliers[extended_inliers_count++] = *check_match;
                inliers_error += inlier_error;
            }
        }

        if (extended_inliers_count < cybervision_ransac_d)
            continue;

        for (size_t i=0;i<ransac_n;i++)
        {
            double inlier_error = fabs(ransac_calculate_error(&inliers[i], fundamental_matrix));
            if (inlier_error > ransac_t)
            {
                inliers_error = NAN;
                break;
            }
            extended_inliers[extended_inliers_count++] = inliers[i];
            inliers_error += inlier_error;
        }
        if (!isfinite(inliers_error))
            continue;
        inliers_error = fabs(inliers_error/(double)extended_inliers_count);

        if (pthread_mutex_lock(&ctx->lock) != 0)
            goto cleanup;
        if (extended_inliers_count >= t->result_matches_count && inliers_error <= ctx->best_error)
        {
            if (t->proj_mode == PROJECTION_MODE_PERSPECTIVE)
                optimize_fundamental_matrix(&ctx_memory, extended_inliers, extended_inliers_count, fundamental_matrix, projection_matrix_2);
            for (size_t i=0;i<9;i++)
                t->fundamental_matrix[i] = fundamental_matrix[i];
            for (size_t i=0;i<12;i++)
                t->projection_matrix_2[i] = projection_matrix_2[i];
            t->result_matches_count = extended_inliers_count;
            ctx->best_error = inliers_error;
        }
        if (pthread_mutex_unlock(&ctx->lock) != 0)
            goto cleanup;
    }
cleanup:
    free(inliers);
    free(extended_inliers);
    free(ctx_memory.a);
    free(ctx_memory.u);
    free(ctx_memory.s);
    free(ctx_memory.vt);
    if (ctx_memory.lm_points3d != NULL)
        free(ctx_memory.lm_points3d);
    if (ctx_memory.lm_residuals != NULL)
        free(ctx_memory.lm_residuals);
    if (ctx_memory.lm_jacobian != NULL)
        free(ctx_memory.lm_jacobian);
    if (ctx_memory.lm_lambda_jtj != NULL)
        free(ctx_memory.lm_lambda_jtj);
    if (ctx_memory.lm_j_1 != NULL)
        free(ctx_memory.lm_j_1);
    if (ctx_memory.lm_j_2 != NULL)
        free(ctx_memory.lm_j_2);
    if (ctx_memory.lm_j_3 != NULL)
        free(ctx_memory.lm_j_3);
    free_svd(ctx_memory.svd);
    free_invert(ctx_memory.invert);
    pthread_mutex_lock(&ctx->lock);
    ctx->threads_completed++;
    pthread_mutex_unlock(&ctx->lock);
    if (ctx->threads_completed >= t->num_threads)
        t->completed = 1;
    return NULL;
}

int correlation_ransac_start(ransac_task *task)
{
    ransac_task_ctx *ctx = malloc(sizeof(ransac_task_ctx));

    srand((unsigned int)time(NULL));
    
    task->internal = ctx;
    ctx->threads= malloc(sizeof(pthread_t)*task->num_threads);

    ctx->iteration = 0;
    ctx->thread_id = 0;
    task->percent_complete = 0.0;
    ctx->threads_completed = 0;
    task->completed = 0;
    task->error = NULL;

    ctx->best_error = INFINITY;
    task->result_matches_count = 0;

    if (pthread_mutex_init(&ctx->lock, NULL) != 0)
        return 0;

    for (int i = 0; i < task->num_threads; i++)
        pthread_create(&ctx->threads[i], NULL, correlate_ransac_task, task);

    return 1;
}

void correlation_ransac_cancel(ransac_task *t)
{
    if (t == NULL)
        return;
    t->completed = 1;
}

int correlation_ransac_complete(ransac_task *t)
{
    ransac_task_ctx *ctx;
    if (t == NULL || t->internal == NULL)
        return 1;
    ctx = t->internal;
    for (int i = 0; i < t->num_threads; i++)
        pthread_join(ctx->threads[i], NULL);

    pthread_mutex_destroy(&ctx->lock);
    free(ctx->threads);
    free(t->internal);
    t->internal = NULL;
    return t->error == NULL;
}
