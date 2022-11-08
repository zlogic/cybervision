#include <stdlib.h>
#include <pthread.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "configuration.h"
#include "triangulation.h"
#include "fundamental_matrix.h"
#include "linmath.h"

void find_min_max_depth(triangulation_task *t, float *min_depth, float *max_depth)
{
    float min = NAN, max = NAN;
    for(size_t i=0;i<t->width*t->height;i++)
    {
        float depth = t->out_depth[i];
        if (!isfinite(depth))
            continue;
        min = (depth<min || !isfinite(min))? depth:min;
        max = (depth>max || !isfinite(max))? depth:max;
    }
    *min_depth = min;
    *max_depth = max;
}

int triangulation_triangulate_point(svd_internal svd_ctx, double projection2[4*3], double x1, double y1, double x2, double y2, double result[4])
{
    // Linear triangulation method
    double a[4*4];
    double u[4*4], s[4], vt[4*4];
    // First row of A: x1*[0 0 1 0]-[1 0 0 0]
    a[0+4*0]= -1.0;
    a[0+4*1]= 0.0;
    a[0+4*2]= x1;
    a[0+4*3]= 0.0;
    // Second row of A: y1*[0 0 1 0]-[0 1 0 0]
    a[1+4*0]= 0.0;
    a[1+4*1]= -1.0;
    a[1+4*2]= y1;
    a[1+4*3]= 0.0;
    // Third row of A: x2*camera_2[2]-camera_2[0]
    a[2+4*0]= x2*projection2[2*4+0]-projection2[0+0];
    a[2+4*1]= x2*projection2[2*4+1]-projection2[0+1];
    a[2+4*2]= x2*projection2[2*4+2]-projection2[0+2];
    a[2+4*3]= x2*projection2[2*4+3]-projection2[0+3];
    // Fourth row of A: y2*camera_2[2]-camera_2[1]
    a[3+4*0]= y2*projection2[2*4+0]-projection2[4+0];
    a[3+4*1]= y2*projection2[2*4+1]-projection2[4+1];
    a[3+4*2]= y2*projection2[2*4+2]-projection2[4+2];
    a[3+4*3]= y2*projection2[2*4+3]-projection2[4+3];

    if (!svdd(svd_ctx, a, 4, 4, u, s, vt))
        return 0;

    for(size_t i=0;i<4;i++)
        result[i] = vt[i*4+3];
    return 1;
}

void filter_depth_histogram(triangulation_task *t, const float histogram_discard_percentile)
{
    const size_t histogram_bins = cybervision_histogram_filter_bins;
    const float histogram_depth_epsilon = cybervision_histogram_filter_epsilon;
    float min, max;
    size_t *histogram = malloc(sizeof(size_t)*histogram_bins);
    size_t histogram_sum = 0;
    size_t current_histogram_sum;
    float min_depth, max_depth;
    find_min_max_depth(t, &min, &max);
    min_depth = min;
    max_depth = max;
    for(int i=0;i<histogram_bins;i++)
        histogram[i] = 0;
    for(size_t i=0;i<t->width*t->height;i++)
    {
        float depth = t->out_depth[i];
        if (!isfinite(depth))
            continue;
        int pos = (int)roundf((depth-min)*histogram_bins/(max-min));
        pos = pos>=0? pos:0;
        pos = pos<histogram_bins? pos:histogram_bins-1;
        histogram[pos]++;
        histogram_sum++;
    }
    current_histogram_sum = 0;
    for(size_t i=0;i<histogram_bins;i++)
    {
        current_histogram_sum += histogram[i];
        if (((float)current_histogram_sum/(float)histogram_sum)>histogram_discard_percentile)
            break;
        min_depth = min + ((float)i/(float)histogram_bins-histogram_depth_epsilon)*(max-min);
    }
    current_histogram_sum = 0;
    for(size_t i=histogram_bins-1;i>=0;i--)
    {
        current_histogram_sum += histogram[i];
        if (((float)current_histogram_sum/(float)histogram_sum)>histogram_discard_percentile)
            break;
        max_depth = min + ((float)i/(float)histogram_bins+histogram_depth_epsilon)*(max-min);
    }
    free(histogram);
    for(size_t i=0;i<t->width*t->height;i++)
    {
        float depth = t->out_depth[i];
        if (!isfinite(depth))
            continue;
        if (depth<min_depth || depth>max_depth)
            t->out_depth[i] = NAN;
    }
}

void triangulation_parallel(triangulation_task *t)
{
    const float depth_scale = t->scale_z*((t->scale_x+t->scale_y)/2.0F);
    for (int y1=0;y1<t->height;y1++)
    {
        for (int x1=0;x1<t->width;x1++)
        {
            size_t pos = y1*t->width+x1;
            int x2 = t->correlated_points[pos*2];
            int y2 = t->correlated_points[pos*2+1];
            if (x2<0 || y2<0)
            {
                t->out_depth[y1*t->width+x1] = NAN;
                continue;
            }
            float dx = (float)x1-(float)x2, dy = (float)y1-(float)y2;
            t->out_depth[y1*t->width+x1] = sqrtf(dx*dx+dy*dy)*depth_scale;
        }
    }
}

int triangulation_optimize(triangulation_task *t)
{
    const size_t block_count = cybervision_triangulation_optimization_block_count;
    const size_t block_width = t->width/block_count;
    const size_t block_height = t->height/block_count;
    ransac_match *matches = malloc(sizeof(ransac_match)*block_count*block_count);
    size_t matches_count = 0;
    for (int j=0;j<block_count;j++)
    {
        for (int i=0;i<block_count;i++)
        {
            int found = 0;
            for (int y1=(j*block_height);(y1<t->height)&&(y1<(j+1)*block_height);y1++)
            {
                for (int x1=(i*block_width);(x1<t->width)&&x1<((i+1)*block_width);x1++)
                {
                    size_t pos = y1*t->width+x1;
                    int x2 = t->correlated_points[pos*2];
                    int y2 = t->correlated_points[pos*2+1];
                    if (x2<0 || y2<0)
                        continue;
                    matches[matches_count].x1 = x1;
                    matches[matches_count].y1 = y1;
                    matches[matches_count].x2 = x2;
                    matches[matches_count].y2 = y2;
                    matches_count++;
                    found = 1;
                    break;
                }
            }
            if (found)
                break;
        }
    }
    matches = realloc(matches, sizeof(ransac_match)*matches_count);
    int result = optimize_fundamental_matrix(NULL, matches, matches_count, NULL, t->projection_matrix_2);

    free(matches);
    return result;
}

int triangulation_perspective(triangulation_task *t)
{
    svd_internal svd_ctx = init_svd();
    const float depth_scale = t->scale_z;

    for(int y1=0;y1<t->height;y1++)
    {
        for (int x1=0;x1<t->width;x1++)
        {
            size_t pos = y1*t->width+x1;
            int x2 = t->correlated_points[pos*2];
            int y2 = t->correlated_points[pos*2+1];
            t->out_depth[pos] = NAN;
            if (x2<0 || y2<0)
                continue;

            double point[4];
            if(!triangulation_triangulate_point(svd_ctx, t->projection_matrix_2, x1, y1, x2, y2, point))
                continue;

            if (fabs(point[3])<cybervision_triangulation_min_scale)
                continue;

            for (size_t p_i=0;p_i<4;p_i++)
                point[p_i] = point[p_i]/point[3];
            // Projection appears to be very precise, with x1==point[0]/point[2] and y==point[1]/point[2]
            t->out_depth[pos] = depth_scale*point[2];
        }
    }
cleanup:
    free_svd(svd_ctx);
}

int triangulation_triangulate(triangulation_task *task)
{
    task->error = NULL;
    if (task->proj_mode == PROJECTION_MODE_PARALLEL)
    {
        triangulation_parallel(task);
        filter_depth_histogram(task, cybervision_histogram_filter_discard_percentile);
        return 1;
    }
    if(task->proj_mode == PROJECTION_MODE_PERSPECTIVE)
    {
        triangulation_perspective(task);
        if (!triangulation_optimize(task))
        {
            task->error = "Failed to optimize projection matrix";
            return 0;
        }
        filter_depth_histogram(task, cybervision_histogram_filter_discard_percentile);
        return 1;
    }
    task->error = "Unsupported projection mode";
    return 0;
}
