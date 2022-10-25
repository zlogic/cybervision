#include <stdlib.h>
#include <pthread.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "configuration.h"
#include "triangulation.h"

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

void filter_depth_histogram(triangulation_task *t, const float histogram_discard_percentile)
{
    const size_t histogram_bins = cybervision_histogram_filter_bins;
    const float histogram_depth_epsilon = cybervision_histogram_filter_epsilon;
    float min, max;
    float min_depth;
    float max_depth;
    size_t *histogram = malloc(sizeof(size_t)*histogram_bins);
    size_t histogram_sum = 0;
    size_t current_histogram_sum;
    find_min_max_depth(t, &min, &max);
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
    min_depth = min;
    max_depth = max;
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
    for(size_t i=0;i<t->width*t->height;i++)
    {
        float depth = t->out_depth[i];
        if (!isfinite(depth))
            continue;
        if (depth<min_depth || depth>max_depth)
            t->out_depth[i] = NAN;
    }
}

void triangulation_triangulate(triangulation_task *t)
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
    filter_depth_histogram(t, cybervision_histogram_filter_discard_percentile_perspective);
}
