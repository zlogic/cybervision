#include <stdlib.h>
#include <math.h>

#include "filter.h"

float* gaussian_blur(surface_data* data, float sigma, int min_points)
{
    float* blurred_image = malloc(sizeof(float)*data->width*data->height);
    int radius = (int)ceilf(4.0F*sigma)+1;
    float* kernel = malloc(sizeof(float)*(2*radius+1));

    for(int i=0;i<data->width*data->height;i++)
        blurred_image[i] = NAN;

    for(int x=-radius;x<=radius;x++)
        kernel[x+radius] = expf(-(x*x)/(2.0F*sigma*sigma))/(sqrt(2.0F*M_PI)*sigma); 

    for (int y=radius;y<data->height-radius;y++)
    {
        for (int x=radius;x<data->width-radius;x++)
        {
            float value = 0.0F;
            int count = 0;
            for(int i=-radius;i<=radius;i++)
            {
                float coeff = kernel[i+radius];
                float depth = data->depth[y*data->width + x+i];
                if (!isfinite(depth))
                    continue; 
                value += coeff*depth;
                count++;
            }
            if (count < min_points)
                continue;
            // TODO: perhaps use counts to adjust weight?
            blurred_image[y*data->width + x] = value;
        }
    }

    for (int y=radius;y<data->height-radius;y++)
    {
        for (int x=radius;x<data->width-radius;x++)
        {
            float value = 0.0F;
            int count = 0;
            for(int i=-radius;i<=radius;i++)
            {
                float coeff = kernel[i+radius];
                float depth = blurred_image[(y+i)*data->width + x];
                if (!isfinite(depth))
                    continue; 
                value += coeff*depth;
                count++;
            }
            if (count < min_points)
                continue;
            // TODO: perhaps use counts to adjust weight?
            blurred_image[y*data->width + x] = value;
        }
    }
    return blurred_image;
}

int filter_peaks(surface_data* data, float sigma, int min_points, float threshold)
{
    float *blurred_image = gaussian_blur(data, sigma, min_points);
    for (int y=0;y<data->height;y++)
    {
        for (int x=0;x<data->width;x++)
        {
            float depth = data->depth[y*data->width + x];
            float blurred = blurred_image[y*data->width + x];
            if (!isfinite(depth))
                continue;
            if (!isfinite(blurred) || fabs(depth-blurred) > threshold)
                data->depth[y*data->width + x] = NAN;
        }
    }

    free(blurred_image);
    return 1;
}
