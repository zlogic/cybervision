#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <fast.h>

#include "correlation.h"
#include "configuration.h"

void increase_contrast(correlation_image *img)
{
    unsigned char min=255, max=0;
    for (int i=0;i<img->width*img->height;i++)
    {
        unsigned char value = img->img[i];
        min = min<value? min:value;
        max = max>value? max:value;
    }

    if (min>=max)
        return;

    float coeff = 255.0F/(max-min);
    for (int i=0;i<img->width*img->height;i++)
    {
        int value = (int)img->img[i];
        value = (int)roundf(coeff*(value - min));
        value = value>255? 255:value;
        value = value<0? 0:value;
        img->img[i] = (unsigned char)value;
    }
}

correlation_point* fast_detect(correlation_image *img, size_t  *count)
{
    int threshold = cybervision_fast_threshold, mode = cybervision_fast_mode, nonmax = cybervision_fast_nonmax;
    int num_corners;
    xy* corners = NULL;
    correlation_point *out = NULL;

    increase_contrast(img);

    *count = 0;
    
    if (nonmax && mode == 9)
        corners = fast9_detect_nonmax(img->img, img->width, img->height, img->width, threshold, &num_corners);
    else if (nonmax && mode == 10)
        corners = fast10_detect_nonmax(img->img, img->width, img->height, img->width, threshold, &num_corners);
    else if (nonmax && mode == 11)
        corners = fast11_detect_nonmax(img->img, img->width, img->height, img->width, threshold, &num_corners);
    else if (nonmax && mode == 12)
        corners = fast12_detect_nonmax(img->img, img->width, img->height, img->width, threshold, &num_corners);
    else if (!nonmax && mode == 9)
        corners = fast9_detect(img->img, img->width,img-> height, img->width, threshold, &num_corners);
    else if (!nonmax && mode == 10)
        corners = fast10_detect(img->img, img->width, img->height, img->width, threshold, &num_corners);
    else if (!nonmax && mode == 11)
        corners = fast11_detect(img->img, img->width, img->height, img->width, threshold, &num_corners);
    else if (!nonmax && mode == 12)
        corners = fast12_detect(img->img, img->width, img->height, img->width, threshold, &num_corners);
    else
        goto cleanup;

    *count = (size_t)num_corners;
    out = malloc(sizeof(correlation_point)* num_corners);
    for(int i=0;i<num_corners;i++) {
        out[i].x = corners[i].x;
        out[i].y = corners[i].y;
    }
cleanup:
    if (corners != NULL)
        free(corners);
    
    return out;
}
