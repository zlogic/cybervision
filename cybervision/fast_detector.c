#include <stdlib.h>

#include <fast.h>

#include "correlation.h"
#include "configuration.h"

correlation_point* fast_detect(correlation_image *img, size_t  *count)
{
    int threshold = cybervision_fast_threshold, mode = cybervision_fast_mode, nonmax = cybervision_fast_nonmax;
    int num_corners;
    xy* corners;
    correlation_point *out = NULL;

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
        return out;

    *count = (size_t)num_corners;
    out = malloc(sizeof(correlation_point)* num_corners);
    for(int i=0;i<num_corners;i++) {
        out[i].x = corners[i].x;
        out[i].y = corners[i].y;
    }

    free(corners);
    return out;
}
