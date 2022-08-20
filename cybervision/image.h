#ifndef IMAGE_H
#define IMAGE_H

#include "correlation.h"

correlation_image* load_image(char *filename);
void resize_image(correlation_image *src, correlation_image *dst, float scale);

#endif
