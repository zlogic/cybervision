#ifndef IMAGE_H
#define IMAGE_H

#include "correlation.h"
#include "surface.h"

char *file_extension(char *filename);
correlation_image* load_image(char *filename);
void resize_image(correlation_image *src, correlation_image *dst, float scale);
int save_surface_image(surface_data *surface, char *filename);

#endif
