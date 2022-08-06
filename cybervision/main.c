#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <tiffio.h>

#include "cybervision.h"

/*
 * FEI tags containing image details
 */
#define TIFFTAG_META_PHENOM 34683
#define TIFFTAG_META_QUANTA 34682

char *file_extension(char *filename)
{
    char* dot_i = strrchr(filename, '.');
    if (dot_i == NULL || dot_i == filename && strlen(dot_i) == 0)
        return "";
    return dot_i+1;
}

static inline char convert_grayscale(int r, int g, int b)
{
    return (char)((19595*r + 38470*g + 7471*b)>>16);
}

int read_tiff_tags(TIFF* tif, int* databar_height)
{
    void* tag_data = NULL;
    uint32_t tag_count;
    char *line = NULL, *line_mem = NULL;
    char *section = NULL;

    if (!TIFFGetField(tif, TIFFTAG_META_PHENOM, &tag_count, &tag_data) && !TIFFGetField(tif, TIFFTAG_META_QUANTA, &tag_count, &tag_data))
        return 0;
    
    line = malloc(sizeof(char)*(tag_count+1));
    memcpy(line, tag_data, tag_count);
    line_mem = line;
    line[tag_count] = '\0';
    while (line != NULL && *line != '\0')
    {
        char *newline = strchr(line, '\n');
        size_t line_len = newline==NULL? strlen(line):newline-line;
        char *cr = strchr(line, '\r');
        if (cr != NULL && cr-line < line_len)
            line_len = cr-line;
        line[line_len] = '\0';
        if (line[0]== '[')
            section = line;
        /*
        // TODO: parse and use scale
        if (strcmp(section, "[Scan]") == 0)
        {
            if (strncmp(line, "PixelWidth=", 11) == 0)
            {
                float pixel_width = strtof(line+11, NULL);
                printf("width=%f\n", pixel_width);
            }
            else if (strncmp(line, "PixelHeight=", 12) == 0)
            {
                float pixel_height = strtof(line+12, NULL);
                printf("height=%f\n", pixel_height);
            }
        }
        // TODO: use rotation
        if (strcmp(section, "[Stage]") == 0)
        {
            if (strncmp(line, "StageT=", 7) == 0)
            {
                float rotation = strtof(line+7, NULL);
                printf("rotation=%f\n", rotation);
            }
        }
        */
        if (strcmp(section, "[PrivateFei]") == 0)
            if (strncmp(line, "DatabarHeight=", 14) == 0)
                *databar_height = atoi(line+14);
        line += line_len+1;
    }

    free(line_mem);
    return 0;
}

correlation_image* load_image(char *filename)
{
    char *file_ext = file_extension(filename);
    correlation_image *img = NULL;
    if (strcmp(file_ext, "tif") == 0 || strcmp(file_ext, "tiff") == 0)
    {
        TIFFSetWarningHandler(NULL);
        TIFF* tif = TIFFOpen(filename, "r");
        if (!tif)
            return NULL;
        uint32_t w, h;
        int databar_height = 0;

        size_t npixels;
        uint32_t* raster;

        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
        
        read_tiff_tags(tif, &databar_height);

        npixels = w * h;
        raster = (uint32_t*) _TIFFmalloc(npixels * sizeof (uint32_t));
        if (raster != NULL)
        {
            if (TIFFReadRGBAImage(tif, w, h, raster, 0))
            {
                img = malloc(sizeof(correlation_image));
                img->width = w;
                img->height = h-databar_height;
                img->img = malloc(sizeof(char)*w*h);
                for (size_t i=0;i<img->width*img->height;i++)
                {
                    uint32_t pixel = raster[i];
                    img->img[i] = convert_grayscale(TIFFGetR(pixel), TIFFGetB(pixel), TIFFGetB(pixel));
                }
            }
            _TIFFfree(raster);
        }
        TIFFClose(tif);
    }
    return img;
}

int do_reconstruction(char *img1_filename, char *img2_filename, char *output_filename)
{
    int result_code = 0;
    correlation_image *img1 = load_image(img1_filename);
    correlation_image *img2 = load_image(img2_filename);
    clock_t start_time = clock();
    clock_t last_operation_time = clock();
    clock_t current_operation_time = clock();



    if (img1 == NULL)
    {
        fprintf(stderr, "Failed to load image %s\n", img1_filename);
        result_code = 1;
        goto cleanup;
    }
    if (img2 == NULL)
    {
        fprintf(stderr, "Failed to load image %s\n", img2_filename);
        result_code = 1;
        goto cleanup;
    }

    correlation_point *img1_points = NULL, *img2_points = NULL;
    size_t img1_points_count, img2_points_count;
    last_operation_time = clock();
    img1_points = fast_detect(img1, &img1_points_count);
    if (img1_points == NULL)
    {
        fprintf(stderr, "Failed to extract points from image %s\n", img1_filename);
        result_code = 1;
        goto cleanup;
    }
    img2_points = fast_detect(img2, &img2_points_count);
    if (img2_points == NULL)
    {
        fprintf(stderr, "Failed to extract points from image %s\n", img2_filename);
        result_code = 1;
        goto cleanup;
    }
    current_operation_time = clock();
    printf("Extracted feature points in %f seconds\n", 1E-6F*(current_operation_time-last_operation_time));
    printf("Image %s has %li feature points\n", img1_filename, img1_points_count);
    printf("Image %s has %li feature points\n", img2_filename, img2_points_count);

cleanup:
    if (img1 != NULL)
        free(img1);
    if (img2 != NULL)
        free(img2);
    if (img1_points != NULL)
        free(img1_points);
    if (img2_points != NULL)
        free(img2_points);
    return result_code;
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Unsupported arguments %i, please run: cybervision <image1> <image2> <output.obj>\n", argc);
        return 1;
    }
    return do_reconstruction(argv[1], argv[2], argv[3]);
}
