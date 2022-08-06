#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <tiffio.h>

#include "fast_detector.h"
#include "system.h"
#include "correlation.h"
#include "configuration.h"

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

const int progressbar_width = 60;
char* progressbar_str = NULL;

static inline int progressbar_str_width()
{
    return 1+progressbar_width+2+7;
}

void reset_progressbar()
{
    char *progressbar_curr = progressbar_str;
    if (*progressbar_str != '\0')
        printf("\r");
    for (int i=0;i<progressbar_str_width();i++)
        progressbar_str[i] = ' ';
    progressbar_str[progressbar_str_width()] = '\0';
    printf("%s\r", progressbar_curr);
    fflush(stdout);
    progressbar_str[0] = '\0';
}

void show_progressbar(float percent)
{
    int x = (int)(percent/100.0*progressbar_width);
    char* progressbar_curr = progressbar_str;
    
    if (*progressbar_str != '\0')
        printf("\r");
    progressbar_str[0] = '\0';

    *(progressbar_curr++) = '[';
    for (int i=0;i<progressbar_width;i++)
        *(progressbar_curr++) = i<=x? '#':' ';
    sprintf(progressbar_curr, "] %2.1f%%", percent);
    printf("%s", progressbar_str);
    fflush(stdout);
}

int do_reconstruction(char *img1_filename, char *img2_filename, char *output_filename)
{
    int result_code = 0;
    correlation_image *img1 = load_image(img1_filename);
    correlation_image *img2 = load_image(img2_filename);
    time_t start_time, last_operation_time, current_operation_time;
    int num_threads = cpu_cores();
    progressbar_str = malloc(sizeof(char)*(progressbar_str_width()+1));
    progressbar_str[0] = '\0';
    
    time(&start_time);

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

    correlation_point *points1 = NULL, *points2 = NULL;
    size_t points1_size, points2_size;
    time(&last_operation_time);
    points1 = fast_detect(img1, &points1_size);
    if (points1 == NULL)
    {
        fprintf(stderr, "Failed to extract points from image %s\n", img1_filename);
        result_code = 1;
        goto cleanup;
    }
    points2 = fast_detect(img2, &points2_size);
    if (points2 == NULL)
    {
        fprintf(stderr, "Failed to extract points from image %s\n", img2_filename);
        result_code = 1;
        goto cleanup;
    }
    time(&current_operation_time);
    printf("Extracted feature points in %.1f seconds\n", difftime(current_operation_time, last_operation_time));
    printf("Image %s has %zi feature points\n", img1_filename, points1_size);
    printf("Image %s has %zi feature points\n", img2_filename, points2_size);

    match_task m_task = {0};
    m_task.num_threads = num_threads;
    m_task.img1 = *img1;
    m_task.img2 = *img2;
    m_task.points1 = points1;
    m_task.points2 = points2;
    m_task.points1_size = points1_size;
    m_task.points2_size = points2_size;

    time(&last_operation_time);
    if (!correlation_match_points_start(&m_task))
    {
        fprintf(stderr, "Failed to start point matching task");
        result_code = 1;
        goto cleanup;
    }
    while(!m_task.completed)
    {
        sleep_ms(200);
        show_progressbar(m_task.percent_complete);
    }
    reset_progressbar();
    correlation_match_points_complete(&m_task);
    time(&current_operation_time);
    printf("Matched keypoints in %.1f seconds\n", difftime(current_operation_time, last_operation_time));
    printf("Found %zi matches\n", m_task.matches_count);

    ransac_task r_task = {0};
    r_task.num_threads = num_threads;
    r_task.matches = malloc(sizeof(ransac_match)*m_task.matches_count);
    r_task.matches_count = 0;
    for (size_t i=0;i<m_task.matches_count;i++)
    {
        correlation_match m = m_task.matches[i];
        int p1 = m.point1, p2 = m.point2;
        float corr;
        float dx, dy, length;

        ransac_match *converted_match = &(r_task.matches[r_task.matches_count]);
        converted_match->x1 = points1[p1].x;
        converted_match->y1 = points1[p1].y;
        converted_match->x2 = points2[p2].x;
        converted_match->y2 = points2[p2].y;

        dx = (float)(converted_match->x2 - converted_match->x1);
        dy = (float)(converted_match->y2 - converted_match->y1);
        length = sqrtf(dx*dx + dy*dy);

        if (length >= cybervision_ransac_min_length)
            r_task.matches_count++;
    }
    free(m_task.matches);
    m_task.matches = NULL;

    time(&last_operation_time);
    if (!correlation_ransac_start(&r_task))
    {
        fprintf(stderr, "Failed to start point matching task");
        result_code = 1;
        goto cleanup;
    }
    while(!r_task.completed)
    {
        sleep_ms(200);
        show_progressbar(r_task.percent_complete);
    }
    reset_progressbar();
    correlation_ransac_complete(&r_task);
    time(&current_operation_time);
    printf("Completed RANSAC fitting in %.1f seconds\n", difftime(current_operation_time, last_operation_time));
    printf("Kept %zi matches\n", r_task.result_matches_count);

    free(points1);
    points1 = NULL;
    free(points2);
    points2 = NULL;

    printf("Completed reconstruction in %.1f seconds\n", difftime(current_operation_time, start_time));
cleanup:
    free(progressbar_str);
    if (img1 != NULL)
        free(img1);
    if (img2 != NULL)
        free(img2);
    if (points1 != NULL)
        free(points1);
    if (points2 != NULL)
        free(points2);
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
