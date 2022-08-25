#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "fast_detector.h"
#include "system.h"
#include "correlation.h"
#include "gpu_correlation.h"
#include "configuration.h"
#include "triangulation.h"
#include "image.h"

double diff_seconds(struct timespec end, struct timespec start)
{
    return (double)(end.tv_nsec-start.tv_nsec)*1E-9 + (double)(end.tv_sec-start.tv_sec);
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

float optimal_keypoint_scale(correlation_image *img)
{
    int min_dimension = img->width<img->height? img->width:img->height;
    int scale = 1;
    while(min_dimension/(1<<scale)>cybervision_keypoint_scale_min_size)
        scale++;
    scale = scale>0? scale-1 : 1;
    return 1.0F/(1<<scale);
}

int optimal_scale_steps(correlation_image *img)
{
    int min_dimension = img->width<img->height? img->width:img->height;
    int scale = 1;
    while(min_dimension/(1<<scale)>cybervision_crosscorrelation_scale_min_size)
        scale++;
    return scale-1;
}

int do_reconstruction(char *img1_filename, char *img2_filename, char *output_filename, float depth_scale, correlation_mode mode)
{
    // TODO: print error messages from failed threads
    int result_code = 0;
    correlation_image *img1 = load_image(img1_filename);
    correlation_image *img2 = load_image(img2_filename);
    struct timespec start_time, last_operation_time, current_operation_time;
    int num_threads = cpu_cores();
    progressbar_str = malloc(sizeof(char)*(progressbar_str_width()+1));
    progressbar_str[0] = '\0';
    
    correlation_point *points1 = NULL, *points2 = NULL;

    match_task m_task = {0};
    ransac_task r_task = {0};
    cross_correlate_task cc_task = {0};

    int (*cross_correlate_start)(cross_correlate_task*);
    int (*cross_correlate_complete)(cross_correlate_task*);

    if (mode == CORRELATION_MODE_CPU)
    {
        cross_correlate_start = &cpu_correlation_cross_correlate_start;
        cross_correlate_complete = &cpu_correlation_cross_correlate_complete;
    }
    else if (mode == CORRELATION_MODE_GPU)
    {
        cross_correlate_start = &gpu_correlation_cross_correlate_start;
        cross_correlate_complete = &gpu_correlation_cross_correlate_complete;
    }
    else
    {
        fprintf(stderr, "Unsupported correlation mode %i\n", mode);
        result_code = 1;
        goto cleanup;
    }

    timespec_get(&start_time, TIME_UTC);
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


    float keypoint_scale = optimal_keypoint_scale(img1);
    size_t points1_size, points2_size;
    {
        correlation_image fast_img = {0};
        resize_image(img1, &fast_img, keypoint_scale);
        timespec_get(&last_operation_time, TIME_UTC);
        points1 = fast_detect(&fast_img, &points1_size);
        free(fast_img.img);
        if (points1 == NULL)
        {
            fprintf(stderr, "Failed to extract points from image %s\n", img1_filename);
            result_code = 1;
            goto cleanup;
        }
        resize_image(img2, &fast_img, keypoint_scale);
        points2 = fast_detect(&fast_img, &points2_size);
        free(fast_img.img);
        if (points2 == NULL)
        {
            fprintf(stderr, "Failed to extract points from image %s\n", img2_filename);
            result_code = 1;
            goto cleanup;
        }
        timespec_get(&current_operation_time, TIME_UTC);
        printf("Extracted feature points in %.1f seconds\n", diff_seconds(current_operation_time, last_operation_time));
        printf("Image %s has %zi feature points\n", img1_filename, points1_size);
        printf("Image %s has %zi feature points\n", img2_filename, points2_size);
    }

    {
        m_task.num_threads = num_threads;
        resize_image(img1, &m_task.img1, keypoint_scale);
        resize_image(img2, &m_task.img2, keypoint_scale);
        m_task.points1 = points1;
        m_task.points2 = points2;
        m_task.points1_size = points1_size;
        m_task.points2_size = points2_size;

        timespec_get(&last_operation_time, TIME_UTC);
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
        timespec_get(&current_operation_time, TIME_UTC);
        printf("Matched keypoints in %.1f seconds\n", diff_seconds(current_operation_time, last_operation_time));
        printf("Found %zi matches\n", m_task.matches_count);
        free(m_task.img1.img);
        m_task.img1.img = NULL;
        free(m_task.img2.img);
        m_task.img2.img = NULL;
    }

    {
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

        timespec_get(&last_operation_time, TIME_UTC);
        if (!correlation_ransac_start(&r_task))
        {
            fprintf(stderr, "Failed to start RANSAC task");
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
        timespec_get(&current_operation_time, TIME_UTC);
        printf("Completed RANSAC fitting in %.1f seconds\n", diff_seconds(current_operation_time, last_operation_time));
        printf("Kept %zi matches\n", r_task.result_matches_count);

        if (r_task.result_matches_count == 0)
        {
            fprintf(stderr, "No reliable matches found");
            result_code = 1;
            goto cleanup;
        }

        free(points1);
        points1 = NULL;
        free(points2);
        points2 = NULL;
        free(r_task.matches);
        r_task.matches = NULL;
    }

    {
        float total_percent = 0.0F;
        int scale_steps = optimal_scale_steps(img1);
        for(int i = 0; i<=scale_steps; i++)
        {
            float scale = 1.0F/(float)(1<<(scale_steps-i));
            total_percent += scale*scale;
        }
        cc_task.dir_x = r_task.dir_x;
        cc_task.dir_y = r_task.dir_y;
        cc_task.num_threads = num_threads;
        cc_task.out_width = img1->width;
        cc_task.out_height = img1->height;
        cc_task.out_points = malloc(sizeof(float)*cc_task.out_width*cc_task.out_height);
        for (int i=0;i<img1->width*img1->height;i++)
            cc_task.out_points[i] = NAN;

        timespec_get(&last_operation_time, TIME_UTC);
        float total_percent_complete = 0.0F;
        for(int i = 0; i<=scale_steps; i++)
        {
            float scale = 1.0F/(float)(1<<(scale_steps-i));
            resize_image(img1, &cc_task.img1, scale);
            resize_image(img2, &cc_task.img2, scale);
            cc_task.iteration = i;
            cc_task.scale = scale;
            if (!(*cross_correlate_complete)(&cc_task))
            {
                fprintf(stderr, "Failed to complete cross correlation task: %s", cc_task.error!=NULL? cc_task.error : "unknown error");
                result_code = 1;
                goto cleanup;
            }
            if (!(*cross_correlate_start)(&cc_task))
            {
                fprintf(stderr, "Failed to start cross correlation task: %s", cc_task.error!=NULL? cc_task.error : "unknown error");
                result_code = 1;
                goto cleanup;
            }
            while(!cc_task.completed)
            {
                sleep_ms(200);
                float percent_complete = total_percent_complete + cc_task.percent_complete*scale*scale/total_percent;
                show_progressbar(percent_complete);
            }
            total_percent_complete = total_percent_complete + 100.0F*scale*scale/total_percent;
            if (!(*cross_correlate_complete)(&cc_task))
            {
                fprintf(stderr, "Failed to complete cross correlation task: %s", cc_task.error!=NULL? cc_task.error : "unknown error");
                result_code = 1;
                goto cleanup;
            }
            free(cc_task.img1.img);
            cc_task.img1.img = NULL;
            free(cc_task.img2.img);
            cc_task.img2.img = NULL;
        }
        reset_progressbar();
        timespec_get(&current_operation_time, TIME_UTC);
        printf("Completed surface generation in %.1f seconds\n", diff_seconds(current_operation_time, last_operation_time));

        free(img1->img);
        img1->img = NULL;
        free(img2->img);
        img2->img = NULL;
        free(img1);
        img1 = NULL;
        free(img2);
        img2 = NULL;
    }

    timespec_get(&last_operation_time, TIME_UTC);
    {
        surface_data surf = {0};
        FILE *output_file = NULL;
        char* output_fileextension = file_extension(output_filename);

        surf.depth = cc_task.out_points;
        cc_task.out_points = NULL;
        surf.width = cc_task.out_width;
        surf.height = cc_task.out_height;        
        for (int i=0;i<surf.width*surf.height;i++)
        {
            surf.depth[i] = surf.depth[i]*depth_scale;
        }

        if (strcasecmp(output_fileextension, "obj") == 0)
        {
            output_file = fopen(output_filename, "w");
            if (!triangulation_triangulate(&surf, output_file))
            {
                fprintf(stderr, "Failed to triangulate points");
                result_code = 1;
            }
            fclose(output_file);
        }
        else if (strcasecmp(output_fileextension, "png") == 0)
        {
            if (!triangulation_interpolate(&surf))
            {
                fprintf(stderr, "Failed to interpolate points");
                result_code = 1;
            }
            if (!save_surface(&surf, output_filename))
            {
                fprintf(stderr, "Failed to save output image");
                result_code = 1;
            }
        }
        else
        {
            fprintf(stderr, "Unsupported output file extension %s", output_fileextension);
            result_code = 1;
        }
        free(surf.depth);
    }
    timespec_get(&current_operation_time, TIME_UTC);
    printf("Completed triangulation in %.1f seconds\n", diff_seconds(current_operation_time, last_operation_time));

    printf("Completed reconstruction in %.1f seconds\n", diff_seconds(current_operation_time, start_time));
cleanup:
    free(progressbar_str);
    if (img1 != NULL && img1->img != NULL)
        free(img1->img);
    if (img2 != NULL && img2->img != NULL)
        free(img2->img);
    if (img1 != NULL)
        free(img1);
    if (img2 != NULL)
        free(img2);
    if (m_task.matches != NULL)
        free(m_task.matches);
    if (m_task.img1.img != NULL)
        free(m_task.img1.img);
    if (m_task.img2.img != NULL)
        free(m_task.img2.img);
    if (r_task.matches != NULL)
        free(r_task.matches);
    if (points1 != NULL)
        free(points1);
    if (points2 != NULL)
        free(points2);
    if (cc_task.img1.img != NULL)
        free(cc_task.img1.img);
    if (cc_task.img2.img != NULL)
        free(cc_task.img2.img);
    if (cc_task.out_points != NULL)
        free(cc_task.out_points);
    return result_code;
}

int main(int argc, char *argv[])
{
    char *image1_filename, *image2_filename, *output_filename;
    float scale = -1.0F;
    correlation_mode mode = CORRELATION_MODE_GPU;
    if (argc < 4)
    {
        fprintf(stderr, "Unsupported arguments %i, please run: cybervision [--scale=<scale>] [--mode=<cpu|gpu>] <image1> <image2> <output>\n", argc);
        return 1;
    }
    for (int i=1; i<argc-3; i++)
    {
        char* arg = argv[i];
        if (strncmp("--scale=", argv[i], 8) == 0)
        {
            scale = strtof(arg+8, NULL);
            printf("Using scale %f from commandline\n", scale);
        }
        else if (strncmp("--mode=", argv[i], 7) == 0)
        {
            char* mode_param = arg+7;
            if (strcmp(mode_param, "cpu")==0)
            {
                mode = CORRELATION_MODE_CPU;
            }
            else if (strcmp(mode_param, "gpu")==0)
            {
                mode = CORRELATION_MODE_GPU;
            }
            else
            {
                fprintf(stderr, "Unsupported correlation mode %s, please use --mode=cpu or --mode=gpu\n", mode_param);
                return 1;
            }
            printf("Using %s correlation mode from commandline\n", mode_param);
        }
        else
        {
            fprintf(stderr, "Unknown argument: %s\n", arg);
            return 1;
        }
    }
    image1_filename = argv[argc-3];
    image2_filename = argv[argc-2];
    output_filename = argv[argc-1];
    return do_reconstruction(image1_filename, image2_filename, output_filename, scale, mode);
}
