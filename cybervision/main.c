#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "fast_detector.h"
#include "system.h"
#include "correlation.h"
#include "fundamental_matrix.h"
#include "gpu_correlation.h"
#include "configuration.h"
#include "triangulation.h"
#include "surface.h"
#include "image.h"

#define MATCH_BUCKET_GROW_SIZE 1000

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
    int scale = 0;
    while(min_dimension/(1<<scale)>cybervision_keypoint_scale_min_size)
        scale++;
    scale = scale>0? scale-1 : 0;
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

int do_reconstruction(char *img1_filename, char *img2_filename, char *output_filename, float depth_scale, projection_mode proj_mode, correlation_mode corr_mode, interpolation_mode interp_mode)
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
    triangulation_task t_task = {0};

    int (*cross_correlate_start)(cross_correlate_task*);
    int (*cross_correlate_complete)(cross_correlate_task*);

    if (corr_mode == CORRELATION_MODE_CPU)
    {
        cross_correlate_start = &cpu_correlation_cross_correlate_start;
        cross_correlate_complete = &cpu_correlation_cross_correlate_complete;
    }
    else if (corr_mode == CORRELATION_MODE_GPU)
    {
        cross_correlate_start = &gpu_correlation_cross_correlate_start;
        cross_correlate_complete = &gpu_correlation_cross_correlate_complete;
    }
    else
    {
        fprintf(stderr, "Unsupported correlation mode %i\n", corr_mode);
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
            fprintf(stderr, "Failed to start point matching task\n");
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
        const size_t grid_size = cybervision_ransac_match_grid_size;
        const size_t grid_step_x = img1->width/grid_size, grid_step_y = img1->height/grid_size;
        size_t *bucket_match_limit = malloc(sizeof(size_t)*grid_size*grid_size);
        r_task.num_threads = num_threads;
        r_task.proj_mode = proj_mode;
        r_task.match_buckets = malloc(sizeof(ransac_match_bucket)*grid_size*grid_size);
        r_task.keypoint_scale = keypoint_scale;
        for (size_t i=0;i<grid_size*grid_size;i++)
        {
            bucket_match_limit[i] = 0;
            r_task.match_buckets[i].matches = NULL;
            r_task.match_buckets[i].matches_count = 0;
        }
        for (size_t i=0;i<m_task.matches_count;i++)
        {
            correlation_match *m = &m_task.matches[i];
            int p1 = m->point1, p2 = m->point2;

            ransac_match converted_match;
            converted_match.x1 = (int)roundf((float)points1[p1].x/keypoint_scale);
            converted_match.y1 = (int)roundf((float)points1[p1].y/keypoint_scale);
            converted_match.x2 = (int)roundf((float)points2[p2].x/keypoint_scale);
            converted_match.y2 = (int)roundf((float)points2[p2].y/keypoint_scale);
            
            size_t grid_x = converted_match.x1/grid_step_x;
            size_t grid_y = converted_match.x1/grid_step_y;
            grid_x = grid_x<grid_size?grid_x:grid_size-1;
            grid_y = grid_y<grid_size?grid_y:grid_size-1;
            size_t grid_pos = grid_y*grid_size+grid_x;
            ransac_match_bucket *bucket = &r_task.match_buckets[grid_pos];
            if (bucket->matches_count >= bucket_match_limit[grid_pos])
            {
                bucket_match_limit[grid_pos] += MATCH_BUCKET_GROW_SIZE;
                size_t new_size = sizeof(ransac_match)*bucket_match_limit[grid_pos];
                bucket->matches = bucket->matches == NULL ? malloc(new_size) : realloc(bucket->matches, new_size);
            }
            bucket->matches[bucket->matches_count++] = converted_match;
        }
        r_task.match_buckets_count = 0;
        for (size_t i=0;i<grid_size*grid_size;i++)
        {
            if (r_task.match_buckets[i].matches != NULL)
            {
                r_task.match_buckets_count++;
                continue;
            }
            for (size_t j=i+1;j<grid_size*grid_size;j++)
            {
                if (r_task.match_buckets[j].matches == NULL)
                    continue;
                r_task.match_buckets_count++;
                r_task.match_buckets[i] = r_task.match_buckets[j];
                r_task.match_buckets[j].matches = NULL;
                break;
            }
        }
        free(m_task.matches);
        m_task.matches = NULL;
        free(bucket_match_limit);
        r_task.match_buckets = realloc(r_task.match_buckets, sizeof(ransac_match_bucket)*r_task.match_buckets_count);

        timespec_get(&last_operation_time, TIME_UTC);
        if (!correlation_ransac_start(&r_task))
        {
            fprintf(stderr, "Failed to start RANSAC task\n");
            result_code = 1;
            goto cleanup;
        }
        while(!r_task.completed)
        {
            sleep_ms(200);
            show_progressbar(r_task.percent_complete);
        }
        reset_progressbar();
        if (!correlation_ransac_complete(&r_task))
        {
            fprintf(stderr, "Failed to complete RANSAC task: %s\n", r_task.error!=NULL? r_task.error : "unknown error");
            result_code = 1;
            goto cleanup;
        }
        timespec_get(&current_operation_time, TIME_UTC);
        printf("Completed RANSAC fitting in %.1f seconds\n", diff_seconds(current_operation_time, last_operation_time));
        printf("Kept %zi matches\n", r_task.result_matches_count);

        if (r_task.result_matches_count == 0)
        {
            fprintf(stderr, "No reliable matches found\n");
            result_code = 1;
            goto cleanup;
        }

        free(points1);
        points1 = NULL;
        free(points2);
        points2 = NULL;
        for(size_t i=0;i<r_task.match_buckets_count;i++)
        {
            free(r_task.match_buckets[i].matches);
            r_task.match_buckets[i].matches = NULL;
        }
        free(r_task.match_buckets);
        r_task.match_buckets = NULL;
    }

    {
        float total_percent = 0.0F;
        int scale_steps = optimal_scale_steps(img1);
        for(int i = 0; i<=scale_steps; i++)
        {
            float scale = 1.0F/(float)(1<<(scale_steps-i));
            total_percent += scale*scale;
        }
        for (size_t i=0;i<9;i++)
            cc_task.fundamental_matrix[i] = r_task.fundamental_matrix[i];
        cc_task.num_threads = num_threads;
        cc_task.out_width = img1->width;
        cc_task.out_height = img1->height;
        cc_task.proj_mode = proj_mode;
        cc_task.correlated_points = malloc(sizeof(int)*cc_task.out_width*cc_task.out_height*2);
        for (size_t i=0;i<(size_t)cc_task.out_width*cc_task.out_height*2;i++)
            cc_task.correlated_points[i] = -1;

        timespec_get(&last_operation_time, TIME_UTC);
        if (corr_mode == CORRELATION_MODE_GPU && !gpu_correlation_cross_correlate_init(&cc_task, img1->width*img1->height, img2->width*img2->height))
        {
            fprintf(stderr, "Failed to initialize GPU: %s\n", cc_task.error!=NULL? cc_task.error : "unknown error");
            result_code = 1;
            goto cleanup;
        }
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
                fprintf(stderr, "Failed to complete cross correlation task: %s\n", cc_task.error!=NULL? cc_task.error : "unknown error");
                result_code = 1;
                goto cleanup;
            }
            if (!(*cross_correlate_start)(&cc_task))
            {
                fprintf(stderr, "Failed to start cross correlation task: %s\n", cc_task.error!=NULL? cc_task.error : "unknown error");
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
                fprintf(stderr, "Failed to complete cross correlation task: %s\n", cc_task.error!=NULL? cc_task.error : "unknown error");
                result_code = 1;
                goto cleanup;
            }
            if (cc_task.img1.img != NULL)
            {
                free(cc_task.img1.img);
                cc_task.img1.img = NULL;
            }
            if (cc_task.img2.img != NULL)
            {
                free(cc_task.img2.img);
                cc_task.img2.img = NULL;
            }
        }
        if (corr_mode == CORRELATION_MODE_GPU && !gpu_correlation_cross_correlate_cleanup(&cc_task))
        {
            fprintf(stderr, "Failed to cleanup GPU: %s\n", cc_task.error!=NULL? cc_task.error : "unknown error");
            result_code = 1;
            goto cleanup;
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
        t_task.num_threads = num_threads;
        t_task.correlated_points = cc_task.correlated_points;
        t_task.out_depth = malloc(sizeof(float)*cc_task.out_width*cc_task.out_height);
        t_task.width = cc_task.out_width;
        t_task.height = cc_task.out_height;
        t_task.depth_scale = depth_scale;
        t_task.proj_mode = proj_mode;
        for (size_t i=0;i<9;i++)
            t_task.fundamental_matrix[i] = r_task.fundamental_matrix[i];

        if (!triangulation_start(&t_task))
        {
            fprintf(stderr, "Failed to start point triangulation task\n");
            result_code = 1;
            goto cleanup;
        }
        while(!t_task.completed)
        {
            sleep_ms(200);
            show_progressbar(t_task.percent_complete);
        }
        reset_progressbar();
        if (!triangulation_complete(&t_task))
        {
            fprintf(stderr, "Failed to complete point triangulation task: %s\n", t_task.error!=NULL? t_task.error : "unknown error");
            result_code = 1;
            goto cleanup;
        }
        
        timespec_get(&current_operation_time, TIME_UTC);
        printf("Completed point triangulation in %.1f seconds\n", diff_seconds(current_operation_time, last_operation_time));

        free(cc_task.correlated_points);
        cc_task.correlated_points = NULL;
    }

    timespec_get(&last_operation_time, TIME_UTC);
    {
        surface_data surf = {0};

        surf.depth = t_task.out_depth;
        t_task.out_depth = NULL;
        surf.width = t_task.width;
        surf.height = t_task.height;

        if (!surface_output(surf, output_filename, interp_mode))
        {
            free(surf.depth);
            fprintf(stderr, "Failed to output result\n");
            result_code = 1;
            goto cleanup;
        }
        free(surf.depth);
    }
    timespec_get(&current_operation_time, TIME_UTC);
    printf("Completed 3D surface generation in %.1f seconds\n", diff_seconds(current_operation_time, last_operation_time));

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
    if (r_task.match_buckets != NULL)
        for(size_t i=0;i<r_task.match_buckets_count;i++)
            if (r_task.match_buckets[i].matches != NULL)
                free(r_task.match_buckets[i].matches);
    if (r_task.match_buckets != NULL)
        free(r_task.match_buckets);
    if (points1 != NULL)
        free(points1);
    if (points2 != NULL)
        free(points2);
    if (cc_task.img1.img != NULL)
        free(cc_task.img1.img);
    if (cc_task.img2.img != NULL)
        free(cc_task.img2.img);
    if (cc_task.correlated_points != NULL)
        free(cc_task.correlated_points);
    if (t_task.out_depth != NULL)
        free(t_task.out_depth);
    return result_code;
}

int main(int argc, char *argv[])
{
    char *image1_filename, *image2_filename, *output_filename;
    float scale = -1.0F;
    projection_mode proj_mode = PROJECTION_MODE_PARALLEL;
    correlation_mode corr_mode = cybervision_crosscorrelation_default_mode;
    interpolation_mode interp_mode = INTERPOLATION_DELAUNAY;
    if (argc < 4)
    {
        fprintf(stderr, "Unsupported arguments %i, please run: cybervision [--scale=<scale>] [--mode=<cpu|gpu>] [--interpolation=<none|delaunay>] [--projection=<parallel|perspective>] <image1> <image2> <output>\n", argc);
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
                corr_mode = CORRELATION_MODE_CPU;
            }
            else if (strcmp(mode_param, "gpu")==0)
            {
                corr_mode = CORRELATION_MODE_GPU;
            }
            else
            {
                fprintf(stderr, "Unsupported correlation mode %s, please use --mode=cpu or --mode=gpu\n", mode_param);
                return 1;
            }
            printf("Using %s correlation mode from commandline\n", mode_param);
        }
        else if (strncmp("--interpolation=", argv[i], 16) == 0)
        {
            char* mode_param = arg+16;
            if (strcmp(mode_param, "delaunay")==0)
            {
                interp_mode = INTERPOLATION_DELAUNAY;
            }
            else if (strcmp(mode_param, "none")==0)
            {
                interp_mode = INTERPOLATION_NONE;
            }
            else
            {
                fprintf(stderr, "Unsupported interpolation mode %s, please use --interpolation=none or --interpolation=delaunay\n", mode_param);
                return 1;
            }
            printf("Using %s interpolation mode from commandline\n", mode_param);
        }
        else if (strncmp("--projection=", argv[i], 13) == 0)
        {
            char* projection_param = arg+13;
            if (strcmp(projection_param, "parallel")==0)
            {
                proj_mode = PROJECTION_MODE_PARALLEL;
            }
            else if (strcmp(projection_param, "perspective")==0)
            {
                proj_mode = PROJECTION_MODE_PERSPECTIVE;
            }
            else
            {
                fprintf(stderr, "Unsupported projection mode %s, please use --projection=parallel or --projection=perspective\n", projection_param);
                return 1;
            }
            printf("Using %s projection mode from commandline\n", projection_param);
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
    return do_reconstruction(image1_filename, image2_filename, output_filename, scale, proj_mode, corr_mode, interp_mode);
}
