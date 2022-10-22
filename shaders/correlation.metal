#include <metal_stdlib>

using namespace metal;

struct Parameters
{
    int img1_width;
    int img1_height;
    int img2_width;
    int img2_height;
    int out_width;
    int out_height;
    float3x3 fundamental_matrix;
    float scale;
    int iteration;
    int corridor_offset;
    int corridor_start;
    int corridor_end;
    int pass;
    int kernel_size;
    float threshold;
    int neighbor_distance;
    float extend_range;
    float min_range;
};

kernel void prepare_initialdata_searchdata(const device Parameters& p [[buffer(0)]], device float* internals[[buffer(3)]], device int* internals_int[[buffer(4)]], uint2 index [[thread_position_in_grid]]) {
    const int x = index.x;
    const int y = index.y;
    const int img1_width = p.img1_width;
    const int img1_height = p.img1_height;

    const int mean_coeff_offset = 0;
    const int corridor_stdev_offset = mean_coeff_offset + img1_width*img1_height;
    const int corridor_range_offset = 0;
    const int neighbor_count_offset = corridor_range_offset + img1_width*img1_height*2;

    if (x < img1_width && y < img1_height)
    {
        internals[mean_coeff_offset + img1_width*y+x] = 0;
        internals[corridor_stdev_offset + img1_width*y+x] = 0;
        internals_int[corridor_range_offset + (img1_width*y+x)*2] = -1;
        internals_int[corridor_range_offset + (img1_width*y+x)*2+1] = -1;
        internals_int[neighbor_count_offset + img1_width*y+x] = 0;
    }
}

kernel void prepare_initialdata_correlation(const device Parameters& p [[buffer(0)]], const device float* images[[buffer(1)]], device float* internals[[buffer(3)]], device int* internals_int[[buffer(4)]], device int* result[[buffer(5)]], uint2 index [[thread_position_in_grid]]) {
    const int x = index.x;
    const int y = index.y;
    const int img1_width = p.img1_width;
    const int img1_height = p.img1_height;
    const int img2_width = p.img2_width;
    const int img2_height = p.img2_height;
    const int kernel_size = p.kernel_size;
    const float kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);

    const int img1_offset = 0;
    const int img2_offset = img1_width*img1_height;

    const int img1_avg_offset = 0;
    const int img1_stdev_offset = img1_avg_offset + img1_width*img1_height;
    const int img2_avg_offset = img1_stdev_offset + img1_width*img1_height;
    const int img2_stdev_offset = img2_avg_offset + img2_width*img2_height;
    const int correlation_offset = img2_stdev_offset + img2_width*img2_height;

    if (x < img1_width && y < img1_height)
    {
        internals[correlation_offset + img1_width*y+x] = 0;
        result[(img1_width*y+x)*2] = -1;
        result[(img1_width*y+x)*2+1] = -1;
    }
    
    if(x >= kernel_size && x < img1_width-kernel_size && y >= kernel_size && y < img1_height-kernel_size)
    {
        float avg = 0;
        float stdev = 0;

        for (int j=-kernel_size;j<=kernel_size;j++)
        {
            for (int i=-kernel_size;i<=kernel_size;i++)
            {
                float value = images[img1_offset + (y+j)*img1_width+(x+i)];
                avg += value;
            }
        }
        avg /= kernel_point_count;

        for (int j=-kernel_size;j<=kernel_size;j++)
        {
            for (int i=-kernel_size;i<=kernel_size;i++)
            {
                float value = images[img1_offset + (y+j)*img1_width+(x+i)];
                float delta = value-avg;
                stdev += delta*delta;
            }
        }
        stdev = sqrt(stdev/kernel_point_count);
        internals[img1_avg_offset + img1_width*y+x] = avg;
        internals[img1_stdev_offset + img1_width*y+x] = stdev;
    }

    if(x >= kernel_size && x < img2_width-kernel_size && y >= kernel_size && y < img2_height-kernel_size)
    {
        float avg = 0;
        float stdev = 0;

        for (int j=-kernel_size;j<=kernel_size;j++)
        {
            for (int i=-kernel_size;i<=kernel_size;i++)
            {
                float value = images[img2_offset + (y+j)*img2_width+(x+i)];
                avg += value;
            }
        }
        avg /= kernel_point_count;

        for (int j=-kernel_size;j<=kernel_size;j++)
        {
            for (int i=-kernel_size;i<=kernel_size;i++)
            {
                float value = images[img2_offset + (y+j)*img2_width+(x+i)];
                float delta = value-avg;
                stdev += delta*delta;
            }
        }
        stdev = sqrt(stdev/kernel_point_count);
        internals[img2_avg_offset + img2_width*y+x] = avg;
        internals[img2_stdev_offset + img2_width*y+x] = stdev;
    }
}

void calculate_epipolar_line(const device Parameters& p, const int x1, const int y1, thread float2& coeff, thread float2& add_const, thread bool& corridor_affects_x, thread bool& corridor_affects_y) {
    const float scale = p.scale;
    const float3 p1 = float3(float(x1)/scale, float(y1)/scale, 1.0);
    const float3 Fp1 = p.fundamental_matrix*p1;
    if (abs(Fp1.x)>abs(Fp1.y)) {
        coeff.x = -Fp1.y/Fp1.x;
        add_const.x = -scale*Fp1.z/Fp1.x;
        corridor_affects_x = true;
        coeff.y = 1.0;
        add_const.y = 0;
        corridor_affects_y = false;
    } else {
        coeff.x = 1.0;
        add_const.x = 0;
        corridor_affects_x = false;
        coeff.y = -Fp1.x/Fp1.y;
        add_const.y = -scale*Fp1.z/Fp1.y;
        corridor_affects_y = true;
    }
}

kernel void prepare_searchdata(const device Parameters& p [[buffer(0)]], const device int* previous_result[[buffer(2)]], device float* internals[[buffer(3)]], device int* internals_int[[buffer(4)]], uint2 index [[thread_position_in_grid]]) {
    const int x = index.x;
    const int y = index.y;
    const int img1_width = p.img1_width;
    const int img1_height = p.img1_height;
    const int img2_width = p.img2_width;
    const int img2_height = p.img2_height;
    const int out_width = p.out_width;
    const int out_height = p.out_height;
    const int kernel_size = p.kernel_size;
    const int corridor_start = p.corridor_start;
    const int corridor_end = p.corridor_end;
    const int neighbor_distance = p.neighbor_distance;
    const float extend_range = p.extend_range;
    const float min_range = p.min_range;
    const int pass = p.pass;
    const int corridor_range_offset = 0;
    const int neighbor_count_offset = corridor_range_offset + img1_width*img1_height*2;
    const int mean_coeff_offset = 0;
    const int corridor_stdev_offset = mean_coeff_offset + img1_width*img1_height;
    float2 coeff, add_const;
    bool corridor_affects_x, corridor_affects_y;
    calculate_epipolar_line(p, x, y, coeff, add_const, corridor_affects_x, corridor_affects_y);
    const bool corridor_vertical = abs(coeff.y) > abs(coeff.x);

    const float scale = p.scale;
    const float inv_scale = 1.0/scale;
    int x_min = int(floor((int(x)-neighbor_distance)*inv_scale));
    int x_max = int(ceil((int(x)+neighbor_distance)*inv_scale));
    int y_min = int(float(y)*inv_scale)+corridor_start;
    int y_max = int(float(y)*inv_scale)+corridor_end;

    x_min = min(max(0, x_min), out_width);
    x_max = min(max(0, x_max), out_width);
    y_min = min(max(0, y_min), out_height);
    y_max = min(max(0, y_max), out_height);

    int neighbor_count = internals_int[neighbor_count_offset+y*img1_width+x];
    if (pass == 1 && neighbor_count<=0)
        return;

    float mid_corridor = internals[mean_coeff_offset + y*img1_width+x];
    float range_stdev = internals[corridor_stdev_offset + y*img1_width+x];
    if (pass == 1)
        mid_corridor /= float(neighbor_count);
    for (int j=y_min;j<y_max;j++)
    {
        for (int i=x_min;i<x_max;i++)
        {
            int out_pos = j*out_width + i;
            float x2 = scale*float(previous_result[out_pos*2]);
            float y2 = scale*float(previous_result[out_pos*2+1]);
            if (x2<0 || y2<0)
                continue;

            int corridor_pos = int(corridor_vertical? round((y2-add_const.y)/coeff.y):round((x2-add_const.x)/coeff.x));
            if (pass == 0) {
                mid_corridor += float(corridor_pos);
                neighbor_count++;
            } else if (pass == 1) {
                float delta = float(corridor_pos)-mid_corridor;
                range_stdev += delta*delta/float(neighbor_count);
            }
        }
    }
    
    if (pass == 0) {
        internals[mean_coeff_offset + y*img1_width+x] = mid_corridor;
        internals_int[neighbor_count_offset+y*img1_width+x] = neighbor_count;
        return;
    }

    internals[corridor_stdev_offset + y*img1_width+x] = range_stdev;
    const int corridor_center = int(round(mid_corridor));
    const int corridor_length = int(round(min_range+sqrt(range_stdev)*extend_range));
    int min_pos = kernel_size;
    int max_pos = corridor_vertical? img2_height-kernel_size : img2_width-kernel_size;
    min_pos = min(max(min_pos, corridor_center - corridor_length), max_pos);
    max_pos = min(max(min_pos, corridor_center + corridor_length), max_pos);
    internals_int[corridor_range_offset + (y*img1_width+x)*2] = min_pos;
    internals_int[corridor_range_offset + (y*img1_width+x)*2+1] = max_pos;
}

kernel void cross_correlate(const device Parameters& p [[buffer(0)]], const device float* images[[buffer(1)]], device float* internals[[buffer(3)]], device int* internals_int[[buffer(4)]], device int* result[[buffer(5)]], uint2 index [[thread_position_in_grid]]) {
    const int x1 = index.x;
    const int y1 = index.y;
    const int img1_width = p.img1_width;
    const int img1_height = p.img1_height;
    const int img2_width = p.img2_width;
    const int img2_height = p.img2_height;
    const float scale = p.scale;
    const int iteration = p.iteration;
    const int corridor_offset = p.corridor_offset;
    const int corridor_start = p.corridor_start;
    const int corridor_end = p.corridor_end;
    const int kernel_size = p.kernel_size;
    const float threshold = p.threshold;
    const float kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);

    if(x1 < kernel_size || x1 >= img1_width-kernel_size || y1 < kernel_size ||  y1 >= img1_height-kernel_size)
        return;

    const uint img1_offset = 0;
    const uint img2_offset = img1_width*img1_height;

    const int img1_avg_offset = 0;
    const int img1_stdev_offset = img1_avg_offset + img1_width*img1_height;
    const int img2_avg_offset = img1_stdev_offset + img1_width*img1_height;
    const int img2_stdev_offset = img2_avg_offset + img2_width*img2_height;
    const int correlation_offset = img2_stdev_offset + img2_width*img2_height;
    const int corridor_range_offset = 0;
    const int out_pos = img1_width*y1 + x1;
    
    const int min_pos = internals_int[corridor_range_offset + out_pos*2];
    const int max_pos = internals_int[corridor_range_offset + out_pos*2+1];
    if (iteration > 0 && (min_pos<0 || max_pos<0))
        return;

    float avg1 = internals[img1_avg_offset + img1_width*y1+x1];
    float stdev1 = internals[img1_stdev_offset + img1_width*y1+x1];

    float best_corr = 0;
    float2 best_match = float2(-1, -1);
    
    float2 coeff, add_const;
    bool corridor_affects_x, corridor_affects_y;
    calculate_epipolar_line(p, x1, y1, coeff, add_const, corridor_affects_x, corridor_affects_y);
    for (int corridor_pos=corridor_start;corridor_pos<corridor_end;corridor_pos++)
    {
        if (iteration > 0 && (corridor_pos<min_pos || corridor_pos>max_pos))
            continue;
        const int x2 = int(round(coeff.x*float(corridor_pos)+add_const.x)) + (corridor_affects_x?corridor_offset:0);
        const int y2 = int(round(coeff.y*float(corridor_pos)+add_const.y)) + (corridor_affects_y?corridor_offset:0);
        if (x2 < kernel_size || x2 >= img2_width-kernel_size || y2 < kernel_size ||  y2 >= img2_height-kernel_size)
            continue;

        const float avg2 = internals[img2_avg_offset + img2_width*y2 + x2];
        const float stdev2 = internals[img2_stdev_offset + img2_width*y2 + x2];

        float corr = 0;
        for (int j=-kernel_size;j<=kernel_size;j++)
        {
            for (int i=-kernel_size;i<=kernel_size;i++)
            {
                float delta1 = images[img1_offset + (y1+j)*img1_width+(x1+i)] - avg1;
                float delta2 = images[img2_offset + (y2+j)*img2_width+(x2+i)] - avg2;
                corr += delta1*delta2;
            }
        }
        corr = corr/(stdev1*stdev2*kernel_point_count);

        if (corr >= threshold && corr > best_corr)
        {
            best_match.x = round(float(x2)/scale);
            best_match.y = round(float(y2)/scale);
            best_corr = corr;
        }
    }

    float current_corr = internals[correlation_offset + img1_width*y1 + x1];
    if (best_corr >= threshold && best_corr > current_corr)
    {
        internals[correlation_offset + img1_width*y1 + x1] = best_corr;
        result[out_pos*2] = int(best_match.x);
        result[out_pos*2+1] = int(best_match.y);
    }
}
