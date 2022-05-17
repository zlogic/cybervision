#include <metal_stdlib>

using namespace metal;

struct Parameters
{
    int img1_width;
    int img1_height;
    int img2_width;
    int img2_height;
    float dir_x;
    float dir_y;
    int corridor_offset;
    int corridor_start;
    int corridor_end;
    int kernel_size;
    float threshold;
};

kernel void prepare_initialdata(const device Parameters& p [[buffer(0)]], const device float* images[[buffer(1)]], device float* internals[[buffer(2)]], device float* result[[buffer(3)]], uint2 index [[thread_position_in_grid]])
{
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
        internals[correlation_offset + img1_width*y + x] = 0;
        result[img1_width*y + x] = NAN;
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
        internals[img1_avg_offset + img1_width*y + x] = avg;
        internals[img1_stdev_offset + img1_width*y + x] = stdev;
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
        internals[img2_avg_offset + img2_width*y + x] = avg;
        internals[img2_stdev_offset + img2_width*y + x] = stdev;
    }
}

kernel void cross_correlate(const device Parameters& p [[buffer(0)]], const device float* images[[buffer(1)]], device float* internals[[buffer(2)]], device float* result[[buffer(3)]], uint2 index [[thread_position_in_grid]])
{
    const int x1 = index.x;
    const int y1 = index.y;
    const int img1_width = p.img1_width;
    const int img1_height = p.img1_height;
    const int img2_width = p.img2_width;
    const int img2_height = p.img2_height;
    const float dir_x = p.dir_x;
    const float dir_y = p.dir_y;
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
    
    float avg1 = internals[img1_avg_offset + img1_width*y1 + x1];
    float stdev1 = internals[img1_stdev_offset + img1_width*y1 + x1];

    float best_corr = 0;
    float best_distance = NAN;
    const bool corridor_vertical = abs(dir_y)>abs(dir_x);
    const float corridor_coeff = corridor_vertical? dir_x/dir_y : dir_y/dir_x;
    for (int corridor_pos=corridor_start;corridor_pos<corridor_end;corridor_pos++)
    {
        int x2, y2;
        if (corridor_vertical)
        {
            y2 = corridor_pos;
            x2 = int(x1)+corridor_offset + int((y2-int(y1))*corridor_coeff);
            if (x2 < kernel_size || x2 >= img2_width-kernel_size)
                continue;
        }
        else
        {
            x2 = corridor_pos;
            y2 = int(y1)+corridor_offset + int((x2-int(x1))*corridor_coeff);
            if (y2 < kernel_size || y2 >= img2_height-kernel_size)
                continue;
        }
        float avg2 = internals[img2_avg_offset + img2_width*y2 + x2];
        float stdev2 = internals[img2_stdev_offset + img2_width*y2 + x2];

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

        if (corr > best_corr)
        {
            float dx = float(x2)-float(x1);
            float dy = float(y2)-float(y1);
            float distance = dx*dx+dy*dy;
            best_distance = distance;
            best_corr = corr;
        }
    }
    float current_corr = internals[correlation_offset + img1_width*y1 + x1];
    if (best_corr > current_corr && best_corr >= threshold)
    {
        internals[correlation_offset + img1_width*y1 + x1] = best_corr;
        result[img1_width*y1 + x1] = -sqrt(best_distance);
    }
}

