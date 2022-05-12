#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1 ) in;

layout(std430, binding = 0) buffer readonly Parameters
{
    int img1_width;
    int img1_height;
    int img2_width;
    int img2_height;
    float dir_x, dir_y;
    int corridor_offset;
    int initial_run;
    int kernel_size;
    float threshold;
};
layout(std430, binding = 1) buffer readonly Images
{
    float images[];
};
layout(std430, binding = 2) buffer Internals
{
    // Layout:
    // img1 avg; img1 stdev; img2 avg; img2 stdev; best correlation
    float internals[];
};
layout(std430, binding = 3) buffer writeonly Result
{
    float result[];
};

const float NaN = 0.0f/0.0f;

void prepare_initialdata() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const float kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);

    uint img1_offset = 0;
    uint img2_offset = img1_width*img1_height;

    const int img1_avg_offset = 0;
    const int img1_stdev_offset = img1_avg_offset + img1_width*img1_height;
    const int img2_avg_offset = img1_stdev_offset + img1_width*img1_height;
    const int img2_stdev_offset = img2_avg_offset + img2_width*img2_height;
    const int correlation_offset = img2_stdev_offset + img2_width*img2_height;

    if (x < img1_width && y < img1_height)
    {
        internals[correlation_offset + img1_width*y + x] = 0;
        result[img1_width*y + x] = NaN;
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

void main() {
    if (initial_run == 1)
    {
        prepare_initialdata();
        return;
    }

    const uint x1 = gl_GlobalInvocationID.x;
    const uint y1 = gl_GlobalInvocationID.y;
    const float kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);

    if(x1 < kernel_size || x1 >= img1_width-kernel_size || y1 < kernel_size ||  y1 >= img1_height-kernel_size)
        return;

    uint img1_offset = 0;
    uint img2_offset = img1_width*img1_height;


    const int img1_avg_offset = 0;
    const int img1_stdev_offset = img1_avg_offset + img1_width*img1_height;
    const int img2_avg_offset = img1_stdev_offset + img1_width*img1_height;
    const int img2_stdev_offset = img2_avg_offset + img2_width*img2_height;
    const int correlation_offset = img2_stdev_offset + img2_width*img2_height;
    
    float avg1 = internals[img1_avg_offset + img1_width*y1 + x1];
    float stdev1 = internals[img1_stdev_offset + img1_width*y1 + x1];

    const int y2_min = kernel_size;
    const int y2_max = img2_height-kernel_size;

    float best_corr = 0;
    float best_distance = NaN;
    for (int y2=y2_min;y2<y2_max;y2++)
    {
        const int x2 = int(x1+corridor_offset) + int(float(y2)*dir_x/dir_y);
        if (x2 < kernel_size || x2 >= img2_width-kernel_size)
            continue;
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
            float distance = -sqrt(dx*dx+dy*dy);
            best_distance = distance;
            best_corr = corr;
        }
    }
    float current_corr = internals[correlation_offset + img1_width*y1 + x1];
    if (best_corr > current_corr && best_corr >= threshold)
    {
        internals[correlation_offset + img1_width*y1 + x1] = best_corr;
        result[img1_width*y1 + x1] = best_distance;
    }
}
