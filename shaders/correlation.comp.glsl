#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std430, binding = 0) buffer readonly Parameters
{
    int img1_width;
    int img1_height;
    int img2_width;
    int img2_height;
    int out_width;
    int out_height;
    float dir_x, dir_y;
    float scale;
    int iteration;
    int corridor_offset;
    int corridor_start;
    int corridor_end;
    int initial_run;
    int kernel_size;
    float threshold;
    int neighbor_distance;
    float max_slope;
    int match_limit;
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
layout(std430, binding = 3) buffer Result
{
    float result[];
};

const float NAN = 0.0f/0.0f;

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

bool estimate_search_range(in uint x1, in uint y1, out float min_distance, out float max_distance)
{
    float min_depth, max_depth;
    bool found = false;
    float inv_scale = 1.0/scale;
    int x_min = int(floor((float(x1)-neighbor_distance)*inv_scale));
    int x_max = int(ceil((float(x1)+neighbor_distance)*inv_scale));
    int y_min = int(floor((float(y1)-neighbor_distance)*inv_scale));
    int y_max = int(ceil((float(y1)+neighbor_distance)*inv_scale));

    x_min = max(0, min(x_min, out_width-1));
    x_max = max(0, min(x_max, out_width-1));
    y_min = max(0, min(y_min, out_height-1));
    y_max = max(0, min(y_max, out_height-1));
    for (int j=y_min;j<y_max;j++)
    {
        for (int i=x_min;i<x_max;i++)
        {
            int out_pos = j*out_width + i;
            float current_depth = result[out_pos];

            if (isnan(current_depth))
                continue;

            float dx = float(i)-float(x1)*inv_scale;
            float dy = float(j)-float(y1)*inv_scale;
            float point_distance = sqrt(dx*dx + dy*dy);
            float min_value = current_depth - point_distance*max_slope;
            float max_value = current_depth + point_distance*max_slope;

            if (!found)
            {
                min_depth = min_value;
                max_depth = max_value;
                found = true;
            }
            else
            {
                min_depth = min(min_value, min_depth);
                max_depth = max(max_value, max_depth);
            }
        }
    }
    if(!found)
        return false;
    min_distance = min_depth;
    max_distance = max_depth;
    return true;
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

    float min_distance, max_distance;
    if (iteration > 0 && !estimate_search_range(x1, y1, min_distance, max_distance))
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

    float best_corr = 0;
    float best_distance = NAN;
    const bool corridor_vertical = abs(dir_y)>abs(dir_x);
    const float corridor_coeff = corridor_vertical? dir_x/dir_y : dir_y/dir_x;
    const float dir_length = sqrt(dir_x*dir_x + dir_y*dir_y);
    const float distance_coeff = abs(corridor_vertical? dir_y/dir_length : dir_x/dir_length)*scale;

    const int min_pos = int(floor(min_distance*distance_coeff));
    const int max_pos = int(ceil(max_distance*distance_coeff));
    const int current_pos = corridor_vertical? int(y1) : int(x1);
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
        if (iteration > 0 && 
            (corridor_pos < current_pos+min_pos || corridor_pos > current_pos+max_pos ||
             corridor_pos < current_pos-max_pos || corridor_pos > current_pos-min_pos))
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
            float dx = (float(x2)-float(x1))/scale;
            float dy = (float(y2)-float(y1))/scale;
            float distance = dx*dx+dy*dy;
            best_distance = distance;
            best_corr = corr;
        }
    }
    float current_corr = internals[correlation_offset + img1_width*y1 + x1];
    // TODO: count + limit number of matches
    if (best_corr > current_corr && best_corr >= threshold)
    {
        const float inv_scale = 1.0/scale;
        const int out_pos = int(round(inv_scale*y1))*out_width + int(round(inv_scale*x1));
        internals[correlation_offset + img1_width*y1 + x1] = best_corr;
        result[out_pos] = sqrt(best_distance);
    }
}
