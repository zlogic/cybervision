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
    int phase;
    int kernel_size;
    float threshold;
    int neighbor_distance;
    float max_slope;
    int match_limit;
};
layout(std430, binding = 1) buffer readonly Images
{
    // Layout:
    // img1; img2; previous result
    float images[];
};
layout(std430, binding = 2) buffer Internals
{
    // Layout:
    // img1 avg; img1 stdev; img2 avg; img2 stdev; best correlation; min distance; max distance
    float internals[];
};
layout(std430, binding = 3) buffer Internals_Int
{
    // Layout:
    // match count
    int internals_int[];
};
layout(std430, binding = 4) buffer writeonly Result
{
    float result[];
};

const float NAN = 0.0f/0.0f;

void prepare_initialdata() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const float kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);

    const uint img1_offset = 0;
    const uint img2_offset = img1_width*img1_height;

    const int img1_avg_offset = 0;
    const int img1_stdev_offset = img1_avg_offset + img1_width*img1_height;
    const int img2_avg_offset = img1_stdev_offset + img1_width*img1_height;
    const int img2_stdev_offset = img2_avg_offset + img2_width*img2_height;
    const int correlation_offset = img2_stdev_offset + img2_width*img2_height;
    const int min_distance_offset = correlation_offset + img1_width*img1_height;
    const int max_distance_offset = min_distance_offset + img1_width*img1_height;
    const int match_count_offset = 0;

    if (x < img1_width && y < img1_height)
    {
        internals[correlation_offset + img1_width*y+x] = 0;
        internals[min_distance_offset + img1_width*y+x] = NAN;
        internals[max_distance_offset + img1_width*y+x] = NAN;
        internals_int[match_count_offset + img1_width*y+x] = 0;
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

void prepare_searchdata() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const uint previous_result_offset = img1_width*img1_height + img2_width*img2_height;
    const int min_distance_offset = (img1_width*img1_height)*3 + (img2_width*img2_height)*2;
    const int max_distance_offset = min_distance_offset + img1_width*img1_height;

    float inv_scale = 1.0/scale;
    int x_min = int(floor((int(x)-neighbor_distance)*inv_scale));
    int x_max = int(ceil((int(x)+neighbor_distance)*inv_scale));
    int y_min = int(float(y)*inv_scale)+corridor_start;
    int y_max = int(float(y)*inv_scale)+corridor_end;

    x_min = min(max(0, x_min), out_width);
    x_max = min(max(0, x_max), out_width);
    y_min = min(max(0, y_min), out_height);
    y_max = min(max(0, y_max), out_height);

    float min_distance = internals[min_distance_offset + y*img1_width+x];
    float max_distance = internals[max_distance_offset + y*img1_width+x];
    for (int j=y_min;j<y_max;j++)
    {
        for (int i=x_min;i<x_max;i++)
        {
            int out_pos = j*out_width + i;
            float current_depth = images[previous_result_offset+out_pos];

            if (isnan(current_depth))
                continue;

            float dx = float(i)-float(x)*inv_scale;
            float dy = float(j)-float(y)*inv_scale;
            float point_distance = sqrt(dx*dx + dy*dy);
            float min_value = current_depth - point_distance*max_slope;
            float max_value = current_depth + point_distance*max_slope;

            if (isnan(min_distance) || isnan(max_distance))
            {
                min_distance = min_value;
                max_distance = min_value;
            }
            else
            {
                min_distance = min(min_value, min_distance);
                max_distance = max(max_value, max_distance);
            }
        }
    }
    internals[min_distance_offset + y*img1_width+x] = min_distance;
    internals[max_distance_offset + y*img1_width+x] = max_distance;
}

void main() {
    if (phase == 1)
    {
        prepare_initialdata();
        return;
    }
    if (phase == 2)
    {
        prepare_searchdata();
        return;
    }

    const uint x1 = gl_GlobalInvocationID.x;
    const uint y1 = gl_GlobalInvocationID.y;
    const float kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);

    if(x1 < kernel_size || x1 >= img1_width-kernel_size || y1 < kernel_size ||  y1 >= img1_height-kernel_size)
        return;

    const uint img1_offset = 0;
    const uint img2_offset = img1_width*img1_height;

    const uint img1_avg_offset = 0;
    const uint img1_stdev_offset = img1_avg_offset + img1_width*img1_height;
    const uint img2_avg_offset = img1_stdev_offset + img1_width*img1_height;
    const uint img2_stdev_offset = img2_avg_offset + img2_width*img2_height;
    const uint correlation_offset = img2_stdev_offset + img2_width*img2_height;
    const uint min_distance_offset = correlation_offset + img1_width*img1_height;
    const uint max_distance_offset = min_distance_offset + img1_width*img1_height;
    const uint match_count_offset = 0;

    const float min_distance = internals[min_distance_offset + y1*img1_width+x1];
    const float max_distance = internals[max_distance_offset + y1*img1_width+x1];
    if (iteration > 0 && (isnan(min_distance) || isnan(max_distance)))
        return;

    const uint out_pos = img1_width*y1 + x1;
    int match_count = internals_int[match_count_offset+out_pos];

    if (match_count > match_limit)
        return;
    
    float avg1 = internals[img1_avg_offset + img1_width*y1+x1];
    float stdev1 = internals[img1_stdev_offset + img1_width*y1+x1];

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
            !((corridor_pos >= current_pos+min_pos && corridor_pos <= current_pos+max_pos) ||
              (corridor_pos >= current_pos-max_pos && corridor_pos <= current_pos-min_pos)))
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
            float point_distance = dx*dx+dy*dy;
            best_distance = point_distance;
            best_corr = corr;
        }
    }

    float current_corr = internals[correlation_offset + img1_width*y1 + x1];
    if (best_corr >= threshold)
    {
        match_count++;
        internals[match_count_offset+out_pos] = match_count;
        if (match_count > match_limit)
        {
            result[out_pos] = NAN;
        }
        else if (best_corr > current_corr)
        {
            const float inv_scale = 1.0/scale;
            internals[correlation_offset + img1_width*y1 + x1] = best_corr;
            result[out_pos] = sqrt(best_distance);
        }
    }
}
