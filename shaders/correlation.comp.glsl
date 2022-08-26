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
layout(std430, binding = 1) buffer readonly Image1
{
    float img1[];
};
layout(std430, binding = 2) buffer readonly Image2
{
    float img2[];
};
layout(std430, binding = 3) buffer Image1_Avg
{
    float img1_avg[];
};
layout(std430, binding = 4) buffer Image1_Stdev
{
    float img1_stdev[];
};
layout(std430, binding = 5) buffer Image2_Avg
{
    float img2_avg[];
};
layout(std430, binding = 6) buffer Image2_Stdev
{
    float img2_stdev[];
};
layout(std430, binding = 7) buffer Best_Correlation
{
    float best_correlation[];
};
layout(std430, binding = 8) buffer Result
{
    float result[];
};

const float NAN = 0.0f/0.0f;

void prepare_initialdata() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const float kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);

    if (x < img1_width && y < img1_height)
    {
        best_correlation[img1_width*y + x] = 0;
    }
    
    if(x >= kernel_size && x < img1_width-kernel_size && y >= kernel_size && y < img1_height-kernel_size)
    {
        float avg = 0;
        float stdev = 0;

        for (int j=-kernel_size;j<=kernel_size;j++)
        {
            for (int i=-kernel_size;i<=kernel_size;i++)
            {
                float value = img1[(y+j)*img1_width+(x+i)];
                avg += value;
            }
        }
        avg /= kernel_point_count;

        for (int j=-kernel_size;j<=kernel_size;j++)
        {
            for (int i=-kernel_size;i<=kernel_size;i++)
            {
                float value = img1[(y+j)*img1_width+(x+i)];
                float delta = value-avg;
                stdev += delta*delta;
            }
        }
        stdev = sqrt(stdev/kernel_point_count);
        img1_avg[img1_width*y + x] = avg;
        img1_stdev[img1_width*y + x] = stdev;
    }

    if(x >= kernel_size && x < img2_width-kernel_size && y >= kernel_size && y < img2_height-kernel_size)
    {
        float avg = 0;
        float stdev = 0;

        for (int j=-kernel_size;j<=kernel_size;j++)
        {
            for (int i=-kernel_size;i<=kernel_size;i++)
            {
                float value = img2[(y+j)*img2_width+(x+i)];
                avg += value;
            }
        }
        avg /= kernel_point_count;

        for (int j=-kernel_size;j<=kernel_size;j++)
        {
            for (int i=-kernel_size;i<=kernel_size;i++)
            {
                float value = img2[(y+j)*img2_width+(x+i)];
                float delta = value-avg;
                stdev += delta*delta;
            }
        }
        stdev = sqrt(stdev/kernel_point_count);
        img2_avg[img2_width*y + x] = avg;
        img2_stdev[img2_width*y + x] = stdev;
    }
}

float correlate(in int x1, in int y1, in int x2, in int y2)
{
    const float kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);
    float avg1 = 0.0, avg2 = 0.0;
    for(int j=-kernel_size;j<=kernel_size;j++)
    {
        for (int i=-kernel_size;i<=kernel_size;i++)
        {
            avg1 += img1[(y1+j)*img1_width + (x1+i)];
            avg2 += img2[(y2+j)*img2_width + (x2+i)];
        }
    }
    avg1 /= kernel_point_count;
    avg2 /= kernel_point_count;
    float delta = 0.0;
    float stdev1 = 0.0, stdev2 = 0.0;
    for(int j=-kernel_size;j<=kernel_size;j++)
    {
        for (int i=-kernel_size;i<=kernel_size;i++)
        {
            float delta1 = img1[(y1+j)*img1_width + (x1+i)]-avg1;
            float delta2 = img2[(y2+j)*img2_width + (x2+i)]-avg2;
            delta += delta1*delta2;
            stdev1 += delta1*delta1;
            stdev2 += delta2*delta2;
        }
    }
    stdev1 = sqrt(stdev1/kernel_point_count);
    stdev2 = sqrt(stdev2/kernel_point_count);
    return delta/(stdev1*stdev2*kernel_point_count);
}

bool estimate_search_range(in uint x1, in uint y1, out float min_distance, out float max_distance)
{
    float min_depth, max_depth;
    bool found = false;
    float inv_scale = 1.0/scale;
    int x_min = int(floor((int(x1)-neighbor_distance)*inv_scale));
    int x_max = int(ceil((int(x1)+neighbor_distance)*inv_scale));
    int y_min = int(floor((int(y1)-neighbor_distance)*inv_scale));
    int y_max = int(ceil((int(y1)+neighbor_distance)*inv_scale));

    x_min = min(max(0, x_min), out_width-1);
    x_max = min(max(0, x_max), out_width-1);
    y_min = min(max(0, y_min), out_height-1);
    y_max = min(max(0, y_max), out_height-1);
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
    
    float avg1 = img1_avg[img1_width*y1 + x1];
    float stdev1 = img1_stdev[img1_width*y1 + x1];

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

        float avg2 = img2_avg[img2_width*y2 + x2];
        float stdev2 = img2_stdev[img2_width*y2 + x2];

        float corr = 0;
        for (int j=-kernel_size;j<=kernel_size;j++)
        {
            for (int i=-kernel_size;i<=kernel_size;i++)
            {
                float delta1 = img1[(y1+j)*img1_width+(x1+i)] - avg1;
                float delta2 = img2[(y2+j)*img2_width+(x2+i)] - avg2;
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
    float current_corr = best_correlation[img1_width*y1 + x1];
    // TODO: count + limit number of matches
    if (best_corr > current_corr && best_corr >= threshold)
    {
        const float inv_scale = 1.0/scale;
        const int out_pos = int(round(inv_scale*y1))*out_width + int(round(inv_scale*x1));
        best_correlation[img1_width*y1 + x1] = best_corr;
        result[out_pos] = sqrt(best_distance);
    }
}
