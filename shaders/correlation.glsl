#version 450
#pragma shader_stage(compute)

#define MAX_KERNEL_POINTS (2*9+1)*(2*9+1)

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1 ) in;

layout(std430, binding = 0) buffer readonly Parameters
{
    int img1_width;
    int img1_height;
    int img2_width;
    int img2_height;
    float dir_x, dir_y;
    int corridor_size;
    int kernel_size;
    float threshold;
};
layout(std430, binding = 1) buffer readonly Image1
{
    float img1[];
};
layout(std430, binding = 2) buffer readonly Image2
{
    float img2[];
};
layout(std430, binding = 3) buffer writeonly Result
{
    float result[];
};

const float NaN = 0.0f/0.0f;

void main() {
    const uint x1 = gl_GlobalInvocationID.x;
    const uint y1 = gl_GlobalInvocationID.y;
    const uint kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);

    if(x1 < kernel_size || x1 >= img1_width-kernel_size || y1 < kernel_size ||  y1 >= img1_height-kernel_size)
    {
        result[y1*img1_width+x1] = NaN;
        return;
    }

    float avg1 = 0;
    float stdev1 = 0;
    float delta1[MAX_KERNEL_POINTS];

    for (int j=-kernel_size;j<=kernel_size;j++)
    {
        for (int i=-kernel_size;i<=kernel_size;i++)
        {
            float value = img1[(y1+j)*img1_width+(x1+i)];
            avg1 += value;
            delta1[(j+kernel_size)*img1_width+(i+kernel_size)] = value;
        }
    }

    for (uint i=0;i<kernel_point_count;i++)
    {
        delta1[i] -= avg1;
        stdev1 += delta1[i] * delta1[i];
    }
    stdev1 = sqrt(stdev1/float(kernel_point_count));

    //const int x2_min = -corridor_size;
    //const int x2_max = corridor_size+1;
    const int x2_min = 0;
    const int x2_max = 1;
    const int y2_min = kernel_size;
    const int y2_max = img2_height-kernel_size;
    
    float best_corr = 0;
    float best_distance = NaN;
    float delta2[MAX_KERNEL_POINTS];
    for (int c=x2_min;c<x2_max;c++)
    {
        for (int s=y2_min;s<y2_max;s++)
        {
            int y2 = s;
            int x2 = int(x1+c)+ int(float(y2)*dir_x/dir_y);
            float avg2 = 0;
            float stdev2 = 0;
            for (int j=-kernel_size;j<=kernel_size;j++)
            {
                for (int i=-kernel_size;i<=kernel_size;i++)
                {
                    float value = img2[(y2+j)*img2_width+(x2+i)];
                    avg2 += value;
                    delta2[(j+kernel_size)*img2_width+(i+kernel_size)] = value;
                }
            }
            for (int i=0;i<kernel_point_count;i++)
            {
                delta2[i] -= avg2;
                stdev2 += delta2[i] * delta2[i];
            }
            stdev2 = sqrt(stdev2/float(kernel_point_count));

            float corr = 0;
            for (int i=0;i<kernel_point_count;i++)
                corr += delta1[i] * delta2[i];
            corr = corr/(stdev1*stdev2*kernel_point_count);

            if (corr >= threshold && corr > best_corr)
            {
                float dx = float(x2)-float(x1);
                float dy = float(y2)-float(y1);
                float distance = -sqrt(dx*dx+dy*dy);
                best_distance = distance;
                best_corr = corr;
            }
        }
    }
    
    result[img1_width*y1 + x1] = best_distance;
}
