#version 450
#pragma shader_stage(compute)

layout(std430, push_constant) uniform readonly Parameters
{
    uint img1_width;
    uint img1_height;
    uint img2_width;
    uint img2_height;
    uint out_width;
    uint out_height;
    float scale;
    uint iteration_pass;
    mat3x3 fundamental_matrix;
    int corridor_offset;
    uint corridor_start;
    uint corridor_end;
    uint kernel_size;
    float threshold;
    float min_stdev;
    uint neighbor_distance;
    float extend_range;
    float min_range;
};
layout(std430, set = 0, binding = 0) buffer readonly Images
{
    // Layout:
    // img1; img2
    float images[];
};
layout(std430, set = 0, binding = 1) buffer Internals_Img1
{
    // Layout:
    // For searchdata: contains [min_corridor, stdev] for image1
    // For cross_correlate: contains [avg, stdev] for image1
    vec2 internals_img1[];
};
layout(std430, set = 0, binding = 2) buffer Internals_Img2
{
    // Layout:
    // Contains [avg, stdev] for image 2
    vec2 internals_img2[];
};
layout(std430, set = 0, binding = 3) buffer Internals_Int
{
    // Layout:
    // Contains [min, max, neighbor_count] for the corridor range
    ivec3 internals_int[];
};
layout(std430, set = 0, binding = 4) buffer Result_Matches
{
    ivec2 result_matches[];
};
layout(std430, set = 0, binding = 5) buffer Result_Corr
{
    float result_corr[];
};
layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;

    const uint kernel_width = kernel_size*2+1;
    const float kernel_point_count = float(kernel_width*kernel_width);

    const uint img1_offset = 0;
    const uint img2_offset = img1_width*img1_height;

    if (x >= kernel_size && x < img1_width-kernel_size && y >= kernel_size && y < img1_height-kernel_size) {
        float avg = 0.0;
        float stdev = 0.0;

        for (uint j=0;j<kernel_width;j++) {
            for (uint i=0;i<kernel_width;i++) {
                float value = images[img1_offset + (y+j-kernel_size)*img1_width+(x+i-kernel_size)];
                avg += value;
            }
        }
        avg /= kernel_point_count;

        for (uint j=0;j<kernel_width;j++) {
            for (uint i=0;i<kernel_width;i++) {
                float value = images[img1_offset + (y+j-kernel_size)*img1_width+(x+i-kernel_size)];
                float delta = value-avg;
                stdev += delta*delta;
            }
        }
        stdev = sqrt(stdev/kernel_point_count);
        internals_img1[img1_width*y+x] = vec2(avg, stdev);
    } else if (x < img1_width && y < img1_height) {
        internals_img1[img1_width*y+x] = vec2(0.0, -1.0);
    }

    if (x >= kernel_size && x < img2_width-kernel_size && y >= kernel_size && y < img2_height-kernel_size) {
        float avg = 0.0;
        float stdev = 0.0;

        for (uint j=0;j<kernel_width;j++) {
            for (uint i=0;i<kernel_width;i++) {
                float value = images[img2_offset + (y+j-kernel_size)*img2_width+(x+i-kernel_size)];
                avg += value;
            }
        }
        avg /= kernel_point_count;

        for (uint j=0;j<kernel_width;j++) {
            for (uint i=0;i<kernel_width;i++) {
                float value = images[img2_offset + (y+j-kernel_size)*img2_width+(x+i-kernel_size)];
                float delta = value-avg;
                stdev += delta*delta;
            }
        }
        stdev = sqrt(stdev/kernel_point_count);
        internals_img2[img2_width*y+x] = vec2(avg, stdev);
    } else if (x < img2_width && y < img2_height) {
        internals_img2[img2_width*y+x] = vec2(0.0, -1.0);
    }
}
