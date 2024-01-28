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
    float internals_img1[];
};
layout(std430, set = 0, binding = 2) buffer Internals_Img2
{
    // Layout:
    // Contains [avg, stdev] for image 2
    float internals_img2[];
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
layout(std430, set = 0, binding = 5) buffer  Result_Corr
{
    float result_corr[];
};
layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    /*
    if (x < out_width && y < out_height) {
        result_matches[out_width*y+x] = ivec2(-1, -1);
        result_corr[out_width*y+x] = -1.0;
    }
    */

    // TODO: remove this debug code
    result_corr[0] = threshold;
    result_corr[1] = (fundamental_matrix * vec3(10.0, 5.0, 2.0)).x;
}
