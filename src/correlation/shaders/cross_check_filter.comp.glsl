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
layout(std430, set = 1, binding = 0) buffer Img1
{
    ivec2 img1[];
};
layout(std430, set = 1, binding = 1) buffer readonly Img2
{
    ivec2 img2[];
};
layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;

    // TODO: remove this debug code
    img1[0] = ivec2(img1_width, img1_height);
}
