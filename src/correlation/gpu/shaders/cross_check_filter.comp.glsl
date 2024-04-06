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
layout(std430, set = 0, binding = 0) buffer Img1
{
    ivec2 img1[];
};
layout(std430, set = 0, binding = 1) buffer readonly Img2
{
    ivec2 img2[];
};
layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main() {
    const uint x1 = gl_GlobalInvocationID.x;
    const uint y1 = gl_GlobalInvocationID.y;

    if (x1 >= img1_width || y1 >= img1_height) {
        return;
    }

    const int search_area = int(neighbor_distance);

    const ivec2 point = img1[img1_width*y1+x1];
    if (point.x < 0 || point.y < 0 || uint(point.x) >= img2_width || uint(point.y) >= img2_height) {
        return;
    }

    const uint min_x = uint(clamp(point.x-search_area, 0, int(img2_width)));
    const uint max_x = uint(clamp(point.x+search_area+1, 0, int(img2_width)));
    const uint min_y = uint(clamp(point.y-search_area, 0, int(img2_height)));
    const uint max_y = uint(clamp(point.y+search_area+1, 0, int(img2_height)));

    const int r_min_x = clamp(int(x1)-search_area, 0, int(img1_width));
    const int r_max_x = clamp(int(x1)+search_area+1, 0, int(img1_width));
    const int r_min_y = clamp(int(y1)-search_area, 0, int(img1_height));
    const int r_max_y = clamp(int(y1)+search_area+1, 0, int(img1_height));

    for (uint y2=min_y;y2<max_y;y2++) {
        for (uint x2=min_x;x2<max_x;x2++) {
            const ivec2 rpoint = img2[img2_width*y2+x2];
            if (rpoint.x < 0 || rpoint.y < 0) {
                continue;
            }
            if (rpoint.x >= r_min_x && rpoint.x < r_max_x && rpoint.y >= r_min_y && rpoint.y < r_max_y) {
                return;
            }
        }
    }

    img1[img1_width*y1+x1] = ivec2(-1, -1);
}
