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

void calculate_epipolar_line(uint x1, uint y1, out vec2 coeff, out vec2 add_const) {
    vec3 p1 = vec3(float(x1)/scale, float(y1)/scale, 1.0);
    vec3 f_p1 = fundamental_matrix*p1;
    if (abs(f_p1.x)>abs(f_p1.y)) {
        coeff = vec2(-f_p1.y/f_p1.x, 1.0);
        add_const = vec2(-scale*f_p1.z/f_p1.x, 0.0);
    } else {
        coeff = vec2(1.0, -f_p1.x/f_p1.y);
        add_const = vec2(0.0, -scale*f_p1.z/f_p1.y);
    }
}

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;

    if (x >= img1_width || y >= img1_height) {
        return;
    }

    const int x_signed = int(x);
    const int y_signed = int(y);

    const bool first_pass = iteration_pass == 0;
    const int neighbor_distance_signed = int(neighbor_distance);

    const int corridor_start_signed = int(corridor_start);
    const int corridor_end_signed = int(corridor_end);
    const int out_neighbor_width = int(ceil(float(neighbor_distance)/scale))*2+1;

    const int start_pos_x = int(floor(float(x_signed-neighbor_distance_signed)/scale));
    const int start_pos_y = int(floor(float(y_signed-neighbor_distance_signed)/scale));

    ivec3 data_int = internals_int[y*img1_width+x];
    int neighbor_count = data_int[2];
    if (!first_pass && neighbor_count<=0) {
        return;
    }

    vec2 data = internals_img1[y*img1_width+x];
    float mid_corridor = data[0];
    float range_stdev = data[1];
    if (!first_pass) {
        mid_corridor /= float(neighbor_count);
    }

    vec2 coeff;
    vec2 add_const;
    calculate_epipolar_line(x, y, coeff, add_const);
    const bool corridor_vertical = abs(coeff.y) > abs(coeff.x);
    for (int i=corridor_start_signed;i<corridor_end_signed;i++) {
        int x_out = start_pos_x + (i%out_neighbor_width);
        int y_out = start_pos_y + (i/out_neighbor_width);

        if (x_out <=0 || y_out<=0 || x_out>=int(out_width) || y_out>=int(out_height)) {
            continue;
        }

        vec2 coord2 = vec2(result_matches[uint(y_out)*out_width + uint(x_out)]) * scale;
        if (coord2.x<0.0 || coord2.y<0.0) {
            continue;
        }

        float corridor_pos;
        if (corridor_vertical) {
            corridor_pos = round((coord2.y-add_const.y)/coeff.y);
        } else {
            corridor_pos = round((coord2.x-add_const.x)/coeff.x);
        }
        if (first_pass) {
            mid_corridor += corridor_pos;
            neighbor_count++;
        } else {
            float delta = corridor_pos-mid_corridor;
            range_stdev += delta*delta;
        }
    }

    if (first_pass) {
        data[0] = mid_corridor;
        internals_img1[y*img1_width+x] = data;
        data_int[2] = neighbor_count;
        internals_int[y*img1_width+x] = data_int;
        return;
    }

    data[1] = range_stdev;
    internals_img1[y*img1_width+x] = data;
    range_stdev = sqrt(range_stdev/float(neighbor_count));
    const int corridor_center = int(round(mid_corridor));
    const int corridor_length = int(round(min_range+range_stdev*extend_range));
    int min_pos = int(kernel_size);
    int max_pos;
    if (corridor_vertical) {
        max_pos = int(img2_height-kernel_size);
    } else {
        max_pos = int(img2_width-kernel_size);
    }
    min_pos = clamp(corridor_center - corridor_length, min_pos, max_pos);
    max_pos = clamp(corridor_center + corridor_length, min_pos, max_pos);
    internals_int[y*img1_width+x] = ivec3(min_pos, max_pos, neighbor_count);
}
