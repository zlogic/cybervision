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
    uint params_corridor_start;
    uint params_corridor_end;
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
layout(std430, set = 0, binding = 5) buffer  Result_Corr
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
    const uint x1 = gl_GlobalInvocationID.x;
    const uint y1 = gl_GlobalInvocationID.y;

    if (x1 < kernel_size || x1 >= img1_width-kernel_size || y1 < kernel_size ||  y1 >= img1_height-kernel_size) {
        return;
    }

    const bool first_iteration = iteration_pass == 0;
    const uint kernel_width = kernel_size*2+1;
    const float kernel_point_count = float(kernel_width*kernel_width);

    const uint img1_offset = 0;
    const uint img2_offset = img1_width*img1_height;

    const vec2 data_img1 = internals_img1[img1_width*y1+x1];
    const float avg1 = data_img1[0];
    const float stdev1 = data_img1[1];
    if (stdev1 < min_stdev) {
        return;
    }

    float best_corr = 0.0;
    ivec2 best_match = ivec2(-1, -1);

    uint corridor_start = kernel_size + params_corridor_start;
    uint corridor_end = kernel_size + params_corridor_end;
    if (!first_iteration) {
        const ivec3 data_int = internals_int[img1_width*y1 + x1];
        int min_pos_signed = data_int[0];
        int max_pos_signed = data_int[1];
        if (min_pos_signed<0 || max_pos_signed<0) {
            return;
        }
        const uint min_pos = uint(min_pos_signed);
        const uint max_pos = uint(max_pos_signed);
        corridor_start = clamp(min_pos+params_corridor_start, min_pos, max_pos);
        corridor_end = clamp(min_pos+params_corridor_end, min_pos, max_pos);
        if (corridor_start >= corridor_end) {
            return;
        }
    }

    vec2 coeff;
    vec2 add_const;
    calculate_epipolar_line(x1, y1, coeff, add_const);
    ivec2 corridor_offset_vector;
    if (abs(add_const.x) > abs(add_const.y)) {
        corridor_offset_vector = ivec2(corridor_offset, 0);
    } else {
        corridor_offset_vector = ivec2(0, corridor_offset);
    }
    for (uint corridor_pos=corridor_start;corridor_pos<corridor_end;corridor_pos++) {
        const int x2_signed = int(floor(coeff.x*float(corridor_pos)+add_const.x)) + corridor_offset_vector.x;
        const int y2_signed = int(floor(coeff.y*float(corridor_pos)+add_const.y)) + corridor_offset_vector.y;
        if (x2_signed < 0 || y2_signed < 0) {
            continue;
        }
        const uint x2 = uint(x2_signed);
        const uint y2 = uint(y2_signed);
        if (x2 < kernel_size || x2 >= img2_width-kernel_size || y2 < kernel_size ||  y2 >= img2_height-kernel_size) {
            continue;
        }

        const vec2 data_img2 = internals_img2[img2_width*y2 + x2];
        const float avg2 = data_img2[0];
        const float stdev2 = data_img2[1];
        if (stdev2 < min_stdev) {
            continue;
        }

        float corr = 0.0;
        for (uint j=0;j<kernel_width;j++) {
            for (uint i=0;i<kernel_width;i++) {
                float delta1 = images[img1_offset + (y1+j-kernel_size)*img1_width+(x1+i-kernel_size)] - avg1;
                float delta2 = images[img2_offset + (y2+j-kernel_size)*img2_width+(x2+i-kernel_size)] - avg2;
                corr += delta1*delta2;
            }
        }
        corr = corr/(stdev1*stdev2*kernel_point_count);

        if (corr >= threshold && corr > best_corr) {
            best_match.x = int(round(float(x2)/scale));
            best_match.y = int(round(float(y2)/scale));
            best_corr = corr;
        }
    }

    uint out_pos = out_width*uint((float(y1)/scale)) + uint(float(x1)/scale);
    float current_corr = result_corr[out_pos];
    if (best_corr >= threshold && best_corr > current_corr)
    {
        result_matches[out_pos] = best_match;
        result_corr[out_pos] = best_corr;
    }
}
