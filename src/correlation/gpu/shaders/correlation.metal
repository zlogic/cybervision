#include <metal_stdlib>

using namespace metal;

struct Parameters
{
    uint img1_width;
    uint img1_height;
    uint img2_width;
    uint img2_height;
    uint out_width;
    uint out_height;
    float scale;
    uint iteration_pass;
    float3x3 fundamental_matrix;
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

kernel void init_out_data(const device Parameters& p [[buffer(0)]], device int2* result_matches[[buffer(5)]], device float* result_corr[[buffer(6)]], uint2 index [[thread_position_in_grid]]) {
    const uint x = index.x;
    const uint y = index.y;

    const uint out_width = p.out_width;
    const uint out_height = p.out_height;

    if (x < out_width && y < out_height) {
        result_matches[out_width*y+x] = int2(-1, -1);
        result_corr[out_width*y+x] = -1.0;
    }
}

kernel void prepare_initialdata_searchdata(const device Parameters& p [[buffer(0)]], device float2* internals_img1[[buffer(2)]], device int3* internals_int[[buffer(4)]], device float* result_corr[[buffer(6)]], uint2 index [[thread_position_in_grid]]) {
    const uint x = index.x;
    const uint y = index.y;

    const uint out_width = p.out_width;
    const uint out_height = p.out_height;
    const uint img1_width = p.img1_width;
    const uint img1_height = p.img1_height;

    if (x < img1_width && y < img1_height) {
        internals_img1[img1_width*y+x] = float2(0.0, 0.0);
        internals_int[img1_width*y+x] = int3(-1, -1, 0);
    }
    if (x < out_width && y < out_height) {
        result_corr[out_width*y+x] = -1.0;
    }
}

kernel void prepare_initialdata_correlation(const device Parameters& p [[buffer(0)]], const device float* images[[buffer(1)]], device float2* internals_img1[[buffer(2)]], device float2* internals_img2[[buffer(3)]], uint2 index [[thread_position_in_grid]]) {
    const uint x = index.x;
    const uint y = index.y;

    const uint img1_width = p.img1_width;
    const uint img1_height = p.img1_height;
    const uint img2_width = p.img2_width;
    const uint img2_height = p.img2_height;
    const uint kernel_size = p.kernel_size;
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
        internals_img1[img1_width*y+x] = float2(avg, stdev);
    } else if (x < img1_width && y < img1_height) {
        internals_img1[img1_width*y+x] = float2(0.0, -1.0);
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
        internals_img2[img2_width*y+x] = float2(avg, stdev);
    } else if (x < img2_width && y < img2_height) {
        internals_img2[img2_width*y+x] = float2(0.0, -1.0);
    }
}

void calculate_epipolar_line(const device Parameters& p, const uint x1, const uint y1, thread float2& coeff, thread float2& add_const) {
    const float scale = p.scale;
    const float3 p1 = float3(float(x1)/scale, float(y1)/scale, 1.0);
    const float3 f_p1 = p.fundamental_matrix*p1;
    if (abs(f_p1.x)>abs(f_p1.y)) {
        coeff = float2(-f_p1.y/f_p1.x, 1.0);
        add_const = float2(-scale*f_p1.z/f_p1.x, 0.0);
    } else {
        coeff = float2(1.0, -f_p1.x/f_p1.y);
        add_const = float2(0.0, -scale*f_p1.z/f_p1.y);
    }
}

kernel void prepare_searchdata(const device Parameters& p [[buffer(0)]], device float2* internals_img1[[buffer(2)]], device int3* internals_int[[buffer(4)]], const device int2* result_matches[[buffer(5)]], uint2 index [[thread_position_in_grid]]) {
    const uint x = index.x;
    const uint y = index.y;

    const uint img1_width = p.img1_width;
    const uint img1_height = p.img1_height;

    if (x >= img1_width || y >= img1_height) {
        return;
    }

    const uint img2_width = p.img2_width;
    const uint img2_height = p.img2_height;
    const uint out_width = p.out_width;
    const uint out_height = p.out_height;
    const float scale = p.scale;
    const uint kernel_size = p.kernel_size;

    const int x_signed = int(x);
    const int y_signed = int(y);

    const bool first_pass = p.iteration_pass == 0;
    const int neighbor_distance_signed = int(p.neighbor_distance);
    const float extend_range = p.extend_range;
    const float min_range = p.min_range;

    const int corridor_start_signed = int(p.corridor_start);
    const int corridor_end_signed = int(p.corridor_end);
    const int out_neighbor_width = int(ceil(float(p.neighbor_distance)/scale))*2+1;

    const int start_pos_x = int(floor(float(x_signed-neighbor_distance_signed)/scale));
    const int start_pos_y = int(floor(float(y_signed-neighbor_distance_signed)/scale));

    int3 data_int = internals_int[y*img1_width+x];
    int neighbor_count = data_int[2];
    if (!first_pass && neighbor_count<=0) {
        return;
    }

    float2 data = internals_img1[y*img1_width+x];
    float mid_corridor = data[0];
    float range_stdev = data[1];
    if (!first_pass) {
        mid_corridor /= float(neighbor_count);
    }

    float2 coeff;
    float2 add_const;
    calculate_epipolar_line(p, x, y, coeff, add_const);
    const bool corridor_vertical = abs(coeff.y) > abs(coeff.x);
    for (int i=corridor_start_signed;i<corridor_end_signed;i++) {
        int x_out = start_pos_x + (i%out_neighbor_width);
        int y_out = start_pos_y + (i/out_neighbor_width);

        if (x_out <=0 || y_out<=0 || x_out>=int(out_width) || y_out>=int(out_height)) {
            continue;
        }

        float2 coord2 = float2(result_matches[uint(y_out)*out_width + uint(x_out)]) * scale;
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
    internals_int[y*img1_width+x] = int3(min_pos, max_pos, neighbor_count);
}

kernel void cross_correlate(const device Parameters& p [[buffer(0)]], const device float* images[[buffer(1)]], const device float2* internals_img1[[buffer(2)]], const device float2* internals_img2[[buffer(3)]], const device int3* internals_int[[buffer(4)]], device int2* result_matches[[buffer(5)]], device float* result_corr[[buffer(6)]], uint2 index [[thread_position_in_grid]]) {
    const uint x1 = index.x;
    const uint y1 = index.y;

    const uint img1_width = p.img1_width;
    const uint img1_height = p.img1_height;
    const uint kernel_size = p.kernel_size;

    if (x1 < kernel_size || x1 >= img1_width-kernel_size || y1 < kernel_size ||  y1 >= img1_height-kernel_size) {
        return;
    }

    const uint img2_width = p.img2_width;
    const uint img2_height = p.img2_height;
    const uint out_width = p.out_width;
    const float scale = p.scale;
    const bool first_iteration = p.iteration_pass == 0;
    const int corridor_offset = p.corridor_offset;
    const float threshold = p.threshold;
    const float min_stdev = p.min_stdev;
    const uint kernel_width = kernel_size*2+1;
    const float kernel_point_count = float(kernel_width*kernel_width);

    const uint img1_offset = 0;
    const uint img2_offset = img1_width*img1_height;

    const float2 data_img1 = internals_img1[img1_width*y1+x1];
    const float avg1 = data_img1[0];
    const float stdev1 = data_img1[1];
    if (stdev1 < min_stdev) {
        return;
    }

    float best_corr = 0.0;
    int2 best_match = int2(-1, -1);

    uint corridor_start = kernel_size + p.corridor_start;
    uint corridor_end = kernel_size + p.corridor_end;
    if (!first_iteration) {
        const int3 data_int = internals_int[img1_width*y1 + x1];
        int min_pos_signed = data_int[0];
        int max_pos_signed = data_int[1];
        if (min_pos_signed<0 || max_pos_signed<0) {
            return;
        }
        const uint min_pos = uint(min_pos_signed);
        const uint max_pos = uint(max_pos_signed);
        corridor_start = clamp(min_pos+p.corridor_start, min_pos, max_pos);
        corridor_end = clamp(min_pos+p.corridor_end, min_pos, max_pos);
        if (corridor_start >= corridor_end) {
            return;
        }
    }

    float2 coeff;
    float2 add_const;
    calculate_epipolar_line(p, x1, y1, coeff, add_const);
    int2 corridor_offset_vector;
    if (abs(add_const.x) > abs(add_const.y)) {
        corridor_offset_vector = int2(corridor_offset, 0);
    } else {
        corridor_offset_vector = int2(0, corridor_offset);
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

        const float2 data_img2 = internals_img2[img2_width*y2 + x2];
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

kernel void cross_check_filter(const device Parameters& p [[buffer(0)]], device int2* img1[[buffer(1)]], const device int2* img2[[buffer(2)]], uint2 index [[thread_position_in_grid]]) {
    const uint x1 = index.x;
    const uint y1 = index.y;

    const uint img1_width = p.img1_width;
    const uint img1_height = p.img1_height;

    if (x1 >= img1_width || y1 >= img1_height) {
        return;
    }

    const uint img2_width = p.img2_width;
    const uint img2_height = p.img2_height;
    const int search_area = int(p.neighbor_distance);

    const int2 point = img1[img1_width*y1+x1];
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
            const int2 rpoint = img2[img2_width*y2+x2];
            if (rpoint.x < 0 || rpoint.y < 0) {
                continue;
            }
            if (rpoint.x >= r_min_x && rpoint.x < r_max_x && rpoint.y >= r_min_y && rpoint.y < r_max_y) {
                return;
            }
        }
    }

    img1[img1_width*y1+x1] = int2(-1, -1);
}
