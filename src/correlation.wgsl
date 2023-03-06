struct Parameters
{
    img1_width: u32,
    img1_height: u32,
    img2_width: u32,
    img2_height: u32,
    out_width: u32,
    out_height: u32,
    fundamental_matrix: mat3x3<f32>,
    scale: f32,
    iteration_pass: u32,
    corridor_offset: i32,
    corridor_start: u32,
    corridor_end: u32,
    kernel_size: u32,
    threshold: f32,
    min_stdev: f32,
    neighbor_distance: u32,
    extend_range: f32,
    min_range: f32,
};

@group(0) @binding(0) var<uniform> parameters: Parameters;
@group(0) @binding(1) var<storage> images: array<f32>;
@group(0) @binding(2) var<storage, read_write> internals: array<f32>;
@group(0) @binding(3) var<storage, read_write> internals_int: array<i32>;
@group(0) @binding(4) var<storage, read_write> result: array<i32>;

@compute @workgroup_size(16, 16, 1)
fn init_out_data(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    let out_width = parameters.out_width;
    let out_height = parameters.out_height;

    if x < out_width && y < out_height {
        result[(out_width*y+x)*2u] = -1;
        result[(out_width*y+x)*2u+1u] = -1;
    }
}

@compute @workgroup_size(16, 16, 1)
fn prepare_initialdata_searchdata(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    let img1_width = parameters.img1_width;
    let img1_height = parameters.img1_height;

    let mean_coeff_offset: u32 = 0u;
    let corridor_stdev_offset: u32 = mean_coeff_offset + img1_width*img1_height;
    let corridor_range_offset: u32 = 0u;
    let neighbor_count_offset: u32 = corridor_range_offset + img1_width*img1_height*2u;

    if x < img1_width && y < img1_height {
        internals[mean_coeff_offset + img1_width*y+x] = 0.0;
        internals[corridor_stdev_offset + img1_width*y+x] = 0.0;
        internals_int[corridor_range_offset + (img1_width*y+x)*2u] = -1;
        internals_int[corridor_range_offset + (img1_width*y+x)*2u+1u] = -1;
        internals_int[neighbor_count_offset + img1_width*y+x] = 0;
    }
}

@compute @workgroup_size(16, 16, 1)
fn prepare_initialdata_correlation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    let img1_width = parameters.img1_width;
    let img1_height = parameters.img1_height;
    let img2_width = parameters.img2_width;
    let img2_height = parameters.img2_height;
    let kernel_size = parameters.kernel_size;
    let kernel_width = kernel_size*2u+1u;
    let kernel_point_count = f32(kernel_width*kernel_width);

    let img1_offset = 0u;
    let img2_offset = img1_width*img1_height;

    let img1_avg_offset = 0u;
    let img1_stdev_offset = img1_avg_offset + img1_width*img1_height;
    let img2_avg_offset = img1_stdev_offset + img1_width*img1_height;
    let img2_stdev_offset = img2_avg_offset + img2_width*img2_height;
    let correlation_offset = img2_stdev_offset + img2_width*img2_height;

    if x < img1_width && y < img1_height {
        internals[correlation_offset + img1_width*y+x] = -1.0;
    }

    if x >= kernel_size && x < img1_width-kernel_size && y >= kernel_size && y < img1_height-kernel_size {
        var avg = 0.0;
        var stdev = 0.0;

        for (var j=0u;j<kernel_width;j++) {
            for (var i=0u;i<kernel_width;i++) {
                let value = images[img1_offset + (y+j-kernel_size)*img1_width+(x+i-kernel_size)];
                avg += value;
            }
        }
        avg /= kernel_point_count;

        for (var j=0u;j<kernel_width;j++) {
            for (var i=0u;i<kernel_width;i++) {
                let value = images[img1_offset + (y+j-kernel_size)*img1_width+(x+i-kernel_size)];
                let delta = value-avg;
                stdev += delta*delta;
            }
        }
        stdev = sqrt(stdev/kernel_point_count);
        internals[img1_avg_offset + img1_width*y+x] = avg;
        internals[img1_stdev_offset + img1_width*y+x] = stdev;
    } else if x < img1_width && y < img1_height {
        internals[img1_avg_offset + img1_width*y+x] = 0.0;
        internals[img1_stdev_offset + img1_width*y+x] = -1.0;
    }

    if x >= kernel_size && x < img2_width-kernel_size && y >= kernel_size && y < img2_height-kernel_size {
        var avg = 0.0;
        var stdev = 0.0;

        for (var j=0u;j<kernel_width;j++) {
            for (var i=0u;i<kernel_width;i++) {
                let value = images[img2_offset + (y+j-kernel_size)*img2_width+(x+i-kernel_size)];
                avg += value;
            }
        }
        avg /= kernel_point_count;

        for (var j=0u;j<kernel_width;j++) {
            for (var i=0u;i<kernel_width;i++) {
                let value = images[img2_offset + (y+j-kernel_size)*img2_width+(x+i-kernel_size)];
                let delta = value-avg;
                stdev += delta*delta;
            }
        }
        stdev = sqrt(stdev/kernel_point_count);
        internals[img2_avg_offset + img2_width*y+x] = avg;
        internals[img2_stdev_offset + img2_width*y+x] = stdev;
    } else if x < img2_width && y < img2_height {
        internals[img2_avg_offset + img2_width*y+x] = 0.0;
        internals[img2_stdev_offset + img2_width*y+x] = -1.0;
    }
}

fn calculate_epipolar_line(x1: u32, y1: u32, corridor_offset: i32, coeff: ptr<function,vec2<f32>>, add_const: ptr<function,vec2<f32>>, corridor_offset_vector: ptr<function,vec2<i32>>) {
    let scale = parameters.scale;
    let p1 = vec3(f32(x1)/scale, f32(y1)/scale, 1.0);
    let f_p1 = parameters.fundamental_matrix*p1;
    if abs(f_p1.x)>abs(f_p1.y) {
        (*coeff).x = f32(-f_p1.y/f_p1.x);
        (*add_const).x = f32(-scale*f_p1.z/f_p1.x);
        (*corridor_offset_vector).x = corridor_offset;
        (*coeff).y = 1.0;
        (*add_const).y = 0.0;
        (*corridor_offset_vector).y = 0;
    } else {
        (*coeff).x = 1.0;
        (*add_const).x = 0.0;
        (*corridor_offset_vector).x = 0;
        (*coeff).y = f32(-f_p1.x/f_p1.y);
        (*add_const).y = f32(-scale*f_p1.z/f_p1.y);
        (*corridor_offset_vector).y = corridor_offset;
    }
}

@compute @workgroup_size(16, 16, 1)
fn prepare_searchdata(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    let img1_width = parameters.img1_width;
    let img1_height = parameters.img1_height;
    let img2_width = parameters.img2_width;
    let img2_height = parameters.img2_height;
    let out_width = parameters.out_width;
    let out_height = parameters.out_height;
    let scale = parameters.scale;
    let first_pass = parameters.iteration_pass == 0u;
    let corridor_start = i32(parameters.corridor_start);
    let corridor_end = i32(parameters.corridor_end);
    let kernel_size = parameters.kernel_size;
    let neighbor_distance = i32(parameters.neighbor_distance);
    let extend_range = parameters.extend_range;
    let min_range = parameters.min_range;

    let corridor_range_offset = 0u;
    let neighbor_count_offset = corridor_range_offset + img1_width*img1_height*2u;
    let mean_coeff_offset = 0u;
    let corridor_stdev_offset = mean_coeff_offset + img1_width*img1_height;
    var coeff: vec2<f32>;
    var add_const: vec2<f32>;
    var corridor_offset_vector: vec2<i32>;
    calculate_epipolar_line(x, y, 0, &coeff, &add_const, &corridor_offset_vector);
    let corridor_vertical = abs(coeff.y) > abs(coeff.x);

    let x_min_signed = i32(floor(f32(i32(x)-neighbor_distance)/scale));
    let x_max_signed = i32(ceil(f32(i32(x)+neighbor_distance)/scale));
    let y_min_signed = i32(f32(i32(y)-neighbor_distance)/scale)+corridor_start;
    let y_max_signed = i32(f32(i32(y)-neighbor_distance)/scale)+corridor_end;
    let x_min = u32(clamp(x_min_signed, 0, i32(out_width)));
    let x_max = u32(clamp(x_max_signed, 0, i32(out_width)));
    let y_min = u32(clamp(y_min_signed, 0, i32(out_height)));
    let y_max = u32(clamp(y_max_signed, 0, i32(out_height)));

    var neighbor_count = internals_int[neighbor_count_offset+y*img1_width+x];
    if !first_pass && neighbor_count<=0 {
        return;
    }

    var mid_corridor = internals[mean_coeff_offset + y*img1_width+x];
    var range_stdev = internals[corridor_stdev_offset + y*img1_width+x];
    if !first_pass {
        mid_corridor /= f32(neighbor_count);
    }
    for (var j=y_min;j<y_max;j++) {
        for (var i=x_min;i<x_max;i++) {
            let out_pos = j*out_width + i;
            let x2 = scale*f32(result[out_pos*2u]);
            let y2 = scale*f32(result[out_pos*2u+1u]);
            if x2<0.0 || y2<0.0 {
                continue;
            }

            var corridor_pos: f32;
            if corridor_vertical {
                corridor_pos = round((y2-add_const.y)/coeff.y);
            } else {
                corridor_pos = round((x2-add_const.x)/coeff.x);
            }
            if first_pass {
                mid_corridor += corridor_pos;
                neighbor_count++;
            } else {
                let delta = corridor_pos-mid_corridor;
                range_stdev += delta*delta;
            }
        }
    }

    if first_pass {
        internals[mean_coeff_offset + y*img1_width+x] = mid_corridor;
        internals_int[neighbor_count_offset+y*img1_width+x] = neighbor_count;
        return;
    }

    internals[corridor_stdev_offset + y*img1_width+x] = range_stdev;
    range_stdev = sqrt(range_stdev/f32(neighbor_count));
    let corridor_center = i32(round(mid_corridor));
    let corridor_length = i32(round(min_range+range_stdev*extend_range));
    var min_pos = i32(kernel_size);
    var max_pos: i32;
    if corridor_vertical {
        max_pos = i32(img2_height-kernel_size);
    } else {
        max_pos = i32(img2_width-kernel_size);
    }
    min_pos = clamp(corridor_center - corridor_length, min_pos, max_pos);
    max_pos = clamp(corridor_center + corridor_length, min_pos, max_pos);
    internals_int[corridor_range_offset + (y*img1_width+x)*2u] = min_pos;
    internals_int[corridor_range_offset + (y*img1_width+x)*2u+1u] = max_pos;
}

@compute @workgroup_size(16, 16, 1)
fn cross_correlate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x1 = global_id.x;
    let y1 = global_id.y;

    let img1_width = parameters.img1_width;
    let img1_height = parameters.img1_height;
    let img2_width = parameters.img2_width;
    let img2_height = parameters.img2_height;
    let out_width = parameters.out_width;
    let out_height = parameters.out_height;
    let scale = parameters.scale;
    let first_iteration = parameters.iteration_pass == 0u;
    let corridor_offset = parameters.corridor_offset;
    let kernel_size = parameters.kernel_size;
    let threshold = parameters.threshold;
    let min_stdev = parameters.min_stdev;
    let kernel_width = kernel_size*2u+1u;
    let kernel_point_count = f32((2u*kernel_size+1u)*(2u*kernel_size+1u));

    if x1 < kernel_size || x1 >= img1_width-kernel_size || y1 < kernel_size ||  y1 >= img1_height-kernel_size {
        return;
    }

    let img1_offset = 0u;
    let img2_offset = img1_width*img1_height;

    let img1_avg_offset = 0u;
    let img1_stdev_offset = img1_avg_offset + img1_width*img1_height;
    let img2_avg_offset = img1_stdev_offset + img1_width*img1_height;
    let img2_stdev_offset = img2_avg_offset + img2_width*img2_height;
    let correlation_offset = img2_stdev_offset + img2_width*img2_height;
    let corridor_range_offset = 0u;
    let search_area_pos = img1_width*y1 + x1;

    let avg1 = internals[img1_avg_offset + img1_width*y1+x1];
    let stdev1 = internals[img1_stdev_offset + img1_width*y1+x1];
    if stdev1 < min_stdev {
        return;
    }

    var best_corr = 0.0;
    var best_match = vec2<i32>(-1, -1);

    var corridor_start = parameters.corridor_start;
    var corridor_end = parameters.corridor_end;
    if !first_iteration {
        let min_pos_signed = internals_int[corridor_range_offset + search_area_pos*2u];
        let max_pos_signed = internals_int[corridor_range_offset + search_area_pos*2u+1u];
        if !first_iteration && (min_pos_signed<0 || max_pos_signed<0) {
            return;
        }
        let min_pos = u32(min_pos_signed);
        let max_pos = u32(max_pos_signed);
        corridor_start = clamp(corridor_start, min_pos, max_pos);
        corridor_end = clamp(corridor_end, min_pos, max_pos);
    }

    var coeff: vec2<f32>;
    var add_const: vec2<f32>;
    var corridor_offset_vector: vec2<i32>;
    calculate_epipolar_line(x1, y1, corridor_offset, &coeff, &add_const, &corridor_offset_vector);
    for (var corridor_pos: u32=corridor_start;corridor_pos<corridor_end;corridor_pos++)
    {
        let x2_signed = i32(round(coeff.x*f32(corridor_pos)+add_const.x)) + corridor_offset_vector.x;
        let y2_signed = i32(round(coeff.y*f32(corridor_pos)+add_const.y)) + corridor_offset_vector.y;
        if x2_signed < 0 || y2_signed < 0 {
            continue;
        }
        let x2 = u32(x2_signed);
        let y2 = u32(y2_signed);
        if x2 < kernel_size || x2 >= img2_width-kernel_size || y2 < kernel_size ||  y2 >= img2_height-kernel_size {
            continue;
        }

        let avg2 = internals[img2_avg_offset + img2_width*y2 + x2];
        let stdev2 = internals[img2_stdev_offset + img2_width*y2 + x2];
        if stdev2 < min_stdev {
            continue;
        }

        var corr = 0.0;
        for (var j=0u;j<kernel_width;j++) {
            for (var i=0u;i<kernel_width;i++) {
                let delta1 = images[img1_offset + (y1+j-kernel_size)*img1_width+(x1+i-kernel_size)] - avg1;
                let delta2 = images[img2_offset + (y2+j-kernel_size)*img2_width+(x2+i-kernel_size)] - avg2;
                corr += delta1*delta2;
            }
        }
        corr = corr/(stdev1*stdev2*kernel_point_count);

        if corr >= threshold && corr > best_corr {
            best_match.x = i32(round(f32(x2)/scale));
            best_match.y = i32(round(f32(y2)/scale));
            best_corr = corr;
        }
    }

    let current_corr = internals[correlation_offset + img1_width*y1 + x1];
    if (best_corr >= threshold && best_corr > current_corr)
    {
        let out_pos = out_width*u32((f32(y1)/scale)) + u32(f32(x1)/scale);
        internals[correlation_offset + img1_width*y1 + x1] = best_corr;
        result[out_pos*2u] = best_match.x;
        result[out_pos*2u+1u] = best_match.y;
    }
}
