struct Parameters
{
    img1_width: u32,
    img1_height: u32,
    img2_width: u32,
    img2_height: u32,
    out_width: u32,
    out_height: u32,
    scale: f32,
    iteration_pass: u32,
    fundamental_matrix: mat3x3<f32>,
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

var<push_constant> parameters: Parameters;
@group(0) @binding(0) var<storage> images: array<u32>;
// For searchdata: contains [mean, stdev, _] for image1
// For cross_correlate: contains [avg, stdev, corr] for image1
@group(0) @binding(1) var<storage, read_write> internals_img1: array<vec3<f32>>;
// Contains [avg, stdev] for image 2
@group(0) @binding(2) var<storage, read_write> internals_img2: array<vec2<f32>>;
// Contains [min, max, neighbor_count] for the corridor range
@group(0) @binding(3) var<storage, read_write> internals_int: array<vec3<i32>>;
@group(0) @binding(4) var<storage, read_write> result: array<vec2<i32>>;

@compute @workgroup_size(16, 16, 1)
fn init_out_data(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    let out_width = parameters.out_width;
    let out_height = parameters.out_height;

    if x < out_width && y < out_height {
        result[out_width*y+x] = vec2(-1, -1);
    }
}

@compute @workgroup_size(16, 16, 1)
fn prepare_initialdata_searchdata(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    let img1_width = parameters.img1_width;
    let img1_height = parameters.img1_height;

    if x < img1_width && y < img1_height {
        internals_img1[img1_width*y+x] = vec3(0.0, 0.0, 0.0);
        internals_int[img1_width*y+x] = vec3(-1, -1, 0);
    }
}

@compute @workgroup_size(16, 16, 1)
fn prepare_initialdata_correlation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    let img1_width = parameters.img1_width;
    let img1_height = parameters.img1_height;
    let img1_stride = ((img1_width+3u)/4u)*4u;
    let img2_width = parameters.img2_width;
    let img2_height = parameters.img2_height;
    let img2_stride = ((img2_width+3u)/4u)*4u;
    let kernel_size = parameters.kernel_size;
    let kernel_width = kernel_size*2u+1u;
    let kernel_point_count = f32(kernel_width*kernel_width);

    let img1_offset = 0u;
    let img2_offset = img1_stride/4u*img1_height;

    if x >= kernel_size && x < img1_width-kernel_size && y >= kernel_size && y < img1_height-kernel_size {
        var avg = 0.0;
        var stdev = 0.0;

        for (var j=0u;j<kernel_width;j++) {
            for (var i=0u;i<kernel_width;i++) {
                let img_pos = (y+j-kernel_size)*img1_stride+(x+i-kernel_size);
                let value = unpack4x8unorm(images[img1_offset+img_pos/4u])[img_pos%4u] * 255.0;
                avg += value;
            }
        }
        avg /= kernel_point_count;

        for (var j=0u;j<kernel_width;j++) {
            for (var i=0u;i<kernel_width;i++) {
                let img_pos = img1_offset + (y+j-kernel_size)*img1_stride+(x+i-kernel_size);
                let value = unpack4x8unorm(images[img1_offset+img_pos/4u])[img_pos%4u] * 255.0;
                let delta = value-avg;
                stdev += delta*delta;
            }
        }
        stdev = sqrt(stdev/kernel_point_count);
        internals_img1[img1_width*y+x] = vec3(avg, stdev, -1.0);
    } else if x < img1_width && y < img1_height {
        internals_img1[img1_width*y+x] = vec3(0.0, -1.0, -1.0);
    }

    if x >= kernel_size && x < img2_width-kernel_size && y >= kernel_size && y < img2_height-kernel_size {
        var avg = 0.0;
        var stdev = 0.0;

        for (var j=0u;j<kernel_width;j++) {
            for (var i=0u;i<kernel_width;i++) {
                let img_pos = (y+j-kernel_size)*img2_stride+(x+i-kernel_size);
                let value = unpack4x8unorm(images[img2_offset+img_pos/4u])[img_pos%4u] * 255.0;
                avg += value;
            }
        }
        avg /= kernel_point_count;

        for (var j=0u;j<kernel_width;j++) {
            for (var i=0u;i<kernel_width;i++) {
                let img_pos = (y+j-kernel_size)*img2_stride+(x+i-kernel_size);
                let value = unpack4x8unorm(images[img2_offset+img_pos/4u])[img_pos%4u] * 255.0;
                let delta = value-avg;
                stdev += delta*delta;
            }
        }
        stdev = sqrt(stdev/kernel_point_count);
        internals_img2[img2_width*y+x] = vec2(avg, stdev);
    } else if x < img2_width && y < img2_height {
        internals_img2[img2_width*y+x] = vec2(0.0, -1.0);
    }
}

fn calculate_epipolar_line(x1: u32, y1: u32, coeff: ptr<function,vec2<f32>>, add_const: ptr<function,vec2<f32>>) {
    let scale = parameters.scale;
    let p1 = vec3(f32(x1)/scale, f32(y1)/scale, 1.0);
    let f_p1 = parameters.fundamental_matrix*p1;
    if abs(f_p1.x)>abs(f_p1.y) {
        (*coeff).x = f32(-f_p1.y/f_p1.x);
        (*add_const).x = f32(-scale*f_p1.z/f_p1.x);
        (*coeff).y = 1.0;
        (*add_const).y = 0.0;
    } else {
        (*coeff).x = 1.0;
        (*add_const).x = 0.0;
        (*coeff).y = f32(-f_p1.x/f_p1.y);
        (*add_const).y = f32(-scale*f_p1.z/f_p1.y);
    }
}

@compute @workgroup_size(16, 16, 1)
fn prepare_searchdata(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let x_signed = i32(global_id.x);
    let y_signed = i32(global_id.y);

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

    let out_neighbor_width = i32(ceil(f32(parameters.neighbor_distance)/scale))*2+1;

    var start_pos_x = i32(floor(f32(x_signed-neighbor_distance)/scale));
    var start_pos_y = i32(floor(f32(y_signed-neighbor_distance)/scale));

    var data_int = internals_int[y*img1_width+x];
    var neighbor_count = data_int[2];
    if !first_pass && neighbor_count<=0 {
        return;
    }

    var data = internals_img1[y*img1_width+x];
    var mid_corridor = data[0];
    var range_stdev = data[1];
    if !first_pass {
        mid_corridor /= f32(neighbor_count);
    }

    var coeff: vec2<f32>;
    var add_const: vec2<f32>;
    calculate_epipolar_line(x, y, &coeff, &add_const);
    let corridor_vertical = abs(coeff.y) > abs(coeff.x);
    for (var i: i32=corridor_start;i<corridor_end;i++) {
        let x_out = start_pos_x + (i%out_neighbor_width);
        let y_out = start_pos_y + (i/out_neighbor_width);

        if x_out <=0 || y_out<=0 || x_out>=i32(out_width) || y_out>=i32(out_height) {
            continue;
        }

        let coord2 = vec2<f32>(result[u32(y_out)*out_width + u32(x_out)]) * scale;
        if coord2.x<0.0 || coord2.y<0.0 {
            continue;
        }

        var corridor_pos: f32;
        if corridor_vertical {
            corridor_pos = round((coord2.y-add_const.y)/coeff.y);
        } else {
            corridor_pos = round((coord2.x-add_const.x)/coeff.x);
        }
        if first_pass {
            mid_corridor += corridor_pos;
            neighbor_count++;
        } else {
            let delta = corridor_pos-mid_corridor;
            range_stdev += delta*delta;
        }
    }

    if first_pass {
        data[0] = mid_corridor;
        internals_img1[y*img1_width+x] = data;
        data_int[2] = neighbor_count;
        internals_int[y*img1_width+x] = data_int;
        return;
    }

    data[1] = range_stdev;
    internals_img1[y*img1_width+x] = data;
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
    internals_int[y*img1_width+x] = vec3(min_pos, max_pos, neighbor_count);
}

@compute @workgroup_size(16, 16, 1)
fn cross_correlate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x1 = global_id.x;
    let y1 = global_id.y;

    let img1_width = parameters.img1_width;
    let img1_height = parameters.img1_height;
    let img1_stride = ((img1_width+3u)/4u)*4u;
    let img2_width = parameters.img2_width;
    let img2_height = parameters.img2_height;
    let img2_stride = ((img2_width+3u)/4u)*4u;
    let out_width = parameters.out_width;
    let out_height = parameters.out_height;
    let scale = parameters.scale;
    let first_iteration = parameters.iteration_pass == 0u;
    let corridor_offset = parameters.corridor_offset;
    let kernel_size = parameters.kernel_size;
    let threshold = parameters.threshold;
    let min_stdev = parameters.min_stdev;
    let kernel_width = kernel_size*2u+1u;
    let kernel_point_count = f32(kernel_width*kernel_width);

    if x1 < kernel_size || x1 >= img1_width-kernel_size || y1 < kernel_size ||  y1 >= img1_height-kernel_size {
        return;
    }

    let img1_offset = 0u;
    let img2_offset = img1_stride/4u*img1_height;

    var data_img1 = internals_img1[img1_width*y1+x1];
    let avg1 = data_img1[0];
    let stdev1 = data_img1[1];
    let current_corr = data_img1[2];
    if stdev1 < min_stdev {
        return;
    }

    var best_corr = 0.0;
    var best_match = vec2<i32>(-1, -1);

    var corridor_start = kernel_size + parameters.corridor_start;
    var corridor_end = kernel_size + parameters.corridor_end;
    if !first_iteration {
        let data_int = internals_int[img1_width*y1 + x1];
        let min_pos_signed = data_int[0];
        let max_pos_signed = data_int[1];
        if min_pos_signed<0 || max_pos_signed<0 {
            return;
        }
        let min_pos = u32(min_pos_signed);
        let max_pos = u32(max_pos_signed);
        corridor_start = clamp(min_pos+parameters.corridor_start, min_pos, max_pos);
        corridor_end = clamp(min_pos+parameters.corridor_end, min_pos, max_pos);
        if corridor_start >= corridor_end {
            return;
        }
    }

    var coeff: vec2<f32>;
    var add_const: vec2<f32>;
    calculate_epipolar_line(x1, y1, &coeff, &add_const);
    var corridor_offset_vector: vec2<i32>;
    if abs(coeff.x) > abs(coeff.y) {
        corridor_offset_vector = vec2(0, corridor_offset);
    } else {
        corridor_offset_vector = vec2(corridor_offset, 0);
    }
    for (var corridor_pos: u32=corridor_start;corridor_pos<corridor_end;corridor_pos++) {
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

        let data_img2 = internals_img2[img2_width*y2 + x2];
        let avg2 = data_img2[0];
        let stdev2 = data_img2[1];
        if stdev2 < min_stdev {
            continue;
        }

        var corr = 0.0;
        for (var j=0u;j<kernel_width;j++) {
            for (var i=0u;i<kernel_width;i++) {
                let img1_pos = (y1+j-kernel_size)*img1_stride+(x1+i-kernel_size);
                let img2_pos = (y2+j-kernel_size)*img2_stride+(x2+i-kernel_size);
                let delta1 =  unpack4x8unorm(images[img1_offset+img1_pos/4u])[img1_pos%4u] * 255.0 - avg1;
                let delta2 =  unpack4x8unorm(images[img2_offset+img2_pos/4u])[img2_pos%4u] * 255.0 - avg2;
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

    if (best_corr >= threshold && best_corr > current_corr)
    {
        let out_pos = out_width*u32((f32(y1)/scale)) + u32(f32(x1)/scale);
        data_img1[2] = best_corr;
        internals_img1[img1_width*y1 + x1] = data_img1;
        result[out_pos] = best_match;
    }
}
