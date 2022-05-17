#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Parameters
{
    int img1_width;
    int img1_height;
    int img2_width;
    int img2_height;
    float dir_x;
    float dir_y;
    int corridor_offset;
    int corridor_segment;
    int kernel_size;
    float threshold;
};

kernel void prepare_initialdata(const device Parameters& v_29 [[buffer(0)]], const device float* v_200[[buffer(1)]], device float* v_104[[buffer(2)]], device float* v_121[[buffer(3)]], uint2 index [[thread_position_in_grid]])
{
    uint x = index.x;
    uint y = index.y;
    float kernel_point_count = float(((2 * v_29.kernel_size) + 1) * ((2 * v_29.kernel_size) + 1));
    uint img1_offset = 0u;
    uint img2_offset = uint(v_29.img1_width * v_29.img1_height);
    int img1_stdev_offset = 0 + (v_29.img1_width * v_29.img1_height);
    int img2_avg_offset = img1_stdev_offset + (v_29.img1_width * v_29.img1_height);
    int img2_stdev_offset = img2_avg_offset + (v_29.img2_width * v_29.img2_height);
    int correlation_offset = img2_stdev_offset + (v_29.img2_width * v_29.img2_height);
    bool _90 = x < uint(v_29.img1_width);
    bool _98;
    if (_90)
    {
        _98 = y < uint(v_29.img1_height);
    }
    else
    {
        _98 = _90;
    }
    if (_98)
    {
        v_104[(uint(correlation_offset) + (uint(v_29.img1_width) * y)) + x] = 0.0;
        v_121[(uint(v_29.img1_width) * y) + x] = as_type<float>(0x7fc00000u /* nan */);
    }
    bool _135 = x >= uint(v_29.kernel_size);
    bool _146;
    if (_135)
    {
        _146 = x < uint(v_29.img1_width - v_29.kernel_size);
    }
    else
    {
        _146 = _135;
    }
    bool _154;
    if (_146)
    {
        _154 = y >= uint(v_29.kernel_size);
    }
    else
    {
        _154 = _146;
    }
    bool _165;
    if (_154)
    {
        _165 = y < uint(v_29.img1_height - v_29.kernel_size);
    }
    else
    {
        _165 = _154;
    }
    if (_165)
    {
        float avg = 0.0;
        float stdev = 0.0;
        int _173 = -v_29.kernel_size;
        for (int j = _173; j <= v_29.kernel_size; j++)
        {
            int _186 = -v_29.kernel_size;
            for (int i = _186; i <= v_29.kernel_size; i++)
            {
                float value = v_200[(img1_offset + ((y + uint(j)) * uint(v_29.img1_width))) + (x + uint(i))];
                avg += value;
            }
        }
        avg /= kernel_point_count;
        int _231 = -v_29.kernel_size;
        for (int j_1 = _231; j_1 <= v_29.kernel_size; j_1++)
        {
            int _244 = -v_29.kernel_size;
            for (int i_1 = _244; i_1 <= v_29.kernel_size; i_1++)
            {
                float value_1 = v_200[(img1_offset + ((y + uint(j_1)) * uint(v_29.img1_width))) + (x + uint(i_1))];
                float delta = value_1 - avg;
                stdev += (delta * delta);
            }
        }
        stdev = sqrt(stdev / kernel_point_count);
        v_104[(0u + (uint(v_29.img1_width) * y)) + x] = avg;
        v_104[(uint(img1_stdev_offset) + (uint(v_29.img1_width) * y)) + x] = stdev;
    }
    bool _315 = x >= uint(v_29.kernel_size);
    bool _326;
    if (_315)
    {
        _326 = x < uint(v_29.img2_width - v_29.kernel_size);
    }
    else
    {
        _326 = _315;
    }
    bool _334;
    if (_326)
    {
        _334 = y >= uint(v_29.kernel_size);
    }
    else
    {
        _334 = _326;
    }
    bool _345;
    if (_334)
    {
        _345 = y < uint(v_29.img2_height - v_29.kernel_size);
    }
    else
    {
        _345 = _334;
    }
    if (_345)
    {
        float avg_1 = 0.0;
        float stdev_1 = 0.0;
        int _353 = -v_29.kernel_size;
        for (int j_2 = _353; j_2 <= v_29.kernel_size; j_2++)
        {
            int _366 = -v_29.kernel_size;
            for (int i_2 = _366; i_2 <= v_29.kernel_size; i_2++)
            {
                float value_2 = v_200[(img2_offset + ((y + uint(j_2)) * uint(v_29.img2_width))) + (x + uint(i_2))];
                avg_1 += value_2;
            }
        }
        avg_1 /= kernel_point_count;
        int _407 = -v_29.kernel_size;
        for (int j_3 = _407; j_3 <= v_29.kernel_size; j_3++)
        {
            int _420 = -v_29.kernel_size;
            for (int i_3 = _420; i_3 <= v_29.kernel_size; i_3++)
            {
                float value_3 = v_200[(img2_offset + ((y + uint(j_3)) * uint(v_29.img2_width))) + (x + uint(i_3))];
                float delta_1 = value_3 - avg_1;
                stdev_1 += (delta_1 * delta_1);
            }
        }
        stdev_1 = sqrt(stdev_1 / kernel_point_count);
        v_104[(uint(img2_avg_offset) + (uint(v_29.img2_width) * y)) + x] = avg_1;
        v_104[(uint(img2_stdev_offset) + (uint(v_29.img2_width) * y)) + x] = stdev_1;
    }
}

kernel void cross_correlate(const device Parameters& v_29 [[buffer(0)]], const device float* v_200[[buffer(1)]], device float* v_104[[buffer(2)]], device float* v_121[[buffer(3)]], uint2 index [[thread_position_in_grid]])
{
    uint x1 = index.x;
    uint y1 = index.y;
    float kernel_point_count = float(((2 * v_29.kernel_size) + 1) * ((2 * v_29.kernel_size) + 1));
    bool _518 = x1 < uint(v_29.kernel_size);
    bool _530;
    if (!_518)
    {
        _530 = x1 >= uint(v_29.img1_width - v_29.kernel_size);
    }
    else
    {
        _530 = _518;
    }
    bool _539;
    if (!_530)
    {
        _539 = y1 < uint(v_29.kernel_size);
    }
    else
    {
        _539 = _530;
    }
    bool _551;
    if (!_539)
    {
        _551 = y1 >= uint(v_29.img1_height - v_29.kernel_size);
    }
    else
    {
        _551 = _539;
    }
    if (_551)
    {
        return;
    }
    uint img1_offset = 0u;
    uint img2_offset = uint(v_29.img1_width * v_29.img1_height);
    int img1_stdev_offset = 0 + (v_29.img1_width * v_29.img1_height);
    int img2_avg_offset = img1_stdev_offset + (v_29.img1_width * v_29.img1_height);
    int img2_stdev_offset = img2_avg_offset + (v_29.img2_width * v_29.img2_height);
    int correlation_offset = img2_stdev_offset + (v_29.img2_width * v_29.img2_height);
    float avg1 = v_104[(0u + (uint(v_29.img1_width) * y1)) + x1];
    float stdev1 = v_104[(uint(img1_stdev_offset) + (uint(v_29.img1_width) * y1)) + x1];
    float best_corr = 0.0;
    float best_distance = as_type<float>(0x7fc00000u /* nan */);
    bool corridor_vertical = abs(v_29.dir_y) > abs(v_29.dir_x);
    int _633;
    if (corridor_vertical)
    {
        _633 = v_29.img2_height;
    }
    else
    {
        _633 = v_29.img2_width;
    }
    int corridor_max = _633 - v_29.kernel_size;
    int min_l = min((v_29.kernel_size + (v_29.corridor_segment * 256)), corridor_max);
    int max_l = min((min_l + 256), corridor_max);
    float _663;
    if (corridor_vertical)
    {
        _663 = v_29.dir_x / v_29.dir_y;
    }
    else
    {
        _663 = v_29.dir_y / v_29.dir_x;
    }
    float corridor_coeff = _663;
    int y2;
    int x2;
    for (int corridor_pos = min_l; corridor_pos < max_l; corridor_pos++)
    {
        if (corridor_vertical)
        {
            y2 = corridor_pos;
            x2 = (int(x1) + v_29.corridor_offset) + int(float(y2 - int(y1)) * corridor_coeff);
            bool _712 = x2 < v_29.kernel_size;
            bool _723;
            if (!_712)
            {
                _723 = x2 >= (v_29.img2_width - v_29.kernel_size);
            }
            else
            {
                _723 = _712;
            }
            if (_723)
            {
                continue;
            }
        }
        else
        {
            x2 = corridor_pos;
            y2 = (int(y1) + v_29.corridor_offset) + int(float(x2 - int(x1)) * corridor_coeff);
            bool _746 = y2 < v_29.kernel_size;
            bool _757;
            if (!_746)
            {
                _757 = y2 >= (v_29.img2_height - v_29.kernel_size);
            }
            else
            {
                _757 = _746;
            }
            if (_757)
            {
                continue;
            }
        }
        float avg2 = v_104[(img2_avg_offset + (v_29.img2_width * y2)) + x2];
        float stdev2 = v_104[(img2_stdev_offset + (v_29.img2_width * y2)) + x2];
        float corr = 0.0;
        int _787 = -v_29.kernel_size;
        for (int j = _787; j <= v_29.kernel_size; j++)
        {
            int _800 = -v_29.kernel_size;
            for (int i = _800; i <= v_29.kernel_size; i++)
            {
                float delta1 = v_200[(img1_offset + ((y1 + uint(j)) * uint(v_29.img1_width))) + (x1 + uint(i))] - avg1;
                float delta2 = v_200[(img2_offset + uint((y2 + j) * v_29.img2_width)) + uint(x2 + i)] - avg2;
                corr += (delta1 * delta2);
            }
        }
        corr /= ((stdev1 * stdev2) * kernel_point_count);
        if (corr > best_corr)
        {
            float dx = float(x2) - float(x1);
            float dy = float(y2) - float(y1);
            float _distance = (dx * dx) + (dy * dy);
            best_distance = _distance;
            best_corr = corr;
        }
    }
    float current_corr = v_104[(uint(correlation_offset) + (uint(v_29.img1_width) * y1)) + x1];
    bool _909 = best_corr > current_corr;
    bool _917;
    if (_909)
    {
        _917 = best_corr >= v_29.threshold;
    }
    else
    {
        _917 = _909;
    }
    if (_917)
    {
        v_104[(uint(correlation_offset) + (uint(v_29.img1_width) * y1)) + x1] = best_corr;
        v_121[(uint(v_29.img1_width) * y1) + x1] = -sqrt(best_distance);
    }
}

