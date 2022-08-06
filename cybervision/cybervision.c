#include <stdlib.h>

#include <fast.h>

#include "correlation.h"
//#include "triangulation.h"

/*
 * Parameters
 */

const int cybervision_fast_threshold = 15;
const int cybervision_fast_mode = 12;
const int cybervision_fast_nonmax = 1;

/*
 * Helper functions
 */
/*
long read_long_attr(PyObject *obj, const char* attr_name)
{
    long value;
    PyObject *attr_value;

    attr_value = PyObject_GetAttrString(obj, attr_name);
    if (attr_value == NULL)
    {
        PyErr_Format(CybervisionError, "Failed to parse attribute %s", attr_name);
        return -1;
    }
    value = PyLong_AsLong(attr_value);
    Py_DECREF(attr_value);
    return value;
}

int read_img_bytes(PyObject *obj, Py_buffer *buffer)
{
    PyObject *img_bytes;
    img_bytes = PyObject_CallMethod(obj, "tobytes", "ss", "raw", "L", NULL);
    if (!img_bytes)
    {
        PyErr_SetString(CybervisionError, "Failed to convert image");
        return 0;
    }
    if (PyObject_GetBuffer(img_bytes, buffer, PyBUF_ANY_CONTIGUOUS) != 0)
    {
        PyErr_SetString(CybervisionError, "Failed to get image buffer");
        Py_DECREF(img_bytes);
        return 0;
    }
    Py_DECREF(img_bytes);
    return 1;
}

correlation_point *read_points(PyObject *points)
{
    Py_ssize_t points_size = PyList_Size(points);
    correlation_point *converted_points = malloc(sizeof(correlation_point)*points_size);
    for (Py_ssize_t p=0;p<points_size;p++)
    {
        PyObject *point = PyList_GetItem(points, p);
        correlation_point *converted_point = &converted_points[p];

        if (!PyArg_ParseTuple(point, "ii", &converted_point->x, &converted_point->y))
        {
            PyErr_SetString(CybervisionError, "Failed to parse point");
            free(converted_points);
            return NULL;
        }
    }
    return converted_points;
}

void free_match_task(PyObject *task_object)
{
    match_task *task = PyCapsule_GetPointer(task_object, NULL);
    if (task == NULL)
        return;
    correlation_match_points_cancel(task);
    correlation_match_points_complete(task);

    if (task->img1.img != NULL)
        free(task->img1.img);
    if (task->img2.img != NULL)
        free(task->img2.img);
    if (task->points1 != NULL)
        free(task->points1);
    if (task->points2 != NULL)
        free(task->points2);
    if (task->matches != NULL)
        free(task->matches);
    free(task);
}

int read_matches(PyObject *points1, PyObject *points2, PyObject *matches, float ransac_min_length, ransac_match **converted_matches, size_t *converted_matches_count)
{
    Py_ssize_t matches_size = PyList_Size(matches);
    *converted_matches = malloc(sizeof(ransac_match)*matches_size);
    *converted_matches_count = 0;
    for (Py_ssize_t m=0;m<matches_size;m++)
    {
        PyObject *match = PyList_GetItem(matches, m);
        int p1, p2;
        float corr;
        PyObject *point1, *point2;
        float dx, dy, length;

        ransac_match *converted_match = &((*converted_matches)[*converted_matches_count]);
        if (!PyArg_ParseTuple(match, "iif", &p1, &p2, &corr))
        {
            PyErr_SetString(CybervisionError, "Failed to parse match");
            free(*converted_matches);
            *converted_matches = NULL;
            return 0;
        }

        point1 = PyList_GetItem(points1, p1);
        if (!PyArg_ParseTuple(point1, "ii", &converted_match->x1, &converted_match->y1))
        {
            PyErr_SetString(CybervisionError, "Failed to parse point 1");
            free(*converted_matches);
            *converted_matches = NULL;
            return 0;
        }

        point2 = PyList_GetItem(points2, p2);
        if (!PyArg_ParseTuple(point2, "ii", &converted_match->x2, &converted_match->y2))
        {
            PyErr_SetString(CybervisionError, "Failed to parse point 2");
            free(*converted_matches);
            *converted_matches = NULL;
            return 0;
        }

        dx = (float)(converted_match->x2 - converted_match->x1);
        dy = (float)(converted_match->y2 - converted_match->y1);
        length = sqrtf(dx*dx + dy*dy);

        if (length >= ransac_min_length)
        {
            (*converted_matches_count)++;
        }
    }
    return 1;
}

void free_cross_correlate_task(PyObject *task_object)
{
    cross_correlate_task *task = PyCapsule_GetPointer(task_object, NULL);
    if (task == NULL)
        return;
    correlation_cross_correlate_cancel(task);
    correlation_cross_correlate_complete(task);

    if (task->img1.img != NULL)
        free(task->img1.img);
    if (task->img2.img != NULL)
        free(task->img2.img);
    if (task->out_points != NULL)
        free(task->out_points);
    free(task);
}

void free_ransac_task(PyObject *task_object)
{
    ransac_task *task = PyCapsule_GetPointer(task_object, NULL);
    if (task == NULL)
        return;
    correlation_ransac_cancel(task);
    correlation_ransac_complete(task);

    if (task->matches != NULL)
        free(task->matches);
    free(task);
}

void free_surface_data(PyObject *data_object)
{
    surface_data *data = PyCapsule_GetPointer(data_object, NULL);
    if (data == NULL)
        return;
    if (data->depth != NULL)
        free(data->depth);
    free(data);
}
*/

correlation_point* fast_detect(correlation_image *img, size_t  *count)
{
    int threshold = cybervision_fast_threshold, mode = cybervision_fast_mode, nonmax = cybervision_fast_nonmax;
    int num_corners;
    xy* corners;
    correlation_point *out = NULL;

    *count = 0;
    
    if (nonmax && mode == 9)
        corners = fast9_detect_nonmax(img->img, img->width, img->height, img->width, threshold, &num_corners);
    else if (nonmax && mode == 10)
        corners = fast10_detect_nonmax(img->img, img->width, img->height, img->width, threshold, &num_corners);
    else if (nonmax && mode == 11)
        corners = fast11_detect_nonmax(img->img, img->width, img->height, img->width, threshold, &num_corners);
    else if (nonmax && mode == 12)
        corners = fast12_detect_nonmax(img->img, img->width, img->height, img->width, threshold, &num_corners);
    else if (!nonmax && mode == 9)
        corners = fast9_detect(img->img, img->width,img-> height, img->width, threshold, &num_corners);
    else if (!nonmax && mode == 10)
        corners = fast10_detect(img->img, img->width, img->height, img->width, threshold, &num_corners);
    else if (!nonmax && mode == 11)
        corners = fast11_detect(img->img, img->width, img->height, img->width, threshold, &num_corners);
    else if (!nonmax && mode == 12)
        corners = fast12_detect(img->img, img->width, img->height, img->width, threshold, &num_corners);
    else
        return out;

    *count = (size_t)num_corners;
    out = malloc(sizeof(correlation_point)* num_corners);
    for(int i=0;i<num_corners;i++) {
        out[i].x = corners[i].x;
        out[i].y = corners[i].y;
    }

    free(corners);
    return out;
}

/*
static PyObject *
machine_match_start(PyObject *self, PyObject *args)
{
    int kernel_size;
    float threshold;
    int num_threads;
    PyObject *img1, *img2;
    Py_buffer img_buffer;
    PyObject *points1, *points2;
    match_task *task;

    PyObject *out = NULL;

    if (!PyArg_ParseTuple(args, "OOOOifi", &img1, &img2, &points1, &points2, &kernel_size, &threshold, &num_threads))
    {
        PyErr_SetString(CybervisionError, "Failed to parse args");
        return NULL;
    }

    task = malloc(sizeof(match_task));
    task->kernel_size = kernel_size;
    task->threshold = threshold;
    task->num_threads = num_threads;
    task->img1.img = NULL;
    task->img2.img = NULL;
    task->points1 = NULL;
    task->points2 = NULL;

    out = PyCapsule_New(task, NULL, free_match_task);

    task->img1.width = read_long_attr(img1, "width");
    task->img1.height = read_long_attr(img1, "height");
    task->img2.width = read_long_attr(img2, "width");
    task->img2.height = read_long_attr(img2, "height");

    if (task->img1.width == -1 || task->img1.height == -1 || task->img2.width == -1 || task->img2.height == -1)
    {
        PyErr_SetString(CybervisionError, "Failed to get image size");
        Py_DECREF(out);
        return NULL;
    }
    task->points1 = read_points(points1);
    if (task->points1 == NULL)
    {
        Py_DECREF(out);
        return NULL;
    }
    task->points2 = read_points(points2);
    if (task->points2 == NULL)
    {
        Py_DECREF(out);
        return NULL;
    }
    task->points1_size = PyList_Size(points1);
    task->points2_size = PyList_Size(points2);
    
    if (!read_img_bytes(img1, &img_buffer))
    {
        Py_DECREF(out);
        return NULL;
    }
    task->img1.img = malloc(img_buffer.len);
    memcpy(task->img1.img, img_buffer.buf, img_buffer.len);
    PyBuffer_Release(&img_buffer);

    if (!read_img_bytes(img2, &img_buffer))
    {
        Py_DECREF(out);
        return NULL;
    }
    task->img2.img = malloc(img_buffer.len);
    memcpy(task->img2.img, img_buffer.buf, img_buffer.len);
    PyBuffer_Release(&img_buffer);

    if(!correlation_match_points_start(task))
    {
        PyErr_SetString(CybervisionError, "Failed to start point matching task");
        Py_DECREF(out);
        return NULL;
    }

    return out;
}

static PyObject *
machine_match_status(PyObject *self, PyObject *args)
{
    PyObject *task_object;
    match_task *task;

    if (!PyArg_ParseTuple(args, "O", &task_object))
    {
        PyErr_SetString(CybervisionError, "Failed to parse args");
        return NULL;
    }

    task = PyCapsule_GetPointer(task_object, NULL);
    if (task == NULL)
    {
        PyErr_SetString(CybervisionError, "Failed to get task from args");
        return NULL;
    }

    return Py_BuildValue("(Of)", task->completed?Py_True:Py_False, task->percent_complete);
}

static PyObject *
machine_match_result(PyObject *self, PyObject *args)
{
    PyObject *task_object;
    match_task *task;

    PyObject *out = NULL;

    if (!PyArg_ParseTuple(args, "O", &task_object))
    {
        PyErr_SetString(CybervisionError, "Failed to parse args");
        return NULL;
    }

    task = PyCapsule_GetPointer(task_object, NULL);
    if (task == NULL)
    {
        PyErr_SetString(CybervisionError, "Failed to get task from args");
        return NULL;
    }
    
    out = PyList_New(task->matches_count);

    correlation_match_points_complete(task);
    for (size_t i=0;i<task->matches_count;i++)
    {
        correlation_match match = task->matches[i];
        PyObject *correlation_value = Py_BuildValue("(iif)", match.point1, match.point2, match.corr);
        PyList_SetItem(out, i, correlation_value);
    }

    return out;
}

static PyObject *
machine_ransac_start(PyObject *self, PyObject *args)
{
    PyObject *points1, *points2, *matches;
    float ransac_min_length;
    int ransac_k;
    int ransac_n;
    float ransac_t;
    int ransac_d;
    int num_threads;
    ransac_task *task;

    PyObject *out = NULL;

    if (!PyArg_ParseTuple(args, "OOOfiifii", &points1, &points2, &matches, &ransac_min_length, &ransac_k, &ransac_n, &ransac_t, &ransac_d, &num_threads))
    {
        PyErr_SetString(CybervisionError, "Failed to parse args");
        return NULL;
    }

    task = malloc(sizeof(ransac_task));
    task->ransac_k = ransac_k;
    task->ransac_n = ransac_n;
    task->ransac_t = ransac_t;
    task->ransac_d = ransac_d;
    task->num_threads = num_threads;
    task->matches = NULL;
    task->internal = NULL;

    out = PyCapsule_New(task, NULL, free_ransac_task);

    if(!read_matches(points1, points2, matches, ransac_min_length, &task->matches, &task->matches_count))
    {
        Py_DECREF(out);
        return NULL;
    }

    if(!correlation_ransac_start(task))
    {
        PyErr_SetString(CybervisionError, "Failed to start RANSAC task");
        Py_DECREF(out);
        return NULL;
    }

    return out;
}

static PyObject *
machine_ransac_status(PyObject *self, PyObject *args)
{
    PyObject *task_object;
    ransac_task *task;

    if (!PyArg_ParseTuple(args, "O", &task_object))
    {
        PyErr_SetString(CybervisionError, "Failed to parse args");
        return NULL;
    }

    task = PyCapsule_GetPointer(task_object, NULL);
    if (task == NULL)
    {
        PyErr_SetString(CybervisionError, "Failed to get task from args");
        return NULL;
    }

    return Py_BuildValue("(Of)", task->completed?Py_True:Py_False, task->percent_complete);
}

static PyObject *
machine_ransac_result(PyObject *self, PyObject *args)
{
    PyObject *task_object;
    ransac_task *task;

    if (!PyArg_ParseTuple(args, "O", &task_object))
    {
        PyErr_SetString(CybervisionError, "Failed to parse args");
        return NULL;
    }

    task = PyCapsule_GetPointer(task_object, NULL);
    if (task == NULL)
    {
        PyErr_SetString(CybervisionError, "Failed to get task from args");
        return NULL;
    }
    
    correlation_ransac_complete(task);
    
    return Py_BuildValue("(iff)", task->result_matches_count, task->dir_x, task->dir_y);
}

static PyObject *
machine_correlate_init(PyObject *self, PyObject *args)
{
    float dir_x, dir_y;
    int neighbor_distance;
    float max_slope;
    int corridor_size;
    int kernel_size;
    PyObject *img1, *img2;
    int w1, h1;
    float threshold;
    int num_threads;
    cross_correlate_task *task;

    PyObject *out = NULL;

    if (!PyArg_ParseTuple(args, "OOffifiifi", &img1, &img2, &dir_x, &dir_y, &neighbor_distance, &max_slope,
        &corridor_size, &kernel_size, &threshold, &num_threads))
    {
        PyErr_SetString(CybervisionError, "Failed to parse args");
        return NULL;
    }

    task = malloc(sizeof(cross_correlate_task));
    task->dir_x = dir_x;
    task->dir_y = dir_y;
    task->neighbor_distance = neighbor_distance;
    task->max_slope = max_slope;
    task->corridor_size = corridor_size;
    task->kernel_size = kernel_size;
    task->threshold = threshold;
    task->num_threads = num_threads;
    task->img1.img = NULL;
    task->img2.img = NULL;
    task->iteration = 0;
    task->internal = NULL;

    out = PyCapsule_New(task, NULL, free_cross_correlate_task);

    w1 = read_long_attr(img1, "width");
    h1 = read_long_attr(img1, "height");
    task->out_points = malloc(sizeof(float)*w1*h1);
    for (int i=0;i<w1*h1;i++)
        task->out_points[i] = NAN;

    task->out_width = w1;
    task->out_height = h1;

    return out;
}

static PyObject *
machine_correlate_start(PyObject *self, PyObject *args)
{
    PyObject *task_object;
    cross_correlate_task *task;
    float scale;
    PyObject *img1, *img2;
    Py_buffer img_buffer;

    if (!PyArg_ParseTuple(args, "OOOf", &task_object, &img1, &img2, &scale))
    {
        PyErr_SetString(CybervisionError, "Failed to parse args");
        return NULL;
    }

    task = PyCapsule_GetPointer(task_object, NULL);
    if (task == NULL)
    {
        PyErr_SetString(CybervisionError, "Failed to get task from args");
        return NULL;
    }

    task->scale = scale;

    if (task->img1.img != NULL)
    {
        free(task->img1.img);
        task->img1.img = NULL;
    }
    if (task->img2.img != NULL)
    {
        free(task->img2.img);
        task->img2.img = NULL;
    }

    task->iteration++;

    task->img1.width = read_long_attr(img1, "width");
    task->img1.height = read_long_attr(img1, "height");
    task->img2.width = read_long_attr(img2, "width");
    task->img2.height = read_long_attr(img2, "height");

    if (task->img1.width == -1 || task->img1.height == -1 || task->img2.width == -1 || task->img2.height == -1)
    {
        PyErr_SetString(CybervisionError, "Failed to get image size");
        return NULL;
    }

    if (!read_img_bytes(img1, &img_buffer))
    {
        return NULL;
    }
    task->img1.img = malloc(img_buffer.len);
    memcpy(task->img1.img, img_buffer.buf, img_buffer.len);
    PyBuffer_Release(&img_buffer);

    if (!read_img_bytes(img2, &img_buffer))
    {
        return NULL;
    }
    task->img2.img = malloc(img_buffer.len);
    memcpy(task->img2.img, img_buffer.buf, img_buffer.len);
    PyBuffer_Release(&img_buffer);

    correlation_cross_correlate_complete(task);
    if(!correlation_cross_correlate_start(task))
    {
        PyErr_SetString(CybervisionError, "Failed to start cross correlation task");
        return NULL;
    }

    return Py_None;
}

static PyObject *
machine_correlate_status(PyObject *self, PyObject *args)
{
    PyObject *task_object;
    cross_correlate_task *task;

    if (!PyArg_ParseTuple(args, "O", &task_object))
    {
        PyErr_SetString(CybervisionError, "Failed to parse args");
        return NULL;
    }

    task = PyCapsule_GetPointer(task_object, NULL);
    if (task == NULL)
    {
        PyErr_SetString(CybervisionError, "Failed to get task from args");
        return NULL;
    }

    return Py_BuildValue("(Of)", task->completed?Py_True:Py_False, task->percent_complete);
}

static PyObject *
machine_correlate_result(PyObject *self, PyObject *args)
{
    PyObject *task_object;
    cross_correlate_task *task;

    surface_data *out = NULL;

    if (!PyArg_ParseTuple(args, "O", &task_object))
    {
        PyErr_SetString(CybervisionError, "Failed to parse args");
        return NULL;
    }

    task = PyCapsule_GetPointer(task_object, NULL);
    if (task == NULL)
    {
        PyErr_SetString(CybervisionError, "Failed to get task from args");
        return NULL;
    }
    
    correlation_cross_correlate_complete(task);

    if (task->error != NULL)
    {
        PyErr_Format(CybervisionError, "Correlation task failed: %s", task->error);
        return NULL;
    }

    for (int i=0;i<task->out_width*task->out_height;i++)
        task->out_points[i] = -task->out_points[i];
    
    out = malloc(sizeof(surface_data));
    out->width = task->out_width;
    out->height = task->out_height;
    out->depth = task->out_points;
    task->out_points = NULL;

    return PyCapsule_New(out, NULL, free_surface_data);
}

static PyObject *
machine_triangulate_points(PyObject *self, PyObject *args)
{
    PyObject *surface_object;
    surface_data *data;
    PyObject *out_points, *out_simplices;

    if (!PyArg_ParseTuple(args, "O", &surface_object))
    {
        PyErr_SetString(CybervisionError, "Failed to parse args");
        return NULL;
    }

    data = PyCapsule_GetPointer(surface_object, NULL);
    if (data == NULL)
    {
        PyErr_SetString(CybervisionError, "Failed to get data from args");
        return NULL;
    }

    out_points = PyList_New(0);
    out_simplices = PyList_New(0);
    if (!triangulation_triangulate(data, out_points, out_simplices))
    {
        PyErr_SetString(CybervisionError, "Failed to triangulate points");
        Py_DECREF(out_points);
        Py_DECREF(out_simplices);
        return NULL;
    }

    return Py_BuildValue("(OO)", out_points, out_simplices);
}
*/
