#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>
#include "correlation.h"
#include "gpu_correlation.h"
#include "triangulation.h"

#include "fast/fast.h"

static PyObject *CybervisionError;

/*
 * Helper functions
 */
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

void free_triangulation_data(PyObject *data_object)
{
    triangulation_data *data = PyCapsule_GetPointer(data_object, NULL);
    if (data == NULL)
        return;
    if (data->points != NULL)
        free(data->points);
    free(data);
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

/*
 * Python exported functions
 */
static PyObject *
machine_detect(PyObject *self, PyObject *args)
{
    PyObject *img;
    Py_buffer img_buffer;
    int width, height, threshold, mode, nonmax;
    xy* corners = NULL;
    int num_corners;

    PyObject *out = NULL;

    if (!PyArg_ParseTuple(args, "Oiib", &img, &threshold, &mode, &nonmax))
    {
        PyErr_SetString(CybervisionError, "Failed to parse args");
        return NULL;
    }

    width = read_long_attr(img, "width");
    if (width == -1)
        return NULL;
    height = read_long_attr(img, "height");
    if (height == -1)
        return NULL;

    if (!read_img_bytes(img, &img_buffer))
        return NULL;
    
    if (nonmax && mode == 9)
        corners = fast9_detect_nonmax(img_buffer.buf, width, height, width, threshold, &num_corners);
    else if (nonmax && mode == 10)
        corners = fast10_detect_nonmax(img_buffer.buf, width, height, width, threshold, &num_corners);
    else if (nonmax && mode == 11)
        corners = fast11_detect_nonmax(img_buffer.buf, width, height, width, threshold, &num_corners);
    else if (nonmax && mode == 12)
        corners = fast12_detect_nonmax(img_buffer.buf, width, height, width, threshold, &num_corners);
    else if (!nonmax && mode == 9)
        corners = fast9_detect(img_buffer.buf, width, height, width, threshold, &num_corners);
    else if (!nonmax && mode == 10)
        corners = fast10_detect(img_buffer.buf, width, height, width, threshold, &num_corners);
    else if (!nonmax && mode == 11)
        corners = fast11_detect(img_buffer.buf, width, height, width, threshold, &num_corners);
    else if (!nonmax && mode == 12)
        corners = fast12_detect(img_buffer.buf, width, height, width, threshold, &num_corners);
    else
    {
        PyErr_Format(CybervisionError, "Unsupported FAST options nonmax=%i mode=%i", nonmax, mode);
        PyBuffer_Release(&img_buffer);
        return NULL;
    }
    PyBuffer_Release(&img_buffer);

    out = PyList_New(num_corners);
    for(int i=0;i<num_corners;i++) {
        PyObject *corner = Py_BuildValue("(ii)", corners[i].x, corners[i].y);
        PyList_SetItem(out, i, corner);
    }

    free(corners);
    return out;
}

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
machine_correlate_start(PyObject *self, PyObject *args)
{
    const char* correlation_mode_str;
    float dir_x, dir_y;
    int corridor_size;
    int kernel_size;
    float threshold;
    int num_threads;
    int corridor_segment_length;
    PyObject *img1, *img2;
    Py_buffer img_buffer;
    cross_correlate_task *task;

    PyObject *out = NULL;

    if (!PyArg_ParseTuple(args, "sOOffiifii", &correlation_mode_str, &img1, &img2, &dir_x, &dir_y, &corridor_size, &kernel_size, &threshold, &num_threads, &corridor_segment_length))
    {
        PyErr_SetString(CybervisionError, "Failed to parse args");
        return NULL;
    }

    task = malloc(sizeof(match_task));
    task->dir_x = dir_x;
    task->dir_y = dir_y;
    task->corridor_size = corridor_size;
    task->kernel_size = kernel_size;
    task->threshold = threshold;
    task->num_threads = num_threads;
    task->corridor_segment_length = corridor_segment_length;
    task->img1.img = NULL;
    task->img2.img = NULL;
    task->out_points = NULL;

    if (strcmp("cpu", correlation_mode_str)==0)
        task->correlation_mode = CORRELATION_MODE_CPU;
    else if (strcmp("gpu", correlation_mode_str)==0)
        task->correlation_mode = CORRELATION_MODE_GPU;
    else
        task->correlation_mode = -1;

    out = PyCapsule_New(task, NULL, free_cross_correlate_task);

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

    task->out_points = malloc(sizeof(float)*task->img1.width*task->img2.height);
    if (task->correlation_mode == CORRELATION_MODE_CPU)
    {
        if(!correlation_cross_correlate_start(task))
        {
            PyErr_SetString(CybervisionError, "Failed to start cross correlation task");
            Py_DECREF(out);
            return NULL;
        }
    }
    else if (task->correlation_mode == CORRELATION_MODE_GPU)
    {
        if(!gpu_correlation_cross_correlate_start(task))
        {
            PyErr_SetString(CybervisionError, "Failed to start cross correlation task");
            Py_DECREF(out);
            return NULL;
        }
    }
    else
    {
        PyErr_Format(CybervisionError, "Unsupported correlation mode %s", correlation_mode_str);
        Py_DECREF(out);
        return NULL;
    }

    return out;
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

    triangulation_data *out = NULL;
    triangulation_point *current_point = NULL;

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
    
    if (task->correlation_mode==CORRELATION_MODE_CPU)
        correlation_cross_correlate_complete(task);
    else if (task->correlation_mode==CORRELATION_MODE_GPU)
        gpu_correlation_cross_correlate_complete(task);

    if (task->error != NULL)
    {
        PyErr_Format(CybervisionError, "Correlation task failed: %s", task->error);
        return NULL;
    }

    out = malloc(sizeof(triangulation_data));
    out->num_points = 0;

    for (int y=0;y<task->img1.height;y++)
    {
        for (int x=0;x<task->img1.width;x++)
        {
            float depth = task->out_points[y*task->img1.width + x];
            if (isfinite(depth))
                out->num_points++;
        }
    }

    out->points = malloc(sizeof(triangulation_point)*out->num_points);
    current_point = out->points;

    for (int y=0;y<task->img1.height;y++)
    {
        for (int x=0;x<task->img1.width;x++)
        {
            float depth = task->out_points[y*task->img1.width + x];
            triangulation_point point;
            if (!isfinite(depth))
                continue;
            point.x = x;
            point.y = y;
            point.z = depth;
            *(current_point++) = point;
        }
    }

    return PyCapsule_New(out, NULL, free_triangulation_data);
}

static PyMethodDef MachineMethods[] = {
    {"detect", machine_detect, METH_VARARGS, "Detect keypoints with FAST."},
    {"match_start", machine_match_start, METH_VARARGS, "Start a task to find correlation between image points."},
    {"match_status", machine_match_status, METH_VARARGS, "Status of a task to find correlation between image points."},
    {"match_result", machine_match_result, METH_VARARGS, "Result of a task to find correlation between image points."},
    {"correlate_start", machine_correlate_start, METH_VARARGS, "Start a task to find cross-correlation between images."},
    {"correlate_status", machine_correlate_status, METH_VARARGS, "Status of a task to find cross-correlation between images."},
    {"correlate_result", machine_correlate_result, METH_VARARGS, "Result of a task to find cross-correlation between images."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef machinemodule = {
    PyModuleDef_HEAD_INIT, "cybervision.machine", NULL, -1, MachineMethods
};

PyMODINIT_FUNC
PyInit_machine(void)
{
    PyObject *m;
    m = PyModule_Create(&machinemodule);
    if (m == NULL)
        return m;

    CybervisionError = PyErr_NewException("cybervision.error", NULL, NULL);
    Py_XINCREF(CybervisionError);
    if (PyModule_AddObject(m, "error", CybervisionError) < 0) {
        Py_XDECREF(CybervisionError);
        Py_CLEAR(CybervisionError);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
