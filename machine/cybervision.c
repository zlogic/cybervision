#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>
#include "correlation.h"

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

void add_match(size_t p1, size_t p2, float corr, void* cb_args)
{
    PyObject *out = cb_args;
    PyObject *correlation_value = Py_BuildValue("(iif)", p1, p2, corr);
    PyList_Append(out, correlation_value);
    Py_DECREF(correlation_value);
}

/*
 * Python exported functions
 */
static PyObject *
cybervision_detect(PyObject *self, PyObject *args)
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
cybervision_match(PyObject *self, PyObject *args)
{
    int kernel_size;
    float threshold;
    int num_threads;
    PyObject *img1, *img2;
    Py_buffer img1_buffer, img2_buffer;
    correlation_image c_image1, c_image2;
    PyObject *points1, *points2;
    correlation_point *c_points1, *c_points2;

    PyObject *out = NULL;

    if (!PyArg_ParseTuple(args, "OOOOifi", &img1, &img2, &points1, &points2, &kernel_size, &threshold, &num_threads))
    {
        PyErr_SetString(CybervisionError, "Failed to parse args");
        return NULL;
    }
    c_image1.width = read_long_attr(img1, "width");
    if (c_image1.width == -1)
        return NULL;
    c_image1.height = read_long_attr(img1, "height");
    if (c_image1.height == -1)
        return NULL;
    c_image2.width = read_long_attr(img2, "width");
    if (c_image2.width == -1)
        return NULL;
    c_image2.height = read_long_attr(img2, "height");
    if (c_image2.height == -1)
        return NULL;

    c_points1 = read_points(points1);
    if (c_points1 == NULL)
        return NULL;

    c_points2 = read_points(points2);
    if (c_points2 == NULL)
    {
        free(c_points1);
        return NULL;
    }

    if (!read_img_bytes(img1, &img1_buffer))
        return NULL;
    if (!read_img_bytes(img2, &img2_buffer))
    {
        PyBuffer_Release(&img1_buffer);
        return NULL;
    }
    c_image1.img = img1_buffer.buf;
    c_image2.img = img2_buffer.buf;

    out = PyList_New(0);

    if(!correlation_correlate_points(&c_image1, &c_image2, c_points1, c_points2, PyList_Size(points1), PyList_Size(points2), 
        kernel_size, threshold, num_threads, add_match, out))
    {
        PyErr_SetString(CybervisionError, "Failed to correlate points");
        Py_DECREF(out);
        PyBuffer_Release(&img1_buffer);
        PyBuffer_Release(&img2_buffer);
        return NULL;
    }

    PyBuffer_Release(&img1_buffer);
    PyBuffer_Release(&img2_buffer);

    return out;
}

static PyObject *
cybervision_correlate(PyObject *self, PyObject *args)
{
    float angle;
    int corridor_size;
    int kernel_size;
    float threshold;
    int num_threads;
    PyObject *img1, *img2;
    Py_buffer img1_buffer, img2_buffer;
    correlation_image c_image1, c_image2;

    float *out_points;

    PyObject *out = NULL;

    if (!PyArg_ParseTuple(args, "OOfiifi", &img1, &img2, &angle, &corridor_size, &kernel_size, &threshold, &num_threads))
    {
        PyErr_SetString(CybervisionError, "Failed to parse args");
        return NULL;
    }
    c_image1.width = read_long_attr(img1, "width");
    if (c_image1.width == -1)
        return NULL;
    c_image1.height = read_long_attr(img1, "height");
    if (c_image1.height == -1)
        return NULL;
    c_image2.width = read_long_attr(img2, "width");
    if (c_image2.width == -1)
        return NULL;
    c_image2.height = read_long_attr(img2, "height");
    if (c_image2.height == -1)
        return NULL;

    if (!read_img_bytes(img1, &img1_buffer))
        return NULL;
    if (!read_img_bytes(img2, &img2_buffer))
    {
        PyBuffer_Release(&img1_buffer);
        return NULL;
    }
    c_image1.img = img1_buffer.buf;
    c_image2.img = img2_buffer.buf;

    out_points = malloc(sizeof(float)*c_image1.width*c_image1.height);
    if(!correlation_correlate_images(&c_image1, &c_image2, angle, corridor_size,
        kernel_size, threshold, num_threads, out_points))
    {
        PyErr_SetString(CybervisionError, "Failed to correlate images");
        PyBuffer_Release(&img1_buffer);
        PyBuffer_Release(&img2_buffer);
        free(out_points);
        return NULL;
    }

    PyBuffer_Release(&img1_buffer);
    PyBuffer_Release(&img2_buffer);

    out = PyList_New(0);
    for (int y=0;y<c_image1.height;y++)
    {
        for (int x=0;x<c_image1.width;x++)
        {
            float depth = out_points[y*c_image1.width + x];
            if (!isfinite(depth))
                continue;
            PyObject *point = Py_BuildValue("(iif)", x, y, depth);
            PyList_Append(out, point);
        }
    }

    free(out_points);
    return out;
}

static PyMethodDef CybervisionMethods[] = {
    {"detect", cybervision_detect, METH_VARARGS, "Detect keypoints with FAST."},
    {"match", cybervision_match, METH_VARARGS, "Find correlation between image points."},
    {"correlate", cybervision_correlate, METH_VARARGS, "Find correlation between images."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef cybervisionmodule = {
    PyModuleDef_HEAD_INIT, "cybervision", NULL, -1, CybervisionMethods
};

PyMODINIT_FUNC
PyInit_cybervision(void)
{
    PyObject *m;
    m = PyModule_Create(&cybervisionmodule);
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
