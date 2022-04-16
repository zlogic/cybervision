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

void context_destroy(PyObject *obj)
{
    context *ctx = PyCapsule_GetPointer(obj, NULL);
    ctx_free(ctx);
    free(ctx);
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
cybervision_ctx_prepare(PyObject *self, PyObject *args)
{
    PyObject *img;
    Py_buffer img_buffer;
    int kernel_size;
    int width, height;
    int num_threads;
    context *ctx;
    
    if (!PyArg_ParseTuple(args, "Oii", &img, &kernel_size, &num_threads))
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

    ctx = malloc(sizeof(context));

    if(!ctx_init(ctx, img_buffer.buf, width, height, kernel_size, num_threads))
    {
        PyBuffer_Release(&img_buffer);
        PyErr_SetString(CybervisionError, "Failed to initialize context");
        return NULL;
    }
    PyBuffer_Release(&img_buffer);

    return PyCapsule_New(ctx, NULL, context_destroy);
}

static PyObject *
cybervision_match(PyObject *self, PyObject *args)
{
    float threshold;
    int num_threads;
    PyObject *ctx1_obj, *ctx2_obj;
    PyObject *points1, *points2;
    correlation_point *c_points1, *c_points2;
    context *ctx1, *ctx2;

    PyObject *out = NULL;

    if (!PyArg_ParseTuple(args, "OOOOfi", &ctx1_obj, &ctx2_obj, &points1, &points2, &threshold, &num_threads))
    {
        PyErr_SetString(CybervisionError, "Failed to parse args");
        return NULL;
    }
    ctx1 = PyCapsule_GetPointer(ctx1_obj, NULL);
    if (ctx1 == NULL)
    {
        PyErr_SetString(CybervisionError, "Failed to get ctx1 pointer");
        return NULL;
    }

    ctx2 = PyCapsule_GetPointer(ctx2_obj, NULL);
    if (ctx2 == NULL)
    {
        PyErr_SetString(CybervisionError, "Failed to get ctx2 pointer");
        return NULL;
    }

    c_points1 = read_points(points1);
    if (c_points1 == NULL)
        return NULL;

    c_points2 = read_points(points2);
    if (c_points2 == NULL)
    {
        free(c_points1);
        return NULL;
    }

    out = PyList_New(0);

    if(!ctx_correlate(ctx1, ctx2, c_points1, c_points2, PyList_Size(points1), PyList_Size(points2), threshold, num_threads, add_match, out))
    {
        PyErr_SetString(CybervisionError, "Failed to correlate points");
        Py_DECREF(out);
        return NULL;
    }

    return out;
}

static PyMethodDef CybervisionMethods[] = {
    {"detect", cybervision_detect, METH_VARARGS, "Detect keypoints with FAST."},
    {"ctx_prepare", cybervision_ctx_prepare, METH_VARARGS, "Prepare correlation context."},
    {"match", cybervision_match, METH_VARARGS, "Find correlation between image points."},
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
