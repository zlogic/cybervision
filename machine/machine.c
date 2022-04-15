#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>

#include "fast/fast.h"

static PyObject *MachineError;

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
        PyErr_Format(MachineError, "Failed to parse attribute %s", attr_name);
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
        PyErr_SetString(MachineError, "Failed to convert image");
        return 0;
    }
    if (PyObject_GetBuffer(img_bytes, buffer, PyBUF_ANY_CONTIGUOUS) != 0)
    {
        PyErr_SetString(MachineError, "Failed to get image buffer");
        Py_DECREF(img_bytes);
        return 0;
    }
    Py_DECREF(img_bytes);
    return 1;
}

#define SQR(x) ((x)*(x))

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
        PyErr_SetString(MachineError, "Failed to parse args");
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
        PyErr_Format(MachineError, "Unsupported FAST options nonmax=%i mode=%i", nonmax, mode);
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
machine_match(PyObject *self, PyObject *args)
{
    PyObject *img1, *img2;
    Py_buffer img1_buffer, img2_buffer;
    char *img1_bytes = NULL, *img2_bytes = NULL;
    int kernel_size, kernel_width, kernel_point_count;
    float threshold;
    int w1, h1, w2, h2;
    PyObject *points1, *points2;
    Py_ssize_t points1_size, points2_size;

    float *delta1 = NULL, *delta2 = NULL;
    float *sigma1 = NULL, *sigma2 = NULL;

    PyObject *out = NULL;

    if (!PyArg_ParseTuple(args, "OOOOif", &img1, &img2, &points1, &points2, &kernel_size, &threshold))
    {
        PyErr_SetString(MachineError, "Failed to parse args");
        return NULL;
    }

    kernel_width = 2*kernel_size + 1;
    kernel_point_count = SQR(kernel_width);

    w1 = read_long_attr(img1, "width");
    if (w1 == -1)
        return NULL;
    h1 = read_long_attr(img1, "height");
    if (h1 == -1)
        return NULL;
    w2 = read_long_attr(img2, "width");
    if (w2 == -1)
        return NULL;
    h2 = read_long_attr(img2, "height");
    if (h2 == -1)
        return NULL;

    if (!read_img_bytes(img1, &img1_buffer))
        return NULL;
    if (!read_img_bytes(img2, &img2_buffer)){
        PyBuffer_Release(&img1_buffer);
        return NULL;
    }

    img1_bytes = (char*)img1_buffer.buf;
    img2_bytes = (char*)img2_buffer.buf;

    out = PyList_New(0);

    points1_size = PyList_Size(points1);
    points2_size = PyList_Size(points2);

    delta1 = malloc(sizeof(float)*kernel_point_count*points1_size);
    delta2 = malloc(sizeof(float)*kernel_point_count*points2_size);
    sigma1 = malloc(sizeof(float)*points1_size);
    sigma2 = malloc(sizeof(float)*points2_size);

    for (Py_ssize_t p=0;p<points1_size;p++)
    {
        PyObject *point;
        int x, y;
        float avg = 0, sigma = 0;

        sigma1[p] = INFINITY;
        point = PyList_GetItem(points1, p);
        if (!PyArg_ParseTuple(point, "ii", &x, &y))
            continue;
        if (x-kernel_size<0 || x+kernel_size>=w1 || y-kernel_size<0 || y+kernel_size>=h1)
            continue;

        for (int i=-kernel_size;i<=kernel_size;i++)
        {
            for(int j=-kernel_size;j<=kernel_size;j++)
            {
                float value;
                value = (float)img1_bytes[(y+j)*w1 + (x+i)];
                avg += value;
                delta1[p*kernel_point_count + (j+kernel_size)*kernel_width + (i+kernel_size)] = value;
            }
        }
        avg /= (float)kernel_point_count;
        for (int i=0;i<kernel_point_count;i++)
        {
            delta1[p*kernel_point_count + i] -= avg;
            sigma += SQR(delta1[p*kernel_point_count + i]);
        }
        sigma1[p] = sqrt(sigma/(float)kernel_point_count);
    }
    PyBuffer_Release(&img1_buffer);

    for (Py_ssize_t p=0;p<points2_size;p++)
    {
        PyObject *point;
        int x, y;
        float avg = 0, sigma = 0;

        sigma2[p] = INFINITY;

        point = PyList_GetItem(points2, p);
        if (!PyArg_ParseTuple(point, "ii", &x, &y))
            continue;
        if (x-kernel_size<0 || x+kernel_size>=w2 || y-kernel_size<0 || y+kernel_size>=h2)
            continue;

        for (int i=-kernel_size;i<=kernel_size;i++)
        {
            for(int j=-kernel_size;j<=kernel_size;j++)
            {
                float value;
                value = (float)img2_bytes[(y+j)*w2 + (x+i)];
                avg += value;
                delta2[p*kernel_point_count + (j+kernel_size)*kernel_width + (i+kernel_size)] = value;
            }
        }
        avg /= (float)kernel_point_count;
        for (int i=0;i<kernel_point_count;i++)
        {
            delta2[p*kernel_point_count + i] -= avg;
            sigma += SQR(delta2[p*kernel_point_count + i]);
        }
        sigma2[p] = sqrt(sigma/(float)kernel_point_count);
    }
    PyBuffer_Release(&img2_buffer);

    for (Py_ssize_t p1=0;p1<points1_size;p1++)
    {
        if (!isfinite(sigma1[p1]))
            continue;

        for (Py_ssize_t p2=0;p2<points2_size;p2++)
        {
            float corr = 0;

            if (!isfinite(sigma2[p1]))
                continue;

            for (int i=0;i<kernel_point_count;i++)
                corr += delta1[p1*kernel_point_count + i] * delta2[p2*kernel_point_count + i];
            corr = corr/(sigma1[p1]*sigma2[p2]*(float)kernel_point_count);

            if (corr >= threshold || -corr <= -threshold)
            {
                PyObject *correlation_value = Py_BuildValue("(iif)", p1, p2, corr);
                {
                    PyList_Append(out, correlation_value);
                }
                Py_DECREF(correlation_value);
            }
        }
    }

    free(delta1);
    free(delta2);
    free(sigma1);
    free(sigma2);

    return out;
}

static PyMethodDef MachineMethods[] = {
    {"detect", machine_detect, METH_VARARGS, "Detect keypoints with FAST."},
    {"match", machine_match, METH_VARARGS, "Find correlation between image points."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef machinemodule = {
    PyModuleDef_HEAD_INIT, "machine", NULL, -1, MachineMethods
};

PyMODINIT_FUNC
PyInit_machine(void)
{
    PyObject *m;
    m = PyModule_Create(&machinemodule);
    if (m == NULL)
        return m;

    MachineError = PyErr_NewException("machine.error", NULL, NULL);
    Py_XINCREF(MachineError);
    if (PyModule_AddObject(m, "error", MachineError) < 0) {
        Py_XDECREF(MachineError);
        Py_CLEAR(MachineError);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
