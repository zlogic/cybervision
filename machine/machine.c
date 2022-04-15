#define PY_SSIZE_T_CLEAN
#include <Python.h>

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
        PyErr_SetString(MachineError, "Failed to parse argument");
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
        PyErr_SetString(MachineError, "Unsupported FAST options");
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

static PyMethodDef MachineMethods[] = {
    {"detect", machine_detect, METH_VARARGS, "Detect keypoints with FAST."},
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
