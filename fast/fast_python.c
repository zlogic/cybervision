#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "fast.h"

static PyObject *
fast_detect(PyObject *self, PyObject *args)
{
    Py_buffer img;
    int width, height, threshold, mode, nonmax;
    xy* corners = NULL;
    int num_corners;

    PyObject *out = NULL;

    // TODO: return exceptions on errors.
    if (!PyArg_ParseTuple(args, "y*iiiib", &img, &width, &height, &threshold, &mode, &nonmax))
        return NULL;
    if (nonmax && mode == 9)
        corners = fast9_detect_nonmax(img.buf, width, height, width, threshold, &num_corners);
    else if (nonmax && mode == 10)
        corners = fast10_detect_nonmax(img.buf, width, height, width, threshold, &num_corners);
    else if (nonmax && mode == 11)
        corners = fast11_detect_nonmax(img.buf, width, height, width, threshold, &num_corners);
    else if (nonmax && mode == 12)
        corners = fast12_detect_nonmax(img.buf, width, height, width, threshold, &num_corners);
    else if (!nonmax && mode == 9)
        corners = fast9_detect(img.buf, width, height, width, threshold, &num_corners);
    else if (!nonmax && mode == 10)
        corners = fast10_detect(img.buf, width, height, width, threshold, &num_corners);
    else if (!nonmax && mode == 11)
        corners = fast11_detect(img.buf, width, height, width, threshold, &num_corners);
    else if (!nonmax && mode == 12)
        corners = fast12_detect(img.buf, width, height, width, threshold, &num_corners);
    else
        return NULL;
    
    out = PyList_New(num_corners);
    for(int i=0;i<num_corners;i++) {
        PyObject *corner = Py_BuildValue("(ii)", corners[i].x, corners[i].y);
        PyList_SetItem(out, i, corner);
    }

    free(corners);
    return out;
}

static PyMethodDef FastMethods[] = {
    {"detect", fast_detect, METH_VARARGS, "Execute a shell command."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef fastmodule = {
    PyModuleDef_HEAD_INIT, "fast", NULL, -1, FastMethods
};

PyMODINIT_FUNC
PyInit_fast(void)
{
    return PyModule_Create(&fastmodule);
}
