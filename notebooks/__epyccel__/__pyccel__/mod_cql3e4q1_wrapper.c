#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <stdlib.h>
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>

long dot(int n0_a, int n1_a, double *a, int n0_b, int n1_b, double *b, int n0_c, int n1_c, double *c);

/*........................................*/



/*........................................*/

/*........................................*/
PyObject *dot_wrapper(PyObject *self, PyObject *args, PyObject *kwargs)
{

    PyArrayObject *a;
    PyArrayObject *b;
    PyArrayObject *c;
    long Out_0001;
    PyObject *result;
    static char *kwlist[] = {
        "a",
        "b",
        "c",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!", kwlist, &PyArray_Type, &a, &PyArray_Type, &b, &PyArray_Type, &c))
    {
        return NULL;
    }
    if (PyArray_NDIM(a) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "a must have rank 2");
        return NULL;
    }
    if (PyArray_TYPE(a) != NPY_DOUBLE)
    {
        printf("%d %d\n", PyArray_TYPE(a), NPY_DOUBLE);
        PyErr_SetString(PyExc_TypeError, "a must be double");
        return NULL;
    }
    if (!PyArray_CHKFLAGS(a, NPY_ARRAY_C_CONTIGUOUS))
    {
        PyErr_SetString(PyExc_NotImplementedError, "Argument does not have the expected ordering (C)");
        return NULL;
    }
    if (PyArray_NDIM(b) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "b must have rank 2");
        return NULL;
    }
    if (PyArray_TYPE(b) != NPY_DOUBLE)
    {
        printf("%d %d\n", PyArray_TYPE(b), NPY_DOUBLE);
        PyErr_SetString(PyExc_TypeError, "b must be double");
        return NULL;
    }
    if (!PyArray_CHKFLAGS(b, NPY_ARRAY_F_CONTIGUOUS))
    {
        PyErr_SetString(PyExc_NotImplementedError, "Argument does not have the expected ordering (F)");
        return NULL;
    }
    if (PyArray_NDIM(c) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "c must have rank 2");
        return NULL;
    }
    if (PyArray_TYPE(c) != NPY_DOUBLE)
    {
        printf("%d %d\n", PyArray_TYPE(c), NPY_DOUBLE);
        PyErr_SetString(PyExc_TypeError, "c must be double");
        return NULL;
    }
    if (!PyArray_CHKFLAGS(c, NPY_ARRAY_C_CONTIGUOUS))
    {
        PyErr_SetString(PyExc_NotImplementedError, "Argument does not have the expected ordering (C)");
        return NULL;
    }
    Out_0001 = dot(PyArray_DIM(a, 0), PyArray_DIM(a, 1), PyArray_DATA(a), PyArray_DIM(b, 0), PyArray_DIM(b, 1), PyArray_DATA(b), PyArray_DIM(c, 0), PyArray_DIM(c, 1), PyArray_DATA(c));
    result = Py_BuildValue("l", Out_0001);
    return result;
}
/*........................................*/

static PyMethodDef mod_cql3e4q1_methods[] = {
    {
        "dot",
        (PyCFunction)dot_wrapper,
        METH_VARARGS | METH_KEYWORDS,
        ""
    },
    { NULL, NULL, 0, NULL}
};

/*........................................*/

static struct PyModuleDef mod_cql3e4q1_module = {
    PyModuleDef_HEAD_INIT,
    /* name of module */
    "mod_cql3e4q1",
    /* module documentation, may be NULL */
    NULL,
    /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    -1,
    mod_cql3e4q1_methods
};

/*........................................*/

PyMODINIT_FUNC PyInit_mod_cql3e4q1(void)
{
    PyObject *m;

    import_array();

    m = PyModule_Create(&mod_cql3e4q1_module);
    if (m == NULL) return NULL;

    return m;
}
