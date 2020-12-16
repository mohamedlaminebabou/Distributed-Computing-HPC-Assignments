#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_18_API_VERSION
#include <stdlib.h>
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>

long solve_1d_nonlinearconv_pyccel(int n0_u, double *u, int n0_un, double *un, long nt, long nx, double dt, double dx);

/*........................................*/



/*........................................*/

/*........................................*/
PyObject *solve_1d_nonlinearconv_pyccel_wrapper(PyObject *self, PyObject *args, PyObject *kwargs)
{

    PyArrayObject *u;
    PyArrayObject *un;
    long nt;
    long nx;
    double dt;
    double dx;
    long Out_0001;
    PyObject *result;
    static char *kwlist[] = {
        "u",
        "un",
        "nt",
        "nx",
        "dt",
        "dx",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!lldd", kwlist, &PyArray_Type, &u, &PyArray_Type, &un, &nt, &nx, &dt, &dx))
    {
        return NULL;
    }
    if (PyArray_NDIM(u) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "u must have rank 1");
        return NULL;
    }
    if (PyArray_TYPE(u) != NPY_DOUBLE)
    {
        printf("%d %d\n", PyArray_TYPE(u), NPY_DOUBLE);
        PyErr_SetString(PyExc_TypeError, "u must be double");
        return NULL;
    }
    if (PyArray_NDIM(un) != 1)
    {
        PyErr_SetString(PyExc_TypeError, "un must have rank 1");
        return NULL;
    }
    if (PyArray_TYPE(un) != NPY_DOUBLE)
    {
        printf("%d %d\n", PyArray_TYPE(un), NPY_DOUBLE);
        PyErr_SetString(PyExc_TypeError, "un must be double");
        return NULL;
    }
    Out_0001 = solve_1d_nonlinearconv_pyccel(PyArray_DIM(u, 0), PyArray_DATA(u), PyArray_DIM(un, 0), PyArray_DATA(un), nt, nx, dt, dx);
    result = Py_BuildValue("l", Out_0001);
    return result;
}
/*........................................*/

static PyMethodDef mod_mr7sk2c6_methods[] = {
    {
        "solve_1d_nonlinearconv_pyccel",
        (PyCFunction)solve_1d_nonlinearconv_pyccel_wrapper,
        METH_VARARGS | METH_KEYWORDS,
        ""
    },
    { NULL, NULL, 0, NULL}
};

/*........................................*/

static struct PyModuleDef mod_mr7sk2c6_module = {
    PyModuleDef_HEAD_INIT,
    /* name of module */
    "mod_mr7sk2c6",
    /* module documentation, may be NULL */
    NULL,
    /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    -1,
    mod_mr7sk2c6_methods
};

/*........................................*/

PyMODINIT_FUNC PyInit_mod_mr7sk2c6(void)
{
    PyObject *m;

    import_array();

    m = PyModule_Create(&mod_mr7sk2c6_module);
    if (m == NULL) return NULL;

    return m;
}
