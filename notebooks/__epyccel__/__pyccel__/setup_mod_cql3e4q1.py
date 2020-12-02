from setuptools import Extension, setup
import numpy

extension_mod = Extension("mod_cql3e4q1",
		[ r'mod_cql3e4q1_wrapper.c' ],
		extra_objects = [r'/home/babou/hpc/Distributed-Computing-HPC-Assignments/notebooks/__epyccel__/__pyccel__/bind_c_mod_cql3e4q1.o',
				r'/home/babou/hpc/Distributed-Computing-HPC-Assignments/notebooks/__epyccel__/__pyccel__/mod_cql3e4q1.o'],
		include_dirs = [r'/home/babou/hpc/Distributed-Computing-HPC-Assignments/notebooks/__epyccel__/__pyccel__', numpy.get_include()],
		libraries = [r'gomp',
				r'gfortran'],
		library_dirs = [r'/usr/lib/gcc/x86_64-linux-gnu/9'],
		extra_link_args = [r'-O3',
				r'-fopenmp',
				r'-fPIC',
				r'-fopenmp',
				r'-I"/home/babou/hpc/Distributed-Computing-HPC-Assignments/notebooks/__epyccel__/__pyccel__"'])

setup(name = "mod_cql3e4q1", ext_modules=[extension_mod])