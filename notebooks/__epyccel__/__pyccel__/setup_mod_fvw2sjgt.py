from setuptools import Extension, setup
import numpy

extension_mod = Extension("mod_fvw2sjgt",
		[ r'mod_fvw2sjgt_wrapper.c' ],
		extra_objects = [r'/home/babou/hpc/Distributed-Computing-HPC-Assignments/notebooks/__epyccel__/__pyccel__/bind_c_mod_fvw2sjgt.o',
				r'/home/babou/hpc/Distributed-Computing-HPC-Assignments/notebooks/__epyccel__/__pyccel__/mod_fvw2sjgt.o'],
		include_dirs = [r'/home/babou/hpc/Distributed-Computing-HPC-Assignments/notebooks/__epyccel__/__pyccel__', numpy.get_include()],
		libraries = [r'gfortran'],
		library_dirs = [r'/usr/lib/gcc/x86_64-linux-gnu/9'],
		extra_link_args = [r'-O3',
				r'-fPIC',
				r'-I"/home/babou/hpc/Distributed-Computing-HPC-Assignments/notebooks/__epyccel__/__pyccel__"'])

setup(name = "mod_fvw2sjgt", ext_modules=[extension_mod])