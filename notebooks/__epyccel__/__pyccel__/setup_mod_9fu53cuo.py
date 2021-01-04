from setuptools import Extension, setup
import numpy

extension_mod = Extension("mod_9fu53cuo",
		[ r'mod_9fu53cuo_wrapper.c' ],
		extra_objects = [r'/home/babou/Desktop/hpc/Distributed-Computing-HPC-Assignments/notebooks/__epyccel__/__pyccel__/bind_c_mod_9fu53cuo.o',
				r'/home/babou/Desktop/hpc/Distributed-Computing-HPC-Assignments/notebooks/__epyccel__/__pyccel__/mod_9fu53cuo.o'],
		include_dirs = [r'/home/babou/Desktop/hpc/Distributed-Computing-HPC-Assignments/notebooks/__epyccel__/__pyccel__', numpy.get_include()],
		libraries = [r'gfortran'],
		library_dirs = [r'/usr/lib/gcc/x86_64-linux-gnu/9'],
		extra_link_args = [r'-O3',
				r'-fPIC',
				r'-I"/home/babou/Desktop/hpc/Distributed-Computing-HPC-Assignments/notebooks/__epyccel__/__pyccel__"'])

setup(name = "mod_9fu53cuo", ext_modules=[extension_mod])