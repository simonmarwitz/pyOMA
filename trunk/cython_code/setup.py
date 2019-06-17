from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    #ext_modules = cythonize("cython_code/cython_helpers.pyx")
    ext_modules = cythonize("hello_world.pyx"),
    include_dirs=[numpy.get_include()]
)