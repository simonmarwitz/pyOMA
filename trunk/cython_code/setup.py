from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("cython_code/cython_helpers.pyx")
)