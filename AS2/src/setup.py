from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
      name = "RGB to Gray Scale conversion",
      ext_modules= cythonize('test.pyx'),
      include_dirs=[numpy.get_include()])