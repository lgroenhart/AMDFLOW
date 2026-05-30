# cython setup 
# use: 
#   python setup.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_transport = Extension(
    name = "transport",
    sources = ["transport.pyx"],
    include_dirs = [np.get_include()],
    extra_compile_args = ["/O2", "/fp:fast", "/openmp", "/wd4551"]
)


setup(
    name = "transport",
    ext_modules = cythonize(
        [ext_transport],
        compiler_directives = {"language_level": "3"},
        annotate = True
    ),
)

ext = Extension(
    name="chemistry",
    sources=["chemistry.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["/O2", "/openmp", "/fp:fast"],
    extra_link_args=["-fopenmp"],
)
 
setup(
    name="chemistry",
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "boundscheck":  False,
            "wraparound":   False,
            "cdivision":    True,
            "nonecheck":    False,
            "language_level": "3",
        },
        annotate=True,          
    ),
)
