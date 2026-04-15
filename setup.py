# cython setup 
# use: 
#   python setup.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name = "amd_chemistry",
    sources = ["amd_chemistry.pyx"],
    include_dirs = [np.get_include()],
    extra_compile_args = ["/O2", "/arch:AVX2", "/fp:fast"]
)

ext_transport = Extension(
    name = "transport",
    sources = ["transport.pyx"],
    include_dirs = [np.get_include()],
    extra_compile_args = ["/O2", "/arch:AVX2", "/fp:fast", "/wd4244", "/wd4551", "/wd4700"]
)

setup(
    name = "amd_chemistry",
    ext_modules = cythonize(
        [ext],
        compiler_directives = {
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
        annotate = True,
    ),
)

setup(
    name = "transport",
    ext_modules = cythonize(
        [ext_transport],
        compiler_directives = {"language_level": "3"},
        annotate = True
    ),
)
