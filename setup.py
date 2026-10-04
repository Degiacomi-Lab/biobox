# Package metadata and configuration are in pyproject.toml. This file only declares the
# Cython extensions in src/biobox/lib, compiled when the package is built, e.g. with:
# pip install .
# or, to compile them next to the sources for development:
# python setup.py build_ext --inplace

import glob
import os

import numpy as np
from Cython.Build import cythonize
from setuptools import setup

pyx_files = sorted(glob.glob(os.path.join("src", "biobox", "lib", "*.pyx")))

setup(
    ext_modules=cythonize(
        pyx_files,
        include_path=[np.get_include()],
        compiler_directives={"boundscheck": False, "wraparound": False}),
    include_dirs=[np.get_include()],
)
