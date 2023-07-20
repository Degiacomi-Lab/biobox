# to compile cython sources, call:
# python setup.py install
#
# Compilation can otherwise be obtained via the following commands:
# cython -a graph.pyx
# gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.6 -o graph.so graph.c
# cython -a fastmath.pyx
# gcc -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -I/usr/include/python2.6 -o fastmath.so fastmath.c
#
# to build documentation:
# sphinx-build -b html doc doc/html

import os
import shutil
import numpy as np
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize


packages=find_packages(where='src')
print('packages being: ', packages)
pyx_files = []
for package in packages:
    package_path = f'src{os.sep}{os.sep.join(package.split("."))}'
    for file in os.listdir(package_path):
        if file.split('.')[-1]=='pyx':
            pyx_files.append(f'{package_path}{os.sep}{file}')
setup(
    name = 'biobox',
    version='1.1.2',
    include_dirs=[np.get_include()],
    ext_modules=cythonize(
        pyx_files,#"*.pyx",
        include_path=[np.get_include()],
        compiler_directives={'boundscheck': False, 'wraparound': False}),
    package_data={"":["*.dat"]},
    packages=packages,
    package_dir={"":"src"},
)
