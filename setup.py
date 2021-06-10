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
from distutils.core import setup
from distutils.command.build_ext import build_ext
from Cython.Build import cythonize


class InstallCommand(build_ext):

    def run(self):

        build_ext.run(self)

        try:
            for root, dirnames, filenames in os.walk("build"):
                for filename in filenames:
                    extension = filename.split(".")[-1]
                    if extension in ["pyd", "so"]:
                        os.rename(os.path.join(root, filename), filename)

        except Exception as ex:
            print("files already exist, skipping...")

        shutil.rmtree("build")

os.chdir("lib")

# small hack to get around a problem in older cython versions, i.e.
# an infinite dependencies loop when __init__.py file is in the same folder as pyx
if os.path.exists("__init__.py"):
    os.rename("__init__.py", "tmp")


setup(
    include_dirs=[np.get_include()],
    ext_modules=cythonize(
        "*.pyx",
        include_path=[np.get_include()],
            compiler_directives={'boundscheck': False, 'wraparound': False}),
    cmdclass={'install': InstallCommand}
)

# continuation of the small hack
os.rename("tmp", "__init__.py")
