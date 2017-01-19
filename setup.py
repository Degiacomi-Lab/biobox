#call: python setup.py install


#graph classes for path detection
#cython -a graph.pyx
#gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.6 -o graph.so graph.c

#fast mathematical methods
#cython -a fastmath.pyx
#gcc -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -I/usr/include/python2.6 -o fastmath.so fastmath.c
#
#
#module = Extension('tools',
#                    define_macros = [('MAJOR_VERSION', '1'),
#                                     ('MINOR_VERSION', '0')],
#                    include_dirs = [np.get_include()],
#                    sources = ['fastmath.c', 'graph.c'],
#                    compiler_directives={'boundscheck': False,'wraparound': False},
#                    extra_compile_args=['-fno-strict-aliasing', '-O3','-fPIC','-shared', '-pthread', '-fwrapv', '-Wall'],
#                    )
#
#setup (name = 'Biobox_fast_tools',
#       description = 'Tools for speeding up operations in Biobox',
#       author = 'Matteo T. Degiacomi',
#       author_email = 'matteo.degiacomi@gmail.com',
#       ext_modules = [module])

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

        except Exception, ex:
            print "files already exist, skipping..."

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


#fout = open("__init__.py", "w")
#fout.write("from biobox.lib.fastmath import *\n")
#fout.write("from biobox.lib.graph import *\n")
#fout.write("from biobox.lib.interaction import *\n")#fout.close()

