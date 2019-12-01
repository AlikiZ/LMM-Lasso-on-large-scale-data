from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules = cythonize("/home/aliki/Documents/hpi/llmlasso/AdaScreen/adascreen/enet_solver.pyx"))