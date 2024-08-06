from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.core import Extension


extensions = [
    Extension("felzenszwalb_3d_cy", ["felzenszwalb_3d_cy.pyx"], include_dirs=[numpy.get_include()])
]
setup(
    name='felzenszwalb_3d_cy',
    ext_modules=cythonize(extensions)
)

extensions = [
    Extension("_ccomp", ["_ccomp.pyx"], include_dirs=[numpy.get_include()])
]
setup(
    name='_ccomp',
    ext_modules=cythonize(extensions)
)