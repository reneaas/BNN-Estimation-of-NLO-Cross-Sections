#Compile with "python3 setup.py build_ext --inplace"

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "py_layer",
        sources=["py_layer.pyx"],
        extra_compile_args=["-Ofast", "-std=c++11", "-mtune=native"],
        include_dirs=[numpy.get_include()],
        language="c++"
    ),
    Extension(
        "py_bnn",
        sources=["py_bnn.pyx"],
        extra_compile_args=["-Ofast", "-std=c++11", "-mtune=native"],
        include_dirs=[numpy.get_include()],
        language="c++"
    )
]

setup(name="BNN",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level' : "3"}), 
)