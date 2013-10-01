from distutils.core import setup
from distutils.extension import Extension
import numpy as np
# from Cython.Distutils import build_ext
# from Cython.Build import cythonize

include_dirs = ['..',  # Should point to wherever the garf/ C++ dir is
                '/usr/local/include/eigen3',  # Location of eigen headers
                np.get_include()  # Numpy headers
                ]

library_dirs = ['/usr/local/lib']
libraries = ['boost_python-mt', 'boost_serialization-mt', 'tbb']

setup(
    name="GARF",
    version="0.1",
    description="Python bindings for GARF random forest library",
    author="Malcolm Reynolds",
    author_email="malcolm.reynolds@gmail.com",
    packages=["garf"],
    ext_modules=[
        Extension("_garf",
            ["garf.cpp"],
            language="c++",
            extra_compile_args=["-std=c++0x", "-stdlib=libc++", "-ferror-limit=3"],
            extra_link_args=["-stdlib=libc++"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries)]
)
