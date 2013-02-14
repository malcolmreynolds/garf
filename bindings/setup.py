from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
# from Cython.Build import cythonize

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("mvn",
                           ["mvn.pyx"],
                           language="c++",
                           extra_compile_args=["-std=c++0x", "-stdlib=libc++"],
                           extra_link_args=["-stdlib=libc++"],
                           include_dirs=['/Users/malc/phd/proj/garf',
                                         '/usr/local/include/eigen3'])]
)
