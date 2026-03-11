"""Debug build script - builds with debug symbols and no optimization."""

from setup import BuildExt, ext_modules
from setuptools import setup

for ext in ext_modules:
    ext.extra_compile_args = []
    ext.extra_link_args = []

setup(
    name="commodities_cpp_debug",
    version="0.1.0-debug",
    description="C++ kernel (debug build)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
