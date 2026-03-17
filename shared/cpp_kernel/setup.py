"""Build script for commodities C++ kernel."""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os


class BuildExt(build_ext):
    """Custom build extension for C++17 support."""

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = []
        if ct == "unix":
            opts.append("-std=c++17")
            opts.append("-O0")
            opts.append("-g")
            opts.append("-fPIC")
        elif ct == "msvc":
            opts.append("/std:c++17")
            opts.append("/Od")
            opts.append("/Zi")

        for ext in self.extensions:
            ext.extra_compile_args = opts
            if ct == "msvc":
                ext.extra_link_args = ["/DEBUG"]
            elif ct == "unix":
                ext.extra_link_args = ["-g"]
        build_ext.build_extensions(self)


# Find pybind11 include path
import pybind11
pybind11_include = pybind11.get_include()

ext_modules = [
    Extension(
        "commodities_cpp",
        sources=["bindings/pybind_module.cpp"],
        include_dirs=[
            "include",
            pybind11_include,
        ],
        language="c++",
    ),
]

setup(
    name="commodities_cpp",
    version="0.1.0",
    description="C++ kernel for energy commodity trading platform",
    packages=[],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
