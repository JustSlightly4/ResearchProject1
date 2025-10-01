from setuptools import setup, Extension
import pybind11
import os
import sys
import platform

#python BuildCPP_Script.py build_ext --inplace

eigen_include_dir = os.path.join(os.path.dirname(__file__), "eigen-3.3.9")

# Set C++ standard flag depending on OS
extra_compile_args = []
if platform.system() == "Windows":
    extra_compile_args.append("/std:c++17")
else:  # Linux / macOS
    extra_compile_args.append("-std=c++17")
    extra_compile_args.append("-O3")           # optional optimization
    extra_compile_args.append("-march=native") # optional CPU optimization

ext_modules = [
    Extension(
        "voxelgridC",
        ["voxelgridC.cpp"],
        include_dirs=[
            pybind11.get_include(),
            eigen_include_dir
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name="voxelgridC",
    version="0.1",
    ext_modules=ext_modules,
)
