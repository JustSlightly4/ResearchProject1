from setuptools import setup, Extension
import pybind11
import os

eigen_include_dir = os.path.join(os.path.dirname(__file__), "eigen-3.3.9")

ext_modules = [
    Extension(
        "voxelgrid",
        ["voxelgrid.cpp"],
        include_dirs=[
            pybind11.get_include(),
            eigen_include_dir
        ],
        language="c++",
    )
]

setup(
    name="voxelgrid",
    version="0.1",
    ext_modules=ext_modules,
)
