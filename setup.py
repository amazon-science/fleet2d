import os

from setuptools import setup

from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext, ParallelCompile, naive_recompile

# Optional multithreaded build
ParallelCompile("NPY_NUM_BUILD_JOBS", needs_recompile=naive_recompile).install()

ext_modules = [
    Pybind11Extension(
        "f2d.utils.fov_utils_cpp",
        sorted(glob("src/f2d/utils/*.cpp")),
        extra_compile_args=["-Wall", "-DPYBIND11_DETAILED_ERROR_MESSAGES"],
        extra_link_args=["-Wall"],
    ),
]

# Declare your non-python data files:
# Files underneath configuration/ will be copied into the build preserving the
# subdirectory structure if they exist.
data_files = []
for root, dirs, files in os.walk("configuration"):
    data_files.append((os.path.relpath(root, "configuration"), [os.path.join(root, f) for f in files]))

setup(
    # include data files
    data_files=data_files,
    entry_points="""\
                  [console_scripts]
                  Fleet2D = f2d.simulation.simulation:main
                  """,
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
