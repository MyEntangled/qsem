from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


CPP_FLAGS = ["-O3", "-std=c++17", "-march=native", "-mtune=native", "-funroll-loops"]
OPENMP_FLAGS = ["-fopenmp"]
LAPACK_FLAGS = ["-llapack", "-lblas"]


ext_modules = [
    Pybind11Extension(
        "CG_Solver",
        ["src/cg.cpp"],
        extra_compile_args=CPP_FLAGS + OPENMP_FLAGS,
        extra_link_args=OPENMP_FLAGS,
    ),
    Pybind11Extension(
        "Solve_Lanczos",
        ["src/lanczos.cpp"],
        extra_compile_args=CPP_FLAGS + OPENMP_FLAGS,
        extra_link_args=OPENMP_FLAGS + LAPACK_FLAGS,
    )
]


setup(
    name="qsem-cg-lanczos-solver",
    version="0.1.0",
    description="Pybind11 Conjugate Gradient solver and Lanczos extension",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)