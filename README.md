# CG and Lanczos Solvers

Contains the source files for lanczos and CG solvers. 

## Installation

For installation run the following commands. Note that the build script is developed with the `GCC` compiler for `C++` code. It should be compatible with clang compilers, but is untested.

Start with a conda environment to build the modules.

```
$ conda env create -f environment.yml
```

```
$ conda activate QSEM
```

Build and install the module.

```
$ python3 setup.py build_ext --inplace
```

## Guide

For usage guide, refer to the `test_cg` notebook on using the CG Solver and `test_lanczos` notebook on the Lanczos Solver.