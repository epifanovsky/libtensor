# libxm 2.0

Libxm is a distributed-parallel C/C++ library that provides routines for
efficient operations (e.g., contractions) on very large (terabytes in size)
disk-backed block-tensors.

With libxm tensors can be stored on hard disks which allows for virtually
unlimited data size. Data are asynchronously prefetched to main memory for fast
access. Tensor contractions are reformulated as multiplications of matrices
done in batches using optimized BLAS routines. Tensor block-level symmetry and
sparsity is used to decrease storage and computational requirements. Libxm
supports single and double precision scalar and complex numbers.

### Compilation

You will need a POSIX-compatible `make` utility, an efficient BLAS library (for
linking), and an ANSI C complaint compiler. To compile libxm, issue:

    cd src && make

To change the default compiler and enable OpenMP and MPI, the following command
can be used:

    cd src && CC=mpicc CFLAGS="-O3 -fopenmp -DXM_USE_MPI" make

To use libxm in your project, include `xm.h` file and link with the
compiled static library `libxm.a`.

### Documentation

Step-by-step tutorial on how to use libxm can be found in the
[example.c](example.c) file.
Full API documentation is available
[here](https://ilyak.github.io/libxm/doxygen_html/files.html).

### Parallel scaling

The table below shows parallel scalability of some libxm operations on the
NERSC Cori Cray XC40 supercomputer with BurstBuffer enabled. Total tensor data
size is over 2 Tb in all tests. Table shows time in seconds with speedup
relative to 1 node shown in parenthesis.

|      Nodes      |  xm\_contract  |   xm\_add   |   xm\_set   |
|:---------------:|:--------------:|:-----------:|:-----------:|
|  1 (32 cores)   |  23660 (1.0x)  | 787 (1.0x)  | 457 (1.0x)  |
|  2 (64 cores)   |  11771 (2.0x)  | 436 (1.8x)  | 324 (1.4x)  |
|  4 (128 cores)  |   5938 (4.0x)  | 203 (3.9x)  | 115 (4.0x)  |
|  8 (256 cores)  |   3167 (7.5x)  | 168 (4.7x)  |  66 (6.9x)  |
| 16 (512 cores)  |   1606 (14.7x) |  69 (11.4x) |  28 (16.3x) |
| 32 (1024 cores) |    836 (28.3x) |  32 (24.6x) |  21 (21.8x) |

### Journal reference

Please reference the following paper when using libxm:

[I.A. Kaliman and A.I. Krylov, JCC 2017](https://dx.doi.org/10.1002/jcc.24713)

### Libxm users

- libxm is integrated with the [Q-Chem](http://www.q-chem.com) quantum
  chemistry package to accelerate large electronic structure calculations
