libxm
=====

Libxm is a library that provides routines for efficient contractions of very
large (up to many terabytes) disk-based block-tensors on multi-core CPUs, GPUs,
and various floating point accelerators.

With libxm tensors can be stored on hard disks which allow virtually unlimited
data size. Data are asynchronously prefetched to main memory for fast access.
Tensor contractions are reformulated as multiplications of big matrices done in
batches. Tensor symmetry and sparsity is used to decrease storage and
computational requirements. Computations can be efficiently accelerated using
multiple GPUs or other accelerators like Intel Xeon Phi. Libxm reaches close to
peak floating-point performance even in cases when data size is much larger
than the available amount of fast random access memory. For very large problems
libxm shows considerable speedups (10x or more) compared to similar tensor
contraction codes. Libxm supports single and double precision scalar and
complex numbers.


Usage
-----

Once tensors are setup the contraction routine is similar to BLAS dgemm call:

   xm_contract(alpha, A, B, beta, C, "abcd", "ijcd", "ijab");

This will preform the following contraction of two 4-index tensors A and B:

   C_ijab := alpha * A_abcd * B_ijcd + beta * C_ijab


Compilation
-----------

To compile libxm you need a POSIX environment, an efficient BLAS library, and
an ANSI C complaint compiler. To use libxm in your project include xm.h file
and compile the code:

   cc myprog.c xm.c alloc.c -lblas -lpthread -lm

Replace "-lblas" with appropriate accelerated libraries (e.g. "-lnvblas") to
get all benefits of corresponding hardware. To compile benchmarks and tests use
provided Makefile. Detailed API documentation can be found in xm.h file.


Source code overview
--------------------

   xm.h - public API header with documentation
   xm.c - main libxm implementation file
   alloc.c/alloc.h - disk-backed allocator for large tensors
   aux.c/aux.h - optional auxiliary functions for tensor creation
   benchmark.c - sample benchmarks
   test.c - facilities for randomized testing


Who uses libxm
--------------

 - libxm is integrated with Q-Chem package to accelerate high-level quantum
   chemistry calculations

 - libxm is used by a C++ tensor library libtensor
