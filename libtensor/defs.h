#ifndef LIBTENSOR_DEFS_H
#define LIBTENSOR_DEFS_H

#include <cstddef>

/** \brief Tensor library
    \ingroup libtensor
 **/
namespace libtensor {

/** \brief Namespace name
 **/
extern const char *g_ns;

}

#ifdef __MINGW32__
#include <cstdlib>
inline void srand48(long seed) { srand(seed); }
inline double drand48() { return (double(rand())/RAND_MAX); }
inline long lrand48() { return rand(); }
#endif

/** \mainpage Tensor Library

    The tensor library is an open-source object-oriented C++ library to
    perform tensor algebra on tensors of arbitrary rank, size and symmetry.
    The provided classes and functions operate on large tensors by splitting
    them into small blocks which can be stored in memory or on disk.
    Parallel divide-and-conquer algorithms are available to perform the tensor
    algebra operations.

    \section Contents
    -# \subpage compile
    -# \subpage prog_guide

 **/

/** \page compile How to compile

    ... coming soon ...
 **/

/** \page prog_guide Programmer's guide

    \section Directory Structure

    - \c cmake/ -- Files related to the build system \c cmake.
    - \c libtensor/ -- The actual tensor library
            (for details \subpage libtensor_guide).
    - \c performance_tests/ -- Set of classes to measure the performance (outdated).
    - \c tests -- Unit tests for the classes and functions in the tensor library.

    ... to be continued ...
 **/

/** \page libtensor_guide Tensor library
    \section Code Structure

    The source code of the tensor library consists of various parts structured
    into a number of subdirectories.
    -# \ref libtensor_core in directory \c core/
    -# \ref libtensor_symmetry in directory \c symmetry/
    -# \ref libtensor_dense_tensor in directory \c dense_tensor/
    -# \ref libtensor_block_tensor in directory \c block_tensor/
    -# \ref libtensor_iface in directory \c iface/
    -# \ref libtensor_gen_block_tensor in directory \c gen_block_tensor/
    -# \ref libtensor_diag_tensor in directory \c diag_tensor/
    -# \ref libtensor_kernels in directory \c kernels/
    -# \ref libtensor_linalg in directory \c linalg/

    \section wheretostart Where to Start

    ... coming soon ...
 **/

/** \defgroup libtensor Tensor library

 **/

/** \defgroup libtensor_core Core components
    \ingroup libtensor
 **/

/** \defgroup libtensor_dense_tensor Dense tensors
    \brief Implementation of dense tensors of arbitrary element type

    \ingroup libtensor
 **/

/** \defgroup libtensor_dense_tensor_tod Tensor operations on dense tensors (double)
    \brief Operations on tensors with real double precision elements
    \ingroup libtensor_dense_tensor
 **/

/** \defgroup libtensor_diag_tensor Generalized diagonal tensors
    \brief Implementation of "diagonal" tensors

    \ingroup libtensor
 **/

/** \defgroup libtensor_gen_block_tensor Generalized block tensors
    \brief Implementation of block tensors with arbitrary types

    \ingroup libtensor
 **/

/** \defgroup libtensor_gen_bto Generalized block %tensor operations
    \brief Implementation of block tensor operations with arbitrary types

    \ingroup libtensor_gen_block_tensor
 **/

/** \defgroup libtensor_block_tensor Block tensors of dense tensors
    \brief Implementation of block tensors with dense tensors, but arbitrary
        element type
    \ingroup libtensor
 **/

/** \defgroup libtensor_block_tensor_btod Block tensor operations (double)
    \brief Operations on block tensors with real double precision elements
    \ingroup libtensor_block_tensor
 **/

/** \defgroup libtensor_iface Block tensor interface
    \brief Easy to use interface to implement equations with block tensors.
    \ingroup libtensor
 **/

/** \defgroup libtensor_symmetry Components for tensor symmetry
    \brief Easy to use interface to implement equations with block tensors.
    \ingroup libtensor
 **/

#endif // LIBTENSOR_DEFS_H

