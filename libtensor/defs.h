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
    -# \subpage reference

 **/

/** \page compile How to compile

    \section Requirements

    To compile libtensor the following programs and libraries are required:
    - \c cmake -- libtensor uses cmake to configure and set up the build
    - \c lapack and \c blas libraries (e.g. MKL, ACML, Atlas, CBLAS, GSL)
    - \c libtest -- Test library (to compile the unit tests for \c libtensor)
    - \c libutil -- Utilities library (machine-dependent code)
    - [optional] \c libvmm -- Virtual memory management library

    The libraries \c libtest, \c libutil, and \c libvmm have to be located in
    the same directory as \c libtensor for the configuration of the build to
    work seemlessly. All libraries have to be configured and compiled before
    \c libtensor can be build. The configuration and build of the libraries is
    similar to that of \c libtensor (see below). It is necessary to compile
    the libraries in the same order as they are listed above.

    \section The Configuration

    The simplest way to configure the build of \c libtensor (and the other
    libraries) is to use the shell script \c configure in the main directory.
    The script accepts two options:
    - the compiler -- \c gcc, \c mingw32, \c intel, \c open64, \c pgi, or \c ibm
    - the build type -- \c debug, \c release, or \c relwdeb
    The script then creates the directory \c build and runs \c cmake in it to
    set up the build system according to the options provided.

    \section The Build

    After the configuration has succeeded \c libtensor can be build by changing
    into directory \c build and executing \c make. This will create the static
    library \c libtensor.a in the subdirectory \c libtensor and
    several unit test suites \c libtensor_.*_tests in the subdirectory
    \c tests of the build folder.

    \section Compiling \c libtensor with Other Programs

    ... coming soon ...

 **/

/** \page prog_guide Programmer's guide

    This guide describes the various concepts and components in \c libtensor
    and how to use them in a program.

    -# \subpage quick_start
    -# \subpage concepts
    -# \subpage components

 **/

/** \page quick_start Quick-start Guide

    This guide gives a short introduction into \c libtensor showing how to setup
    initial \c btensors and use them in an equation. It covers only a small
    portion of the existing functionality. For a more detailed account on
    the available classes please refer to the other parts of this manual.

    \section Setting up dimensions

    The initial step to create \c btensors is to setup the information on the
    dimensions and the block structure. For this purpose \c libtensor provides
    the class \c bispace<N>. A one-dimensional \c bispace (N=1) can be constructed
    by providing the total length of the dimension. Afterwards the total length
    can be split into smaller blocks by using the member function \c split.
    The 1D bispaces can then be combined into higher-dimensional spaces by
    using the overloaded operators "|" and "&" where the latter is only used,
    if the two bispaces it combines are identical.

    \code
    size_t nalpha, nbeta, nbasis, nb2;
    size_t nel = nalpha + nbeta; nv = 2 * nbasis - nel;
    bispace<1> o(nel), v(nv), b(nbasis), s2(nb2);
    o.split(nalpha);
    v.split(nbasis - nalpha);
    // ... add further splits ...
    for (size_t i = 1; i < (nbasis - 1) / 16; i++) {
        b.split(i * 16);
    }

    bispace<2> oo(o&o), ov(o|v), vv(v&v), bb(b&b);
    // ...
    \endcode

    \section Creating tensors

    With bispaces available it is straightforward to create one or more
    btensors. The resulting objects will not yet contain any data beside
    the information on the basic structure.

    \code
    btensor<2, double> f_oo(oo), f_vv(vv), f_bb(bb);
    btensor<4, double> i_oovv((o&o)|(v&v));
    \endcode

    \section Initializing symmetry

    After the btensor objects have been created, their symmetry can be set up.
    This should always be done before adding data to the objects. There are
    several steps required to add symmetry to a btensor:
    -# Obtain the symmetry object from the btensor
    \code
    block_tensor_wr_ctrl<2, double> ctrl(f_oo);
    symmetry<2, double> &sym = ctrl.req_symmetry();
    \endcode
    -# Create symmetry element objects:
    \code
    // permutational symmetry
    permutation<2> p01; p01.permute(0, 1);
    scalar_transf<double> tr;
    se_perm<2, double> se(p01, tr);

    // spin symmetry
    mask<2> msk; msk[0] = msk[1] = true;
    index<2> i00, i01, i10, i11;
    i10[0] = i01[1] = 1;
    i11[0] = i11[1] = 1;
    se_part<2, double> sp(oo.get_bis(), msk, 2);
    sp.add_map(i00, i11, tr);
    sp.mark_forbidden(i01);
    sp.mark_forbidden(i10);

    // ...
    \endcode
    -# Add the symmetry element objects to the symmetry
    \code
    sym.insert(se);
    sym.insert(sp);
    \endcode

    \section Fill a block tensor with data

    As soon as the symmetry has been set up, a btensor object can be populated
    with data. There are several ways to do so. For example, the classes
    \c btod_read and \c btod_import_raw allow to read the tensor data from an
    input stream or a simple data pointer. The most comprehensive way, however,
    is to loop over all canonical blocks (i.e. tensor blocks which are allowed
    by the symmetry and cannot be obtained from previous blocks using a tensor
    transformation) and write data into each block separately.

    \subsection Loop over tensor blocks

    The classes \c orbit_list and \c orbit provide the required functionality
    to perform the loop over the canonical blocks of a block tensor.

    \code
    // Request a control object
    block_tensor_wr_ctrl<2, double> ctrl(f_oo);

    // Loop over all canonical blocks using orbit_list
    orbit_list<2, double> ol(ctrl.req_const_symmetry());
    for (orbit_list<2, double>::iterator it = ol.begin();
               it != ol.end(); it++) {

        // Obtain the index of the current canonical block
        index<2> bidx;
        ol.get_index(it, bidx);

        // Request tensor block from control object
        dense_tensor_wr_i<2, double> &blk = ctrl.req_block(bidx);

        // Fill with data (see next section)

        // Return the tensor block (mark as done)
        ctrl.ret_block(bidx);
    }
    \endcode

    \subsection Fill tensor blocks with data

    The tensor blocks of a block tensor are themselves tensor objects of type
    dense_tensor. If a tensor block is requested from a block tensor, it will
    return a reference to either the read- or the write-interface of the
    respective dense_tensor (here we only make use of the write-interface).
    From them we can retrieve a pointer to the actual data array which will
    be filled with data.

    \code
    dense_tensor_wr_i<2, double> &blk = ctrl.req_block(bidx);
    dense_tensor_wr_ctrl<2, double> tc(blk);

    // Obtain dimensions of tensor block
    const dimensions<2> &tdims = blk.get_dims();
    // Request data pointer
    double *ptr = blk.req_dataptr();

    for (size_t i = 0; i < tdims.get_size(); i++) {
        ptr[i] = 0.0;
    }

    // Return data pointer
    blk.ret_dataptr(ptr);
    \endcode

    \section Using tensors to store the result of an equation

    The previous sections described how to put data from external source
    into some previously created tensor. This is only necessary for the
    tensors which hold the initial data. For other tensors whose contents
    can be computed using the initial tensors the "Creating tensors" step
    is sufficient. As soon as this is done the tensors can be filled with
    data by employing them as the l.h.s of an expression. The symmetry
    and contents of the tensor blocks is automatically determined by the
    expression.

    \code
    btensor<2, double> v_ov(o|v), w_ov(o|v);

    // Fill v_ov with data
    // ...

    // Compute w_ov using an expression
    letter i, j, a, b;
    w_ov(i|a) = contract(b, i_oovv(i|j|a|b), v_ov(j|b));
    \endcode

 **/

/** \page reference Reference Manual

    \section Directory Structure

    - \c cmake/ -- Files related to the build system \c cmake.
    - \c libtensor/ -- The actual tensor library (for details see below).
    - \c performance_tests/ -- Set of classes to measure the performance (outdated).
    - \c tests -- Unit tests for the classes and functions in the tensor library.

    ... to be continued ...

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

/** \defgroup libtensor_symmetry Symmetry related components
    \brief Easy to use interface to implement equations with block tensors.
    \ingroup libtensor
 **/

#endif // LIBTENSOR_DEFS_H

