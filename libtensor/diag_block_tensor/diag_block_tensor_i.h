#ifndef LIBTENSOR_DIAG_BLOCK_TENSOR_I_H
#define LIBTENSOR_DIAG_BLOCK_TENSOR_I_H

#include <libtensor/gen_block_tensor/gen_block_tensor_i.h>
#include "diag_block_tensor_i_traits.h"

namespace libtensor {


template<size_t N, typename T> class diag_block_tensor_ctrl;


/** \brief Diagonal block tensor base interface
    \tparam N Tensor order.
    \tparam T Tensor element type.

    See diag_block_tensor_i for full documentation.

    \sa diag_block_tensor_rd_i, diag_block_tensor_wr_i, diag_block_tensor_i,
        gen_block_tensor_i

    \ingroup libtensor_diag_block_tensor
 **/
template<size_t N, typename T>
class diag_block_tensor_base_i :
    virtual public gen_block_tensor_base_i< N, diag_block_tensor_i_traits<T> > {

    friend class diag_block_tensor_ctrl<N, T>;

public:
    /** \brief Virtual destructor
     **/
    virtual ~diag_block_tensor_base_i() { }

};


/** \brief Diagonal block tensor read-only interface
    \tparam N Tensor order.
    \tparam T Tensor element type.

    See diag_block_tensor_i for full documentation.

    \sa diag_block_tensor_base_i, diag_block_tensor_wr_i, diag_block_tensor_i,
        gen_block_tensor_i

    \ingroup libtensor_diag_block_tensor
 **/
template<size_t N, typename T>
class diag_block_tensor_rd_i :
    virtual public diag_block_tensor_base_i<N, T>,
    virtual public gen_block_tensor_rd_i< N, diag_block_tensor_i_traits<T> > {

    friend class diag_block_tensor_ctrl<N, T>;

public:
    /** \brief Virtual destructor
     **/
    virtual ~diag_block_tensor_rd_i() { }

};


/** \brief Diagonal block tensor read-write interface
    \tparam N Tensor order.
    \tparam T Tensor element type.

    See diag_block_tensor_i for full documentation.

    \sa diag_block_tensor_base_i, diag_block_tensor_rd_i, diag_block_tensor_i,
        gen_block_tensor_i

    \ingroup libtensor_diag_block_tensor
 **/
template<size_t N, typename T>
class diag_block_tensor_wr_i :
    virtual public diag_block_tensor_base_i<N, T>,
    virtual public gen_block_tensor_wr_i< N, diag_block_tensor_i_traits<T> > {

    friend class diag_block_tensor_ctrl<N, T>;

public:
    /** \brief Virtual destructor
     **/
    virtual ~diag_block_tensor_wr_i() { }

};


/** \brief Diagonal block tensor interface
    \tparam N Tensor order.
    \tparam T Tensor element type.

    See gen_block_tensor_i for a general description of block tensors.

    The diagonal block tensor is a type of block tensors that contains
    diag_tensor_i objects as blocks. The block tensor itself does not
    necessarily have to be diagonal.

    \sa diag_block_tensor_i_traits, diag_tensor_i

    \ingroup libtensor_diag_block_tensor
 **/
template<size_t N, typename T>
class diag_block_tensor_i :
    virtual public diag_block_tensor_rd_i<N, T>,
    virtual public diag_block_tensor_wr_i<N, T>,
    virtual public gen_block_tensor_i< N, diag_block_tensor_i_traits<T> > {

    friend class diag_block_tensor_ctrl<N, T>;

public:
    /** \brief Virtual destructor
     **/
    virtual ~diag_block_tensor_i() { }

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_BLOCK_TENSOR_I_H
