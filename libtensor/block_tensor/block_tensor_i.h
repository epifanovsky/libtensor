#ifndef LIBTENSOR_BLOCK_TENSOR_I_H
#define LIBTENSOR_BLOCK_TENSOR_I_H

#include <libtensor/gen_block_tensor/gen_block_tensor_i.h>
#include "block_tensor_i_traits.h"

namespace libtensor {


template<size_t N, typename T> class block_tensor_ctrl;


/** \brief Block tensor base interface
    \tparam N Tensor order.
    \tparam T Tensor element type.

    See block_tensor_i for full documentation.

    \sa block_tensor_rd_i, block_tensor_wr_i, block_tensor_i, gen_block_tensor_i

    \ingroup libtensor_block_tensor
 **/
template<size_t N, typename T>
class block_tensor_base_i :
    virtual public gen_block_tensor_base_i< N, block_tensor_i_traits<N, T> > {

    friend class block_tensor_ctrl<N, T>;

public:
    /** \brief Virtual destructor
     **/
    virtual ~block_tensor_base_i() { }

};


/** \brief Block tensor read-only interface
    \tparam N Tensor order.
    \tparam T Tensor element type.

    See block_tensor_i for full documentation.

    \sa block_tensor_base_i, block_tensor_wr_i, block_tensor_i,
        gen_block_tensor_i

    \ingroup libtensor_block_tensor
 **/
template<size_t N, typename T>
class block_tensor_rd_i :
    virtual public block_tensor_base_i<N, T>,
    virtual public gen_block_tensor_rd_i< N, block_tensor_i_traits<N, T> > {

    friend class block_tensor_ctrl<N, T>;

public:
    /** \brief Virtual destructor
     **/
    virtual ~block_tensor_rd_i() { }

};


/** \brief Block tensor read-write interface
    \tparam N Tensor order.
    \tparam T Tensor element type.

    See block_tensor_i for full documentation.

    \sa block_tensor_base_i, block_tensor_rd_i, block_tensor_i,
        gen_block_tensor_i

    \ingroup libtensor_block_tensor
 **/
template<size_t N, typename T>
class block_tensor_wr_i :
    virtual public block_tensor_base_i<N, T>,
    virtual public gen_block_tensor_wr_i< N, block_tensor_i_traits<N, T> > {

    friend class block_tensor_ctrl<N, T>;

public:
    /** \brief Virtual destructor
     **/
    virtual ~block_tensor_wr_i() { }

};


/** \brief Block tensor interface
    \tparam N Tensor order.
    \tparam T Tensor element type.

    See gen_block_tensor_i for a general description of block tensors and
    block_tensor_i for the type of block tensors with dense blocks.

    \ingroup libtensor_block_tensor
 **/
template<size_t N, typename T>
class block_tensor_i :
    virtual public block_tensor_rd_i<N, T>,
    virtual public block_tensor_wr_i<N, T>,
    virtual public gen_block_tensor_i< N, block_tensor_i_traits<N, T> > {

    friend class block_tensor_ctrl<N, T>;

public:
    /** \brief Virtual destructor
     **/
    virtual ~block_tensor_i() { }

};


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_I_H
