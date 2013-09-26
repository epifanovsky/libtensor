#ifndef LIBTENSOR_CTF_BLOCK_TENSOR_I_H
#define LIBTENSOR_CTF_BLOCK_TENSOR_I_H

#include <libtensor/gen_block_tensor/gen_block_tensor_i.h>
#include "ctf_block_tensor_i_traits.h"

namespace libtensor {


/** \brief Read-only interface of CTF block tensor
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N, typename T>
class ctf_block_tensor_rd_i :
    virtual public gen_block_tensor_rd_i< N, ctf_block_tensor_i_traits<T> > {

public:
    /** \brief Virtual destructor
     **/
    virtual ~ctf_block_tensor_rd_i() { }

};


/** \brief Read-write interface of CTF block tensor
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N, typename T>
class ctf_block_tensor_wr_i :
    virtual public gen_block_tensor_wr_i< N, ctf_block_tensor_i_traits<T> > {

public:
    /** \brief Virtual destructor
     **/
    virtual ~ctf_block_tensor_wr_i() { }

};


/** \brief Interface of CTF block tensor
    \tparam N Tensor order.
    \tparam T Tensor element type.

    See gen_block_tensor_i for a general description of block tensors
    and their interface.

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N, typename T>
class ctf_block_tensor_i :
    virtual public ctf_block_tensor_rd_i<N, T>,
    virtual public ctf_block_tensor_wr_i<N, T>,
    virtual public gen_block_tensor_i< N, ctf_block_tensor_i_traits<T> > {

public:
    /** \brief Virtual destructor
     **/
    virtual ~ctf_block_tensor_i() { }

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BLOCK_TENSOR_I_H
