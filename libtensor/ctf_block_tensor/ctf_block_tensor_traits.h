#ifndef LIBTENSOR_CTF_BLOCK_TENSOR_TRAITS_H
#define LIBTENSOR_CTF_BLOCK_TENSOR_TRAITS_H

#include <libtensor/block_tensor/block_factory.h>
#include <libtensor/ctf_dense_tensor/ctf_dense_tensor.h>
#include "ctf_block_tensor_i_traits.h"

namespace libtensor {


/** \brief CTF block tensor traits
    \tparam T Tensor element type.
    \tparam Alloc Memory allocator.

    This structure specifies the types that define the identity of
    the distributed block tensors.

    \ingroup libtensor_ctf_block_tensor
 **/
template<typename T>
struct ctf_block_tensor_traits {

    //! Type of tensor elements
    typedef T element_type;

    //! Traits of block tensor interface
    typedef ctf_block_tensor_i_traits<T> bti_traits;

    //! Type of blocks
    template<size_t N>
    struct block_type {
        typedef ctf_dense_tensor<N, T> type;
    };

    //! Type of block factory
    template<size_t N>
    struct block_factory_type {
        typedef block_factory<N, T, typename block_type<N>::type> type;
    };

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BLOCK_TENSOR_TRAITS_H
