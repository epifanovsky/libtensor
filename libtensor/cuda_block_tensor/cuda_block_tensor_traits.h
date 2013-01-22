#ifndef LIBTENSOR_CUDA_BLOCK_TENSOR_TRAITS_H
#define LIBTENSOR_CUDA_BLOCK_TENSOR_TRAITS_H

#include <libtensor/block_tensor/block_factory.h>
#include "cuda_block_tensor.h"
#include "cuda_block_tensor_i_traits.h"

namespace libtensor {


/** \brief CUDA block tensor traits
    \tparam T Tensor element type.
    \tparam Alloc Memory allocator.

    This structure specifies the types that define the identity of CUDA
    block tensors.

    \ingroup libtensor_cuda_block_tensor
 **/
template<typename T, typename Alloc>
struct cuda_block_tensor_traits {

    //! Type of tensor elements
    typedef T element_type;

    //! Type of allocator
    typedef Alloc allocator_type;

    //! Traits of block tensor interface
    typedef cuda_block_tensor_i_traits<T> bti_traits;

    //! Type of blocks
    template<size_t N>
    struct block_type {
        typedef dense_tensor<N, T, Alloc> type;
    };

    //! Type of block factory
    template<size_t N>
    struct block_factory_type {
        typedef block_factory<N, T, typename block_type<N>::type> type;
    };

};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_BLOCK_TENSOR_TRAITS_H
