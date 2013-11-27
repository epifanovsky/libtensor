#ifndef LIBTENSOR_DIAG_BLOCK_TENSOR_TRAITS_H
#define LIBTENSOR_DIAG_BLOCK_TENSOR_TRAITS_H

#include <libtensor/diag_tensor/diag_tensor.h>
#include "diag_block_factory.h"
#include "diag_block_tensor_i_traits.h"

namespace libtensor {


/** \brief Diagonal block tensor traits
    \tparam T Tensor element type.
    \tparam Alloc Memory allocator.

    \sa block_tensor_traits

    \ingroup libtensor_diag_block_tensor
 **/
template<typename T, typename Alloc>
struct diag_block_tensor_traits {

    //! Type of tensor elements
    typedef T element_type;

    //! Type of allocator
    typedef Alloc allocator_type;

    //! Traits of block tensor interface
    typedef diag_block_tensor_i_traits<T> bti_traits;

    //! Type of blocks
    template<size_t N>
    struct block_type {
        typedef diag_tensor<N, T, Alloc> type;
    };

    //! Type of block factory
    template<size_t N>
    struct block_factory_type {
        typedef diag_block_factory<N, T, Alloc> type;
    };

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_BLOCK_TENSOR_TRAITS_H
