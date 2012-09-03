#ifndef LIBTENSOR_BLOCK_TENSOR_TRAITS_H
#define LIBTENSOR_BLOCK_TENSOR_TRAITS_H

namespace libtensor {


template<size_t N, typename T> class dense_tensor_i;


/** \brief Block tensor traits
    \tparam T Tensor element type.

    This structure specifies the types that define the identity of simple
    block tensors.

    \ingroup libtensor_block_tensor
 **/
template<typename T>
struct block_tensor_traits {

    //! Type of tensor elements
    typedef T element_type;

    //! Type of read-only blocks as returned by the block tensor
    template<size_t N>
    struct rd_block_type {
        typedef dense_tensor_i<N, T> type;
    };

    //! Type of read-write blocks as returned by the block tensor
    template<size_t N>
    struct wr_block_type {
        typedef dense_tensor_i<N, T> type;
    };

};


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_TRAITS_H
