#ifndef LIBTENSOR_DIAG_BLOCK_TENSOR_I_TRAITS_H
#define LIBTENSOR_DIAG_BLOCK_TENSOR_I_TRAITS_H

namespace libtensor {


template<size_t N, typename T> class diag_tensor_rd_i;
template<size_t N, typename T> class diag_tensor_wr_i;
template<size_t N, typename T> class diag_tensor_i;


/** \brief Diagonal block tensor interface traits
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \sa block_tensor_i_traits, diag_block_tensor_i

    \ingroup libtensor_diag_block_tensor
 **/
template<typename T>
struct diag_block_tensor_i_traits {

    //! Type of tensor elements
    typedef T element_type;

    //! Type of read-only blocks as returned by the block tensor
    template<size_t N>
    struct rd_block_type {
        typedef diag_tensor_i<N, T> type;
    };

    //! Type of read-write blocks as returned by the block tensor
    template<size_t N>
    struct wr_block_type {
        typedef diag_tensor_i<N, T> type;
    };

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_BLOCK_TENSOR_I_TRAITS_H
