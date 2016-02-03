#ifndef LIBTENSOR_CTF_BLOCK_TENSOR_I_TRAITS_H
#define LIBTENSOR_CTF_BLOCK_TENSOR_I_TRAITS_H

namespace libtensor {


template<size_t N, typename T> class ctf_dense_tensor_i;


/** \brief Interface traits for the distributed CTF block tensor
    \tparam N Tensor order.
    \tparam T Tensor element type.

    This structure specifies the types that define the identity of
    the distributed block tensors.

    \sa ctf_dense_tensor_i

    \ingroup libtensor_ctf_block_tensor
 **/
template<typename T>
struct ctf_block_tensor_i_traits {

    //! Type of tensor elements
    typedef T element_type;

    //! Type of read-only blocks as returned by the block tensor
    template<size_t N>
    struct rd_block_type {
        typedef ctf_dense_tensor_i<N, T> type;
    };

    //! Type of read-write blocks as returned by the block tensor
    template<size_t N>
    struct wr_block_type {
        typedef ctf_dense_tensor_i<N, T> type;
    };

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BLOCK_TENSOR_I_TRAITS_H
