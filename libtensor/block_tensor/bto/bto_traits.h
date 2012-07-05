#ifndef LIBTENSOR_BTO_TRAITS_H
#define LIBTENSOR_BTO_TRAITS_H

#include <libtensor/core/block_tensor_i.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>

namespace libtensor {

/** \brief Tempate
 **/
template<typename T>
struct bto_traits {

    //! BTO traits type required by additive_bto
    typedef bto_traits<T> additive_bto_traits;

    //! Element type
    typedef T element_type;

    //! Type of block tensor
    template<size_t N> struct block_tensor_type;

    //! Type of block tensor control
    template<size_t N> struct block_tensor_ctrl_type;

    //! Type of block of block tensors
    template<size_t N> struct block_type;

    //! \name Types of tensor operations
    //@{

    template<size_t N> struct to_add_type;

    template<size_t N> struct to_apply_type;

    template<size_t N> struct to_compare_type;

    template<size_t N> struct to_contract2_type;

    template<size_t N> struct to_copy_type;

    template<size_t N> struct to_diag_type;

    template<size_t N> struct to_dirsum_type;

    template<size_t N> struct to_dotprod_type;

    template<size_t N> struct to_ewmult2_type;

    template<size_t N> struct to_extract_type;

    template<size_t N> struct to_import_raw_type;

    template<size_t N> struct to_import_raw_stream_type;

    template<size_t N> struct to_mult_type;

    template<size_t N> struct to_mult1_type;

    template<size_t N> struct to_random_type;

    template<size_t N> struct to_scale_type;

    template<size_t N> struct to_scatter_type;

    template<size_t N> struct to_screen_type;

    template<size_t N> struct to_set_type;

    template<size_t N> struct to_set_diag_type;

    template<size_t N> struct to_set_elem_type;

    template<size_t N> struct to_trace_type;

    template<size_t N> struct to_vmpriority_type;

    //@}

    /** \brief Checks if the element is zero
     **/
    static bool is_zero(const element_type& d);

    /** \brief Returns the zero element
     **/
    static element_type zero();

    /** \brief Returns the one/identity element
     **/
    static element_type identity();
};


} // namespace libtensor

#endif // LIBTENSOR_BTO_TRAITS_H
