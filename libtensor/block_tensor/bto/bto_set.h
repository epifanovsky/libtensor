#ifndef LIBTENSOR_BTO_SET_H
#define LIBTENSOR_BTO_SET_H

#include <libtensor/defs.h>

namespace libtensor {


/** \brief Sets all elements of a block tensor to a value preserving
        the symmetry
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.

    In every block that is allowed by the symmetry in the block tensor,
    this operation sets all elements to a specified value.

    If the value is zero, all the blocks are zeroed out.

    This operation does not make sure that the symmetry of each block is
    preserved. For example, it will not zero the diagonal of a anti-symmetric
    tensor.

    \ingroup libtensor_bto
 **/
template<size_t N, typename Traits>
class bto_set {
public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Type of block tensors
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_type;

private:
    element_type m_v; //!< Value

public:
    /** \brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    bto_set(const element_type &v) : m_v(v) { }

    /** \brief Performs the operation
        \param bt Output block tensor.
     **/
    void perform(block_tensor_type &bt);

private:
    /** \brief Forbidden copy constructor
     **/
    bto_set(const bto_set&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_SET_H
