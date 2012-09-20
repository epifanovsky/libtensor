#ifndef LIBTENSOR_BTO_VMPRIORITY_H
#define LIBTENSOR_BTO_VMPRIORITY_H

#include <libtensor/block_tensor/block_tensor_i.h>

namespace libtensor {


/** \brief Sets or unsets the virtual memory in-core priority
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.

    \ingroup libtensor_bto
 **/
template<size_t N, typename Traits>
class bto_vmpriority {
public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Type of block tensors
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_type;

private:
    block_tensor_type &m_bt; //!< Block tensor

public:
    /** \brief Initializes the operation
        \param bt Block tensor.
     **/
    bto_vmpriority(block_tensor_type &bt) : m_bt(bt) { }

    /** \brief Sets the VM in-core priority
     **/
    void set_priority();

    /** \brief Unsets the VM in-core priority
     **/
    void unset_priority();

private:
    /** \brief Forbidden copy constructor
     **/
    bto_vmpriority(const bto_vmpriority&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_VMPRIORITY_H
