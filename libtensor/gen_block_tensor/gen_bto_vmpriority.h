#ifndef LIBTENSOR_GEN_BTO_VMPRIORITY_H
#define LIBTENSOR_GEN_BTO_VMPRIORITY_H

#include <libtensor/core/noncopyable.h>
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Sets or unsets the virtual memory in-core priority
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.

    \ingroup libtensor_bto
 **/
template<size_t N, typename Traits>
class gen_bto_vmpriority : public noncopyable {
public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_block_tensor_rd_i<N, bti_traits> &m_bt; //!< Block tensor

public:
    /** \brief Initializes the operation
        \param bt Block tensor.
     **/
    gen_bto_vmpriority(gen_block_tensor_rd_i<N, bti_traits> &bt) : m_bt(bt) { }

    /** \brief Sets the VM in-core priority
     **/
    void set_priority();

    /** \brief Unsets the VM in-core priority
     **/
    void unset_priority();
};


} // namespace libtensor

#endif // LIBTENSOR_BTO_VMPRIORITY_H
