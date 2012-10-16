#ifndef LIBTENSOR_BTOD_VMPRIORITY_H
#define LIBTENSOR_BTOD_VMPRIORITY_H

#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/gen_bto_vmpriority.h>

namespace libtensor {


/** \brief Sets or unsets the VM in-core priority
    \tparam N Tensor order.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_vmpriority : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    gen_bto_vmpriority< N, btod_traits> m_gbto;

public:
    /** \brief Initializes the operation
        \param bt Block tensor to set priority for
     **/
    btod_vmpriority(block_tensor_i<N, double> &bt) :
        m_gbto(bt)
    { }

    /** \brief Sets the VM in-core priority
     **/
    void set_priority() {
        m_gbto.set_priority();
    }

    /** \brief Unsets the VM in-core priority
     **/
    void unset_priority() {
        m_gbto.unset_priority();
    }
};


template<size_t N>
const char *btod_vmpriority<N>::k_clazz = "btod_vmpriority<N>";


} // namespace libtensor

#endif // LIBTENSOR_BTOD_VMPRIORITY_H
