#ifndef LIBTENSOR_BTO_VMPRIORITY_H
#define LIBTENSOR_BTO_VMPRIORITY_H

#include <libtensor/block_tensor/bto_traits.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/gen_bto_vmpriority.h>

namespace libtensor {


/** \brief Sets or unsets the VM in-core priority
    \tparam N Tensor order.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, typename T>
class bto_vmpriority : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    gen_bto_vmpriority< N, bto_traits<T> > m_gbto;

public:
    /** \brief Initializes the operation
        \param bt Block tensor to set priority for
     **/
    bto_vmpriority(block_tensor_i<N, T> &bt) :
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
using btod_vmpriority = bto_vmpriority<N, double>;

template<size_t N, typename T>
const char *bto_vmpriority<N, T>::k_clazz = "bto_vmpriority<N, T>";


} // namespace libtensor

#endif // LIBTENSOR_BTO_VMPRIORITY_H
