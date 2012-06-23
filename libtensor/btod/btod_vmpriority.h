#ifndef LIBTENSOR_BTOD_VMPRIORITY_H
#define LIBTENSOR_BTOD_VMPRIORITY_H

#include <libtensor/core/block_tensor_i.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include <libtensor/dense_tensor/tod_vmpriority.h>
#include <libtensor/block_tensor/bto/bto_vmpriority.h>
#include <libtensor/block_tensor/btod/btod_traits.h>

namespace libtensor {


struct btod_vmpriority_traits : public bto_traits<double> {

    template<size_t N> struct to_vmpriority_type {
        typedef tod_vmpriority<N> type;
    };
};


/** \brief Sets or unsets the VM in-core priority
    \tparam N Tensor order.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_vmpriority : public bto_vmpriority<N, btod_vmpriority_traits> {
public:
    static const char *k_clazz; //!< Class name

public:
    /** \brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    btod_vmpriority(block_tensor_i<N, double> &bt) :
        bto_vmpriority<N, btod_vmpriority_traits>(bt)
    { }

};


template<size_t N>
const char *btod_vmpriority<N>::k_clazz = "btod_vmpriority<N>";


} // namespace libtensor

#endif // LIBTENSOR_BTOD_VMPRIORITY_H
