#ifndef LIBTENSOR_IFACE_BTENSOR_I_H
#define LIBTENSOR_IFACE_BTENSOR_I_H

#include <libtensor/block_tensor/block_tensor_i.h>
#include "any_tensor_impl.h"

namespace libtensor {
namespace iface {


/** \brief Block tensor interface
    \tparam N Block tensor order.
    \tparam T Block tensor element type.

    \ingroup libtensor_iface
**/
template<size_t N, typename T>
class btensor_i :
    virtual public block_tensor_rd_i<N, T>, public any_tensor<N, T> {

public:
    btensor_i() : any_tensor<N, T>(*this) { }

};


} // namespace iface
} // namespace libtensor


namespace libtensor {

using iface::btensor_i;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_BTENSOR_I_H
