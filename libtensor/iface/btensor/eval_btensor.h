#ifndef LIBTENSOR_IFACE_EVAL_BTENSOR_H
#define LIBTENSOR_IFACE_EVAL_BTENSOR_H

namespace libtensor {
namespace iface {


/** \brief Processor of evaluation plan for btensor result type
    \tparam T Tensor element type.

    \ingroup libtensor_iface
 **/
template<typename T>
class eval_btensor;


} // namespace iface
} // namespace libtensor

#include "eval_btensor_double.h"

#endif // LIBTENSOR_IFACE_EVAL_BTENSOR_H
