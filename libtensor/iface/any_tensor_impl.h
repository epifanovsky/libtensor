#ifndef LIBTENSOR_IFACE_ANY_TENSOR_IMPL_H
#define LIBTENSOR_IFACE_ANY_TENSOR_IMPL_H

#include "any_tensor.h"
#include "ident/ident_core.h"

namespace libtensor {
namespace iface {
using libtensor::labeled_btensor_expr::ident_core;


template<size_t N, typename T>
expr_rhs<N, T> any_tensor<N, T>::make_rhs(const letter_expr<N> &label) {

    return expr_rhs<N, T>(new ident_core<N, T>(*this, label));
}


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_ANY_TENSOR_IMPL_H
