#ifndef LIBTENSOR_TO_GET_ELEM_IMPL_H
#define LIBTENSOR_TO_GET_ELEM_IMPL_H

#include <libtensor/core/abs_index.h>
#include "../dense_tensor_ctrl.h"
#include "../to_get_elem.h"

namespace libtensor {


template<size_t N, typename T>
void to_get_elem<N, T>::perform(dense_tensor_rd_i<N, T> &t,
    const index<N> &idx, T& d) {

    dense_tensor_rd_ctrl<N, T> ctrl(t);
    const T *p = ctrl.req_const_dataptr();
    d = p[abs_index<N>(idx, t.get_dims()).get_abs_index()];
    ctrl.ret_const_dataptr(p);
}


} // namespace libtensor

#endif // LIBTENSOR_TO_GET_ELEM_IMPL_H
