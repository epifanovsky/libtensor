#ifndef LIBTENSOR_TO_SET_ELEM_IMPL_H
#define LIBTENSOR_TO_SET_ELEM_IMPL_H

#include <libtensor/core/abs_index.h>
#include "../dense_tensor_ctrl.h"
#include "../to_set_elem.h"

namespace libtensor {


template<size_t N, typename T>
void to_set_elem<N, T>::perform(dense_tensor_wr_i<N, T> &t,
    const index<N> &idx, T d) {

    dense_tensor_wr_ctrl<N, T> ctrl(t);
    T *p = ctrl.req_dataptr();
    p[abs_index<N>(idx, t.get_dims()).get_abs_index()] = d;
    ctrl.ret_dataptr(p);
}


} // namespace libtensor

#endif // LIBTENSOR_TO_SET_ELEM_IMPL_H
