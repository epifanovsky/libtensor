#ifndef LIBTENSOR_TOD_SET_ELEM_IMPL_H
#define LIBTENSOR_TOD_SET_ELEM_IMPL_H

#include <libtensor/core/abs_index.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include "../tod_set_elem.h"

namespace libtensor {


template<size_t N>
void tod_set_elem<N>::perform(dense_tensor_wr_i<N, double> &t,
    const index<N> &idx, double d) {

    dense_tensor_wr_ctrl<N, double> ctrl(t);
    double *p = ctrl.req_dataptr();
    p[abs_index<N>(idx, t.get_dims()).get_abs_index()] = d;
    ctrl.ret_dataptr(p);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_ELEM_IMPL_H
