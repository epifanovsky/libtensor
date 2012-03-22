#ifndef LIBTENSOR_TOD_SET_DIAG_IMPL_H
#define LIBTENSOR_TOD_SET_DIAG_IMPL_H

#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/tod/bad_dimensions.h>
#include "../tod_set_diag.h"

namespace libtensor {


template<size_t N>
const char *tod_set_diag<N>::k_clazz = "tod_set_diag<N>";


template<size_t N>
void tod_set_diag<N>::perform(dense_tensor_wr_i<N, double> &t) {

    static const char *method = "perform(dense_tensor_wr_i<N, double>&)";

    const dimensions<N> &dims = t.get_dims();
    size_t n = dims[0];
    for(size_t i = 1; i < N; i++) {
        if(dims[i] != n) {
            throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
                "t");
        }
    }

    size_t inc = 0;
    for(size_t i = 0; i < N; i++) inc += dims.get_increment(i);

    dense_tensor_wr_ctrl<N, double> ctrl(t);
    double *d = ctrl.req_dataptr();
    for(size_t i = 0; i < n; i++) d[i * inc] = m_v;
    ctrl.ret_dataptr(d);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_DIAG_IMPL_H
