#ifndef LIBTENSOR_TOD_SET_IMPL_H
#define LIBTENSOR_TOD_SET_IMPL_H

#include "../dense_tensor_ctrl.h"
#include "../tod_set.h"

namespace libtensor {


template<size_t N>
void tod_set<N>::perform(cpu_pool &cpus, dense_tensor_wr_i<N, double> &t) {

    dense_tensor_wr_ctrl<N, double> ctrl(t);
    double *d = ctrl.req_dataptr();

    {
        auto_cpu_lock cpu(cpus);
        size_t sz = t.get_dims().get_size();
        for(size_t i = 0; i < sz; i++) d[i] = m_v;
    }

    ctrl.ret_dataptr(d);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_IMPL_H
