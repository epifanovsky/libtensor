#ifndef LIBTENSOR_TOD_SCALE_IMPL_H
#define LIBTENSOR_TOD_SCALE_IMPL_H

#include <libtensor/mp/auto_cpu_lock.h>
#include <libtensor/linalg/linalg.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_scale.h"

namespace libtensor {


template<size_t N>
const char *tod_scale<N>::k_clazz = "tod_scale<N>";


template<size_t N>
void tod_scale<N>::perform(cpu_pool &cpus, dense_tensor_wr_i<N, double> &ta) {

    tod_scale<N>::start_timer();

    try {

        dense_tensor_wr_ctrl<N, double> ca(ta);
        double *p = ca.req_dataptr();

        {
            auto_cpu_lock cpu(cpus);
            size_t sz = ta.get_dims().get_size();
            linalg::i_x(sz, m_c, p, 1);
        }

        ca.ret_dataptr(p); p = 0;

    } catch(...) {
        tod_scale<N>::stop_timer();
        throw;
    }

    tod_scale<N>::stop_timer();
}




} // namespace libtensor

#endif // LIBTENSOR_TOD_SCALE_IMPL_H
