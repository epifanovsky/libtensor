#ifndef LIBTENSOR_TOD_SET_IMPL_H
#define LIBTENSOR_TOD_SET_IMPL_H

#include "../dense_tensor_ctrl.h"
#include "../tod_set.h"

namespace libtensor {


template<size_t N>
const char *tod_set<N>::k_clazz = "tod_set<N>";


template<size_t N>
void tod_set<N>::perform(bool zero, dense_tensor_wr_i<N, double> &ta) {

    if (! zero && m_v == 0.0) return;

    tod_set<N>::start_timer();

    try {

        dense_tensor_wr_ctrl<N, double> ca(ta);
        double *p = ca.req_dataptr();

        size_t sz = ta.get_dims().get_size();
        if (zero)
            for(size_t i = 0; i < sz; i++) p[i] = m_v;
        else
            for(size_t i = 0; i < sz; i++) p[i] += m_v;
        ca.ret_dataptr(p); p = 0;

    } catch(...) {
        tod_set<N>::stop_timer();
        throw;
    }

    tod_set<N>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_IMPL_H
