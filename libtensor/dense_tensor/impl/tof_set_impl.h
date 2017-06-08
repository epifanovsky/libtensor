#ifndef LIBTENSOR_TOD_SET_IMPL_H
#define LIBTENSOR_TOD_SET_IMPL_H

#include "../dense_tensor_ctrl.h"
#include "../tof_set.h"

namespace libtensor {


template<size_t N>
const char *tof_set<N>::k_clazz = "tof_set<N>";


template<size_t N>
void tof_set<N>::perform(bool zero, dense_tensor_wr_i<N, float> &ta) {


// Functionality is temporary disabled to create tests

/*
    if (! zero && m_v == 0.0) return;

    tof_set<N>::start_timer();

    try {

        dense_tensor_wr_ctrl<N, float> ca(ta);
        float *p = ca.req_dataptr();

        size_t sz = ta.get_dims().get_size();
        if (zero)
            for(size_t i = 0; i < sz; i++) p[i] = m_v;
        else
            for(size_t i = 0; i < sz; i++) p[i] += m_v;
        ca.ret_dataptr(p); p = 0;

    } catch(...) {
        tof_set<N>::stop_timer();
        throw;
    }

    tof_set<N>::stop_timer();
*/
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_IMPL_H
