#ifndef LIBTENSOR_TOD_SCREEN_IMPL_H
#define LIBTENSOR_TOD_SCREEN_IMPL_H

#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include "../tod_screen.h"

namespace libtensor {


template<size_t N>
const char *tod_screen<N>::k_clazz = "tod_screen<N>";


template<size_t N>
bool tod_screen<N>::perform_replace(dense_tensor_wr_i<N, double> &t) {

    tod_screen::start_timer("replace");

    dense_tensor_wr_ctrl<N, double> ctrl(t);

    bool ret = false;

    size_t sz = t.get_dims().get_size();
    double *p = ctrl.req_dataptr();

    for(size_t i = 0; i < sz; i++) {
        if(fabs(p[i] - m_a) < m_thresh) {
            p[i] = m_a;
            ret = true;
        }
    }

    ctrl.ret_dataptr(p);

    tod_screen::stop_timer("replace");

    return ret;
}


template<size_t N>
bool tod_screen<N>::perform_screen(dense_tensor_rd_i<N, double> &t) {

    tod_screen::start_timer("screen");

    dense_tensor_rd_ctrl<N, double> ctrl(t);

    bool ret = false;

    size_t sz = t.get_dims().get_size();
    const double *p = ctrl.req_const_dataptr();

    for(size_t i = 0; i < sz; i++) {
        if(fabs(p[i] - m_a) < m_thresh) {
            ret = true;
            break;
        }
    }

    ctrl.ret_const_dataptr(p);

    tod_screen::stop_timer("screen");

    return ret;
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_SCREEN_IMPL_H
