#ifndef LIBTENSOR_TO_SCREEN_IMPL_H
#define LIBTENSOR_TO_SCREEN_IMPL_H

#include "../dense_tensor_ctrl.h"
#include "../to_screen.h"

namespace libtensor {


template<size_t N, typename T>
const char *to_screen<N, T>::k_clazz = "to_screen<N, T>";


template<size_t N, typename T>
bool to_screen<N, T>::perform_replace(dense_tensor_wr_i<N, T> &t) {

    to_screen::start_timer("replace");

    dense_tensor_wr_ctrl<N, T> ctrl(t);

    bool ret = false;

    size_t sz = t.get_dims().get_size();
    T *p = ctrl.req_dataptr();

    for(size_t i = 0; i < sz; i++) {
        if(std::abs(p[i] - m_a) < m_thresh) {
            p[i] = m_a;
            ret = true;
        }
    }

    ctrl.ret_dataptr(p);

    to_screen::stop_timer("replace");

    return ret;
}


template<size_t N, typename T>
bool to_screen<N, T>::perform_screen(dense_tensor_rd_i<N, T> &t) {

    to_screen::start_timer("screen");

    dense_tensor_rd_ctrl<N, T> ctrl(t);

    bool ret = false;

    size_t sz = t.get_dims().get_size();
    const T *p = ctrl.req_const_dataptr();

    for(size_t i = 0; i < sz; i++) {
        if(std::abs(p[i] - m_a) < m_thresh) {
            ret = true;
            break;
        }
    }

    ctrl.ret_const_dataptr(p);

    to_screen::stop_timer("screen");

    return ret;
}


} // namespace libtensor

#endif // LIBTENSOR_TO_SCREEN_IMPL_H
