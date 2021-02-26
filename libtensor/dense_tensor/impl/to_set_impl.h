#ifndef LIBTENSOR_TO_SET_IMPL_H
#define LIBTENSOR_TO_SET_IMPL_H

#include "../dense_tensor_ctrl.h"
#include "../to_set.h"

namespace libtensor {


template<size_t N, typename T>
const char *to_set<N,T>::k_clazz = "to_set<N,T>";


template<size_t N, typename T>
void to_set<N,T>::perform(bool zero, dense_tensor_wr_i<N, T> &ta) {

    if (! zero && m_v == 0.0) return;

    to_set<N,T>::start_timer();

    try {

        dense_tensor_wr_ctrl<N, T> ca(ta);
        T *p = ca.req_dataptr();

        size_t sz = ta.get_dims().get_size();
        if (zero)
            for(size_t i = 0; i < sz; i++) p[i] = m_v;
        else
            for(size_t i = 0; i < sz; i++) p[i] += m_v;
        ca.ret_dataptr(p); p = 0;

    } catch(...) {
        to_set<N,T>::stop_timer();
        throw;
    }

    to_set<N,T>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TO_SET_IMPL_H
