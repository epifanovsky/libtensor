#ifndef LIBTENSOR_TO_SCALE_IMPL_H
#define LIBTENSOR_TO_SCALE_IMPL_H

#include <libtensor/linalg/linalg.h>
#include "../dense_tensor_ctrl.h"
#include "../to_scale.h"

namespace libtensor {


template<size_t N, typename T>
const char *to_scale<N, T>::k_clazz = "to_scale<N, T>";


template<size_t N, typename T>
inline to_scale<N, T>::to_scale(const scalar_transf<T> &c) :
    m_c(c.get_coeff()) {

}


template<size_t N, typename T>
inline to_scale<N, T>::to_scale(T c) : m_c(c) {

}


template<size_t N, typename T>
void to_scale<N, T>::perform(dense_tensor_wr_i<N, T> &ta) {

    to_scale<N, T>::start_timer();

    try {

        dense_tensor_wr_ctrl<N, T> ca(ta);
        T *p = ca.req_dataptr();

        size_t sz = ta.get_dims().get_size();
        linalg::mul1_i_x(0, sz, m_c, p, 1);

        ca.ret_dataptr(p); p = 0;

    } catch(...) {
        to_scale<N, T>::stop_timer();
        throw;
    }

    to_scale<N, T>::stop_timer();
}




} // namespace libtensor

#endif // LIBTENSOR_TO_SCALE_IMPL_H
