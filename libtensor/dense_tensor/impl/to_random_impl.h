#ifndef LIBTENSOR_TO_RANDOM_IMPL_H
#define LIBTENSOR_TO_RANDOM_IMPL_H

#include <libtensor/linalg/linalg.h>
#include "../dense_tensor_ctrl.h"
#include "../to_random.h"

namespace libtensor {


template<size_t N, typename T>
const char *to_random<N, T>::k_clazz = "to_random<N, T>";


template<size_t N, typename T>
to_random<N, T>::to_random(const scalar_transf<T> &c) : m_c(c.get_coeff()) {

}


template<size_t N, typename T>
to_random<N, T>::to_random(T c) : m_c(c) {

}


template<size_t N, typename T>
void to_random<N, T>::perform(dense_tensor_wr_i<N, T> &t) {

    perform(true, t);
}


template<size_t N, typename T>
void to_random<N, T>::perform(bool zero, dense_tensor_wr_i<N, T> &t) {

    dense_tensor_wr_ctrl<N, T> ctrl(t);
    size_t sz = t.get_dims().get_size();
    T *ptr = ctrl.req_dataptr();

    if(zero) linalg::rng_set_i_x(0, sz, ptr, 1, m_c);
    else linalg::rng_add_i_x(0, sz, ptr, 1, m_c);

    ctrl.ret_dataptr(ptr);
}


} // namespace libtensor

#endif // LIBTENSOR_TO_RANDOM_IMPL_H
