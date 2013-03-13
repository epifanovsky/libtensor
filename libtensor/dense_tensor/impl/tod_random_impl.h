#ifndef LIBTENSOR_TOD_RANDOM_IMPL_H
#define LIBTENSOR_TOD_RANDOM_IMPL_H

#include <libtensor/linalg/linalg.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_random.h"

namespace libtensor {


template<size_t N>
const char *tod_random<N>::k_clazz = "tod_random<N>";


template<size_t N>
tod_random<N>::tod_random(const scalar_transf<double> &c) : m_c(c.get_coeff()) {

}


template<size_t N>
tod_random<N>::tod_random(double c) : m_c(c) {

}


template<size_t N>
void tod_random<N>::perform(dense_tensor_wr_i<N, double> &t) {

    perform(true, t);
}


template<size_t N>
void tod_random<N>::perform(bool zero, dense_tensor_wr_i<N, double> &t) {

    dense_tensor_wr_ctrl<N, double> ctrl(t);
    size_t sz = t.get_dims().get_size();
    double *ptr = ctrl.req_dataptr();

    if(zero) linalg::rng_set_i_x(0, sz, ptr, 1, m_c);
    else linalg::rng_add_i_x(0, sz, ptr, 1, m_c);

    ctrl.ret_dataptr(ptr);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_RANDOM_IMPL_H
