#ifndef LIBTENSOR_TOD_RANDOM_IMPL_H
#define LIBTENSOR_TOD_RANDOM_IMPL_H

#include <ctime>
#include <cstdlib>
#include "../dense_tensor_ctrl.h"
#include "../tod_random.h"

namespace libtensor {


template<size_t N>
const char *tod_random<N>::k_clazz = "tod_random<N>";


template<size_t N>
tod_random<N>::tod_random(const scalar_transf<double> &c) : m_c(c.get_coeff()) {

    update_seed();
}


template<size_t N>
tod_random<N>::tod_random(double c) : m_c(c) {

    update_seed();
}


template<size_t N>
void tod_random<N>::update_seed() {

    static time_t timestamp = time(0);
    static long seed = timestamp;
    if(time(0) - timestamp > 60) {

        timestamp = time(0);
        seed += timestamp + lrand48();
        srand48(seed);
    }
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

    if(zero) {
        for(size_t i = 0; i < sz; i++) ptr[i] = m_c * drand48();
    } else {
        for(size_t i = 0; i < sz; i++) ptr[i] += m_c * drand48();
    }

    ctrl.ret_dataptr(ptr);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_RANDOM_IMPL_H
