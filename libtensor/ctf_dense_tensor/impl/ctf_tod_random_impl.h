#ifndef LIBTENSOR_CTF_TOD_RANDOM_IMPL_H
#define LIBTENSOR_CTF_TOD_RANDOM_IMPL_H

#include <libtensor/linalg/linalg.h>
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_tod_random.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_random<N>::k_clazz[] = "ctf_tod_random<N>";


template<size_t N>
ctf_tod_random<N>::ctf_tod_random(const scalar_transf<double> &c) :
    m_c(c.get_coeff()) {

}


template<size_t N>
ctf_tod_random<N>::ctf_tod_random(double c) : m_c(c) {

}


template<size_t N>
void ctf_tod_random<N>::perform(bool zero, ctf_dense_tensor_i<N, double> &ta) {

    ctf_dense_tensor_ctrl<N, double> ca(ta);
    tCTF_Tensor<double> &dta = ca.req_ctf_tensor();

    long_int np, *idx;
    double *p;

    dta.read_local(&np, &idx, &p);
    if(zero) linalg::rng_set_i_x(0, np, p, 1, m_c);
    else linalg::rng_add_i_x(0, np, p, 1, m_c);
    dta.write(np, idx, p);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_RANDOM_IMPL_H
