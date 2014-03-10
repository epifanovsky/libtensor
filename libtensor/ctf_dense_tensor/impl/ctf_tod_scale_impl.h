#ifndef LIBTENSOR_CTF_TOD_SCALE_IMPL_H
#define LIBTENSOR_CTF_TOD_SCALE_IMPL_H

#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_error.h"
#include "../ctf_tod_scale.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_scale<N>::k_clazz[] = "ctf_tod_scale<N>";


template<size_t N>
ctf_tod_scale<N>::ctf_tod_scale(const scalar_transf<double> &c) :

    m_c(c.get_coeff()) {

}


template<size_t N>
void ctf_tod_scale<N>::perform(ctf_dense_tensor_i<N, double> &ta) {

    ctf_dense_tensor_ctrl<N, double> ca(ta);
    tCTF_Tensor<double> &dta = ca.req_ctf_tensor();

    char mapa[N];
    for(size_t i = 0; i < N; i++) mapa[i] = char(i + 1);

    dta.scale(m_c, mapa);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_SCALE_IMPL_H
