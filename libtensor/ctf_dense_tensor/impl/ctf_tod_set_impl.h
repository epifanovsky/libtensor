#ifndef LIBTENSOR_CTF_TOD_SET_IMPL_H
#define LIBTENSOR_CTF_TOD_SET_IMPL_H

#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_tod_set.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_set<N>::k_clazz[] = "ctf_tod_set<N>";


template<size_t N>
void ctf_tod_set<N>::perform(bool zero, ctf_dense_tensor_i<N, double> &ta) {

    size_t sz = ta.get_dims().get_size();

    ctf_dense_tensor_ctrl<N, double> ca(ta);
    tCTF_Tensor<double> &dta = ca.req_ctf_tensor();

    dta = m_v;
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_SET_IMPL_H
