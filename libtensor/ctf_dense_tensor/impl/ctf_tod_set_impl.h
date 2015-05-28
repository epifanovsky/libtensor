#ifndef LIBTENSOR_CTF_TOD_SET_IMPL_H
#define LIBTENSOR_CTF_TOD_SET_IMPL_H

#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_tod_set.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_set<N>::k_clazz[] = "ctf_tod_set<N>";


namespace {

struct ctf_func_shift {
    double d;
    double operator()(double a) const { return a - d; }
};
    
} // unnamed namespace


template<size_t N>
void ctf_tod_set<N>::perform(bool zero, ctf_dense_tensor_i<N, double> &ta) {

    size_t sz = ta.get_dims().get_size();

    ctf_dense_tensor_ctrl<N, double> ca(ta);
    tCTF_Tensor<double> &dta = ca.req_ctf_tensor();

    if(zero) {
        dta = m_v;
    } else {
        char ij[N];
        for(size_t i = 0; i < N; i++) ij[i] = char(i) + 1;
        CTF::Scalar<> v(m_v, ctf::get_world());
        dta[ij] += v[""];
    }
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_SET_IMPL_H
