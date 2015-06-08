#ifndef LIBTENSOR_CTF_TOD_SET_DIAG_IMPL_H
#define LIBTENSOR_CTF_TOD_SET_DIAG_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include <libtensor/core/mask.h>
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_tod_set_diag.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_set_diag<N>::k_clazz[] = "ctf_tod_set_diag<N>";


template<size_t N>
void ctf_tod_set_diag<N>::perform(
    bool zero, ctf_dense_tensor_i<N, double> &ta) {

    static const char method[] =
        "perform(bool, ctf_dense_tensor_i<N, double>&)";

    const dimensions<N> &dims = ta.get_dims();
    mask<N> mdone;

    for(size_t i = 0; i < N; i++) if(!mdone[i]) {
        size_t g = m_msk[i];
        if(g == 0) continue;
        for(size_t j = i; j < N; j++) {
            if(m_msk[j] != g) continue;
            if(dims[i] != dims[j]) {
                throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "t");
            }
            mdone[j] = true;
        }
    }
    for(size_t i = 0; i < N; i++) mdone[i] = false;

    char label[N + 1];
    for(size_t i = 0, k = 0; i < N; i++) if(!mdone[i]) {
        if(m_msk[i] == 0) {
            label[N - i - 1] = char(k) + 1;
            mdone[i] = true;
            k++;
        } else {
            size_t g = m_msk[i];
            for(size_t j = i; j < N; j++) if(m_msk[j] == g) {
                label[N - j - 1] = char(k) + 1;
                mdone[j] = true;
            }
            k++;
        }
    }
    label[N] = '\0';

    ctf_dense_tensor_ctrl<N, double> ca(ta);
    CTF::Tensor<double> &dta = ca.req_ctf_tensor();

    CTF::Scalar<> v(m_v, ctf::get_world());
    if(zero) dta[label] = v[""];
    else dta[label] += v[""];
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_SET_DIAG_IMPL_H
