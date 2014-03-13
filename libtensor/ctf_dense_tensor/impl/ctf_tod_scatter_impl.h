#ifndef LIBTENSOR_CTF_TOD_SCATTER_IMPL_H
#define LIBTENSOR_CTF_TOD_SCATTER_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_error.h"
#include "../ctf_tod_scatter.h"

namespace libtensor {


template<size_t N, size_t M>
const char ctf_tod_scatter<N, M>::k_clazz[] = "ctf_tod_scatter<N, M>";


template<size_t N, size_t M>
ctf_tod_scatter<N, M>::ctf_tod_scatter(
    ctf_dense_tensor_i<NA, double> &ta,
    const tensor_transf<NC, double> &trc) :

    m_ta(ta), m_trc(trc) {

}


template<size_t N, size_t M>
void ctf_tod_scatter<N, M>::perform(
    bool zero,
    ctf_dense_tensor_i<NC, double> &tc) {

    static const char method[] =
        "perform(bool, ctf_dense_tensor_i<NC, double>&)";

    ctf_dense_tensor_ctrl<NA, double> ca(m_ta);
    ctf_dense_tensor_ctrl<NC, double> cc(tc);
    tCTF_Tensor<double> &dta = ca.req_ctf_tensor();
    tCTF_Tensor<double> &dtc = cc.req_ctf_tensor();

    sequence<NA, int> seqa(0);
    sequence<NC, int> seqc(0);
    for(size_t i = 0; i < NA; i++) seqc[M + i] = seqa[i] = int(i);
    for(size_t i = 0; i < M; i++) seqc[i] = int(NA + i);
    m_trc.get_perm().apply(seqc);

    char mapa[NA], mapc[NC];
    for(size_t i = 0; i < NA; i++) mapa[i] = seqa[NA - i - 1] + 1;
    for(size_t i = 0; i < NC; i++) mapc[i] = seqc[NC - i - 1] + 1;

    double kc = m_trc.get_scalar_tr().get_coeff();
    dtc.sum(kc, dta, mapa, zero ? 0.0 : 1.0, mapc);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_SCATTER_IMPL_H

