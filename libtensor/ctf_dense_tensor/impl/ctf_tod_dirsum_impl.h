#ifndef LIBTENSOR_CTF_TOD_DIRSUM_IMPL_H
#define LIBTENSOR_CTF_TOD_DIRSUM_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include <libtensor/dense_tensor/to_dirsum_dims.h>
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_error.h"
#include "../ctf_tod_dirsum.h"

namespace libtensor {


template<size_t N, size_t M>
const char ctf_tod_dirsum<N, M>::k_clazz[] = "ctf_tod_dirsum<N, M>";


template<size_t N, size_t M>
ctf_tod_dirsum<N, M>::ctf_tod_dirsum(
    ctf_dense_tensor_i<NA, double> &ta,
    const scalar_transf<double> &ka,
    ctf_dense_tensor_i<NB, double> &tb,
    const scalar_transf<double> &kb,
    const tensor_transf<NC, double> &trc) :

    m_ta(ta), m_tb(tb), m_ka(ka.get_coeff()), m_kb(kb.get_coeff()),
    m_c(trc.get_scalar_tr().get_coeff()), m_permc(trc.get_perm()),
    m_dimsc(to_dirsum_dims<N, M>(ta.get_dims(), tb.get_dims(), m_permc).
        get_dimsc()) {

}


template<size_t N, size_t M>
void ctf_tod_dirsum<N, M>::perform(
    bool zero,
    ctf_dense_tensor_i<NC, double> &tc) {

    static const char method[] =
        "perform(bool, ctf_dense_tensor_i<NC, double>&)";

    if(!m_dimsc.equals(tc.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tc");
    }

    ctf_dense_tensor_ctrl<NA, double> ca(m_ta);
    ctf_dense_tensor_ctrl<NB, double> cb(m_tb);
    ctf_dense_tensor_ctrl<NC, double> cc(tc);
    tCTF_Tensor<double> &dta = ca.req_ctf_tensor();
    tCTF_Tensor<double> &dtb = cb.req_ctf_tensor();
    tCTF_Tensor<double> &dtc = cc.req_ctf_tensor();

    sequence<NA, int> seqa(0);
    sequence<NB, int> seqb(0);
    sequence<NC, int> seqc(0);
    for(size_t i = 0; i < NA; i++) seqc[i] = seqa[i] = int(i);
    for(size_t i = 0; i < NB; i++) seqc[NA + i] = seqb[i] = int(NA + i);
    m_permc.apply(seqc);

    char mapa[NA], mapb[NB], mapc[NC];
    for(size_t i = 0; i < NA; i++) mapa[i] = seqa[NA - i - 1] + 1;
    for(size_t i = 0; i < NB; i++) mapb[i] = seqb[NB - i - 1] + 1;
    for(size_t i = 0; i < NC; i++) mapc[i] = seqc[NC - i - 1] + 1;

    dtc.sum(m_c * m_ka, dta, mapa, zero ? 0.0 : 1.0, mapc);
    dtc.sum(m_c * m_kb, dtb, mapb, 1.0, mapc);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_DIRSUM_IMPL_H

