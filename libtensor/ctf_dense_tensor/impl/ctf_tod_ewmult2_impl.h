#ifndef LIBTENSOR_CTF_TOD_EWMULT2_IMPL
#define LIBTENSOR_CTF_TOD_EWMULT2_IMPL

#include <libtensor/core/bad_dimensions.h>
#include <libtensor/dense_tensor/to_ewmult2_dims.h>
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_error.h"
#include "../ctf_tod_ewmult2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char ctf_tod_ewmult2<N, M, K>::k_clazz[] = "ctf_tod_ewmult2<N, M, K>";


template<size_t N, size_t M, size_t K>
ctf_tod_ewmult2<N, M, K>::ctf_tod_ewmult2(
    ctf_dense_tensor_i<NA, double> &ta,
    const tensor_transf<NA, double> &tra,
    ctf_dense_tensor_i<NB, double> &tb,
    const tensor_transf<NB, double> &trb,
    const tensor_transf<NC, double> &trc) :

    m_ta(ta), m_tra(tra), m_tb(tb), m_trb(trb), m_trc(trc),
    m_dimsc(to_ewmult2_dims<N, M, K>(m_ta.get_dims(), m_tra.get_perm(),
        m_tb.get_dims(), m_trb.get_perm(), m_trc.get_perm()).get_dimsc()) {

}


template<size_t N, size_t M, size_t K>
void ctf_tod_ewmult2<N, M, K>::perform(
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
    CTF::Tensor<double> &dta = ca.req_ctf_tensor();
    CTF::Tensor<double> &dtb = cb.req_ctf_tensor();
    CTF::Tensor<double> &dtc = cc.req_ctf_tensor();

    sequence<NA, int> seqa(0);
    sequence<NB, int> seqb(0);
    sequence<NC, int> seqc(0);
    for(size_t i = 0; i != N; i++) seqa[i] = int(i);
    for(size_t i = 0; i != M; i++) seqb[i] = int(N + i);
    for(size_t i = 0; i != K; i++) seqa[N + i] = seqb[M + i] = int(N + M + i);
    for(size_t i = 0; i != NC; i++) seqc[i] = int(i);
    permutation<NA>(m_tra.get_perm(), true).apply(seqa);
    permutation<NB>(m_trb.get_perm(), true).apply(seqb);
    m_trc.get_perm().apply(seqc);

    char mapa[NA], mapb[NB], mapc[NC];
    for(size_t i = 0; i < NA; i++) mapa[i] = seqa[NA - i - 1] + 1;
    for(size_t i = 0; i < NB; i++) mapb[i] = seqb[NB - i - 1] + 1;
    for(size_t i = 0; i < NC; i++) mapc[i] = seqc[NC - i - 1] + 1;

    double d = m_tra.get_scalar_tr().get_coeff() *
        m_trb.get_scalar_tr().get_coeff() * m_trc.get_scalar_tr().get_coeff();

    dtc.contract(d, dta, mapa, dtb, mapb, zero ? 0.0 : 1.0, mapc);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_EWMULT2_IMPL

