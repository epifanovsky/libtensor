#ifndef LIBTENSOR_CTF_TOD_MULT_IMPL_H
#define LIBTENSOR_CTF_TOD_MULT_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../ctf_dense_tensor.h"
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_error.h"
#include "../ctf_tod_copy.h"
#include "../ctf_tod_mult.h"
#include "ctf_fctr.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_mult<N>::k_clazz[] = "ctf_tod_mult<N>";


template<size_t N>
ctf_tod_mult<N>::ctf_tod_mult(
    ctf_dense_tensor_i<N, double> &ta,
    const tensor_transf<N, double> &tra,
    ctf_dense_tensor_i<N, double> &tb,
    const tensor_transf<N, double> &trb,
    bool recip,
    const scalar_transf<double> &trc) :

    m_ta(ta), m_tra(tra), m_tb(tb), m_trb(trb), m_recip(recip),
    m_c(trc.get_coeff()), m_dimsc(m_ta.get_dims()) {

    static const char method[] = "ctf_tod_mult()";

    m_dimsc.permute(m_tra.get_perm());

    dimensions<N> dimsc1(m_tb.get_dims());
    dimsc1.permute(m_trb.get_perm());
    if(!dimsc1.equals(m_dimsc)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }
}


template<size_t N>
ctf_tod_mult<N>::ctf_tod_mult(
    ctf_dense_tensor_i<N, double> &ta,
    ctf_dense_tensor_i<N, double> &tb,
    bool recip, double c) :

    m_ta(ta), m_tb(tb), m_recip(recip), m_c(c), m_dimsc(m_ta.get_dims()) {

}


template<size_t N>
ctf_tod_mult<N>::ctf_tod_mult(
    ctf_dense_tensor_i<N, double> &ta,
    const permutation<N> &pa,
    ctf_dense_tensor_i<N, double> &tb,
    const permutation<N> &pb,
    bool recip, double c) :

    m_ta(ta), m_tra(pa, scalar_transf<double>()),
    m_tb(tb), m_trb(pb, scalar_transf<double>()),
    m_recip(recip), m_c(c), m_dimsc(m_ta.get_dims()) {

    static const char method[] = "ctf_tod_mult()";

    m_dimsc.permute(m_tra.get_perm());

    dimensions<N> dimsc1(m_tb.get_dims());
    dimsc1.permute(m_trb.get_perm());
    if(!dimsc1.equals(m_dimsc)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }
}


template<size_t N>
void ctf_tod_mult<N>::perform(bool zero, ctf_dense_tensor_i<N, double> &tc) {

    static const char method[] =
        "perform(bool, ctf_dense_tensor_i<N, double>&)";

    if(!m_dimsc.equals(tc.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tc");
    }

    ctf_dense_tensor<N, double> tta(m_ta.get_dims());
    ctf_dense_tensor<N, double> ttb(m_tb.get_dims());
    ctf_dense_tensor<N, double> ttc(tc.get_dims());
    ctf_tod_copy<N>(m_ta).perform(true, tta);
    ctf_tod_copy<N>(m_tb).perform(true, ttb);

    ctf_dense_tensor_ctrl<N, double> cc(tc);
    ctf_dense_tensor_ctrl<N, double> cta(tta), ctb(ttb), ctc(ttc);

    double c = m_c * m_tra.get_scalar_tr().get_coeff() *
        m_trb.get_scalar_tr().get_coeff();

    sequence<N, int> seqa(0), seqb(0), seqc(0);
    for(size_t i = 0; i < N; i++) seqa[i] = seqb[i] = seqc[i] = i;
    permutation<N>(m_tra.get_perm(), true).apply(seqa);
    permutation<N>(m_trb.get_perm(), true).apply(seqb);

    char mapa[N], mapb[N], mapc[N];
    for(size_t i = 0; i < N; i++) {
        mapa[i] = seqa[N - i - 1] + 1;
        mapb[i] = seqb[N - i - 1] + 1;
        mapc[i] = seqc[N - i - 1] + 1;
    }

    CTF::Tensor<double> &dta = cta.req_ctf_tensor(0);
    CTF::Tensor<double> &dtb = ctb.req_ctf_tensor(0);
    CTF::Tensor<double> &dtc = ctc.req_ctf_tensor(0);

    if(m_recip) {
        CTF::Bivar_Function<double> op(&ctf_fctr_ddiv);
        dtc.contract(c, dta, mapa, dtb, mapb, 0.0, mapc, op);
    } else {
        dtc.contract(c, dta, mapa, dtb, mapb, 0.0, mapc);
    }

    ctc.adjust_symmetry(cc.req_symmetry());
    ctf_tod_copy<N>(ttc).perform(zero, tc);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_MULT_IMPL_H

