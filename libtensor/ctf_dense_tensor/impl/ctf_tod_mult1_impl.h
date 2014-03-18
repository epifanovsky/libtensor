#ifndef LIBTENSOR_CTF_TOD_MULT1_IMPL_H
#define LIBTENSOR_CTF_TOD_MULT1_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_error.h"
#include "../ctf_tod_mult1.h"
#include "ctf_fctr.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_mult1<N>::k_clazz[] = "ctf_tod_mult1<N>";


template<size_t N>
ctf_tod_mult1<N>::ctf_tod_mult1(
    ctf_dense_tensor_i<N, double> &tb,
    const tensor_transf<N, double> &trb,
    bool recip,
    const scalar_transf<double> &c) :

    m_tb(tb), m_permb(trb.get_perm()), m_c(c.get_coeff()), m_recip(recip) {

}


template<size_t N>
ctf_tod_mult1<N>::ctf_tod_mult1(
    ctf_dense_tensor_i<N, double> &tb,
    bool recip,
    double c) :

    m_tb(tb), m_c(c), m_recip(recip) {

}


template<size_t N>
ctf_tod_mult1<N>::ctf_tod_mult1(
    ctf_dense_tensor_i<N, double> &tb,
    const permutation<N> &p,
    bool recip,
    double c) :

    m_tb(tb), m_permb(p), m_c(c), m_recip(recip) {

}


template<size_t N>
void ctf_tod_mult1<N>::perform(bool zero, ctf_dense_tensor_i<N, double> &ta) {

    static const char method[] =
        "perform(bool, ctf_dense_tensor_i<N, double>&)";

    dimensions<N> dimsa(m_tb.get_dims());
    dimsa.permute(m_permb);
    if(!dimsa.equals(ta.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "ta");
    }

    ctf_dense_tensor_ctrl<N, double> ca(ta), cb(m_tb);
    tCTF_Tensor<double> &dta = ca.req_ctf_tensor();
    tCTF_Tensor<double> &dtb = cb.req_ctf_tensor();

    sequence<N, int> seqa(0), seqb(0);
    for(size_t i = 0; i < N; i++) seqa[i] = seqb[i] = i;
    permutation<N>(m_permb, true).apply(seqb);

    char mapa[N], mapb[N];
    for(size_t i = 0; i < N; i++) {
        mapa[i] = seqa[N - i - 1] + 1;
        mapb[i] = seqb[N - i - 1] + 1;
    }

    tCTF_fsum<double> op;
    if(m_recip) {
        if(zero) op.func_ptr = &ctf_fsum_ddiv;
        else op.func_ptr = &ctf_fsum_ddiv_add;
    } else {
        if(zero) op.func_ptr = &ctf_fsum_dmul;
        else op.func_ptr = &ctf_fsum_dmul_add;
    }
    dta.sum(m_c, dtb, mapb, 1.0, mapa, op);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_MULT1_IMPL_H

