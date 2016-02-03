#ifndef LIBTENSOR_CTF_TOD_DOTPROD_IMPL_H
#define LIBTENSOR_CTF_TOD_DOTPROD_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_error.h"
#include "../ctf_tod_dotprod.h"
#include "ctf_world.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_dotprod<N>::k_clazz[] = "ctf_tod_dotprod<N>";


template<size_t N>
ctf_tod_dotprod<N>::ctf_tod_dotprod(
    ctf_dense_tensor_i<N, double> &ta, ctf_dense_tensor_i<N, double> &tb) :

    m_ta(ta), m_tb(tb) {

    const char method[] = "ctf_tod_dotprod("
        "ctf_dense_tensor_i<N, double>&, ctf_dense_tensor_i<N, double>&)";

    if(!verify_dims()) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "ta,tb");
    }
}


template<size_t N>
ctf_tod_dotprod<N>::ctf_tod_dotprod(
    ctf_dense_tensor_i<N, double> &ta, const permutation<N> &perma,
    ctf_dense_tensor_i<N, double> &tb, const permutation<N> &permb) :

    m_ta(ta), m_tra(perma), m_tb(tb), m_trb(permb) {

    const char method[] = "ctf_tod_dotprod("
        "ctf_dense_tensor_i<N, double>&, const permutation<N>&, "
        "ctf_dense_tensor_i<N, double>&, const permutation<N>&)";

    if(!verify_dims()) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "ta,tb");
    }
}


template<size_t N>
ctf_tod_dotprod<N>::ctf_tod_dotprod(
    ctf_dense_tensor_i<N, double> &ta, const tensor_transf<N, double> &tra,
    ctf_dense_tensor_i<N, double> &tb, const tensor_transf<N, double> &trb) :

    m_ta(ta), m_tra(tra), m_tb(tb), m_trb(trb) {

    const char method[] = "ctf_tod_dotprod("
        "ctf_dense_tensor_i<N, double>&, const tensor_transf<N, double>&, "
        "ctf_dense_tensor_i<N, double>&, const tensor_transf<N, double>&)";

    if(!verify_dims()) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "ta,tb");
    }
}


template<size_t N>
double ctf_tod_dotprod<N>::calculate() {

    static const char method[] = "calculate()";

    ctf_dense_tensor_ctrl<N, double> ca(m_ta), cb(m_tb);

    const permutation<N> &perma = m_tra.get_perm();
    const permutation<N> &permb = m_trb.get_perm();

    double c = m_tra.get_scalar_tr().get_coeff() *
        m_trb.get_scalar_tr().get_coeff();

    permutation<N> perm(perma);
    perm.permute(permutation<N>(permb, true));

    sequence<N, size_t> seq(0);
    for(size_t i = 0; i < N; i++) seq[i] = N - i - 1;
    perm.apply(seq);
    char idxmapa[N], idxmapb[N], idxmapd[N];
    for(size_t i = 0; i < N; i++) idxmapa[i] = i + 1;
    for(size_t i = 0; i < N; i++) idxmapb[i] = seq[N - i - 1] + 1;
    for(size_t i = 0; i < N; i++) idxmapd[i] = N + 1;

    CTF::Scalar<double> dtd(0.0, ctf_world::get_world());

    const ctf_symmetry<N, double> &syma = ca.req_symmetry();
    const ctf_symmetry<N, double> &symb = cb.req_symmetry();
    for(size_t i = 0; i < syma.get_ncomp(); i++)
    for(size_t j = 0; j < symb.get_ncomp(); j++) {
        CTF::Tensor<double> &dta = ca.req_ctf_tensor(i);
        CTF::Tensor<double> &dtb = cb.req_ctf_tensor(j);
        dtd.contract(1.0, dta, idxmapa, dtb, idxmapb, 1.0, idxmapd);
    }
    return c * dtd.get_val();
}


template<size_t N>
bool ctf_tod_dotprod<N>::verify_dims() const {

    dimensions<N> dimsa(m_ta.get_dims()), dimsb(m_tb.get_dims());
    dimsa.permute(m_tra.get_perm());
    dimsb.permute(m_trb.get_perm());
    return dimsa.equals(dimsb);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_DOTPROD_IMPL_H

