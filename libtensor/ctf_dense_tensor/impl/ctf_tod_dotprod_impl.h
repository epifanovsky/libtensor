#ifndef LIBTENSOR_CTF_TOD_DOTPROD_IMPL_H
#define LIBTENSOR_CTF_TOD_DOTPROD_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../ctf.h"
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_error.h"
#include "../ctf_tod_dotprod.h"

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
    int taid = ca.req_tensor_id(), tbid = cb.req_tensor_id();

    const permutation<N> &perma = m_tra.get_perm();
    const permutation<N> &permb = m_trb.get_perm();

    double c = m_tra.get_scalar_tr().get_coeff() *
        m_trb.get_scalar_tr().get_coeff();
    double d = 0.0;

    permutation<N> perm(perma);
    perm.permute(permutation<N>(permb, true));

    sequence<N, size_t> seq(0);
    for(size_t i = 0; i < N; i++) seq[i] = N - i - 1;
    perm.apply(seq);
    int idxmapa[N], idxmapb[N], idxmapd[N];
    for(size_t i = 0; i < N; i++) idxmapa[i] = i;
    for(size_t i = 0; i < N; i++) idxmapb[i] = seq[N - i - 1];
    for(size_t i = 0; i < N; i++) idxmapd[i] = N;

    int tdid, dimsd = 1, symd = NS;

    if(ctf::get().define_tensor(1, &dimsd, &symd, &tdid) !=
        DIST_TENSOR_SUCCESS) {
        throw ctf_error(g_ns, k_clazz, method, __FILE__, __LINE__,
            "define_tensor");
    }

    CTF_ctr_type_t contr;
    contr.tid_A = taid;
    contr.tid_B = tbid;
    contr.tid_C = tdid;
    contr.idx_map_A = idxmapa;
    contr.idx_map_B = idxmapb;
    contr.idx_map_C = idxmapd;

    if(ctf::get().contract(&contr, 1.0, 1.0) != DIST_TENSOR_SUCCESS) {
        throw ctf_error(g_ns, k_clazz, method, __FILE__, __LINE__, "contract");
    }

    int64_t len;
    double *data;

    if(ctf::get().allread_tensor(tdid, &len, &data) != DIST_TENSOR_SUCCESS) {
        throw ctf_error(g_ns, k_clazz, method, __FILE__, __LINE__,
            "allread_tensor");
    }

    d = data[0];

    ctf::get().clean_tensor(tdid);

    return c * d;
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

