#include <libtensor/core/bad_dimensions.h>
#include "../ctf.h"
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_error.h"
#include "../ctf_tod_copy.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_copy<N>::k_clazz[] = "ctf_tod_copy<N>";


template<size_t N>
ctf_tod_copy<N>::ctf_tod_copy(ctf_dense_tensor_i<N, double> &ta, double c) :

    m_ta(ta), m_tra(permutation<N>(), scalar_transf<double>(c)),
    m_dimsb(m_ta.get_dims()) {

}


template<size_t N>
ctf_tod_copy<N>::ctf_tod_copy(ctf_dense_tensor_i<N, double> &ta,
    const permutation<N> &perma, double c) :

    m_ta(ta), m_tra(perma, scalar_transf<double>(c)),
    m_dimsb(m_ta.get_dims()) {

    m_dimsb.permute(perma);
}


template<size_t N>
ctf_tod_copy<N>::ctf_tod_copy(ctf_dense_tensor_i<N, double> &ta,
    const tensor_transf<N, double> &tra) :

    m_ta(ta), m_tra(tra), m_dimsb(m_ta.get_dims()) {

    m_dimsb.permute(m_tra.get_perm());
}


template<size_t N>
void ctf_tod_copy<N>::perform(bool zero, ctf_dense_tensor_i<N, double> &tb) {

    static const char method[] =
        "perform(bool, ctf_dense_tensor_i<N, double>&)";

    if(!m_dimsb.equals(tb.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }

    ctf_dense_tensor_ctrl<N, double> ca(m_ta), cb(tb);
    int taid = ca.req_tensor_id(), tbid = cb.req_tensor_id();

    double c = m_tra.get_scalar_tr().get_coeff();

    sequence<N, int> seqa(0), seqb(0);
    int mapa[N], mapb[N];
    for(size_t i = 0; i < N; i++) seqa[i] = seqb[i] = i;
    m_tra.get_perm().apply(seqa);
    for(size_t i = 0; i < N; i++) {
        mapa[i] = seqa[i];
        mapb[i] = seqb[i];
    }

    if(zero) {
        if(ctf::get().set_zero_tensor(tbid) != DIST_TENSOR_SUCCESS) {
            throw ctf_error(g_ns, k_clazz, method, __FILE__, __LINE__,
                "set_zero_tensor");
        }
    }

    CTF_sum_type_t sumtyp;
    sumtyp.tid_A = taid;
    sumtyp.tid_B = tbid;
    sumtyp.idx_map_A = mapa;
    sumtyp.idx_map_B = mapb;
    if(ctf::get().sum_tensors(&sumtyp, c, 1.0) != DIST_TENSOR_SUCCESS) {
        throw ctf_error(g_ns, k_clazz, method, __FILE__, __LINE__,
            "sum_tensors");
    }
}


} // namespace libtensor

