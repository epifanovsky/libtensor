#ifndef LIBTENSOR_CTF_TOD_CONTRACT2_IMPL
#define LIBTENSOR_CTF_TOD_CONTRACT2_IMPL

#include <libtensor/core/bad_dimensions.h>
#include "../ctf.h"
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_error.h"
#include "../ctf_tod_contract2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char ctf_tod_contract2<N, M, K>::k_clazz[] =
    "ctf_tod_contract2<N, M, K>";

template<size_t N, size_t M, size_t K>
ctf_tod_contract2<N, M, K>::ctf_tod_contract2(
    const contraction2<N, M, K> &contr,
    ctf_dense_tensor_i<NA, double> &ta,
    ctf_dense_tensor_i<NB, double> &tb,
    double d) :

    m_contr(contr), m_ta(ta), m_tb(tb),
    m_dimsc(contr, ta.get_dims(), tb.get_dims()), m_d(d) {

}


template<size_t N, size_t M, size_t K>
void ctf_tod_contract2<N, M, K>::perform(
    bool zero,
    ctf_dense_tensor_i<NC, double> &tc) {

    static const char method[] =
        "perform(bool, ctf_dense_tensor_i<NC, double>&)";

    if(!m_dimsc.get_dims().equals(tc.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tc");
    }

    ctf_dense_tensor_ctrl<NA, double> ca(m_ta);
    ctf_dense_tensor_ctrl<NB, double> cb(m_tb);
    ctf_dense_tensor_ctrl<NC, double> cc(tc);
    int taid = ca.req_tensor_id(), tbid = cb.req_tensor_id(),
        tcid = cc.req_tensor_id();

    int map[NC + NA + NB];
    const sequence<NA + NB + NC, size_t> &conn = m_contr.get_conn();
    for(size_t i = 0; i < NC; i++) {
        map[i] = i;
        map[conn[i]] = i;
    }
    for(size_t i = 0, j = NC; i < NA; i++) {
        size_t ii = NC + i;
        if(conn[ii] >= NC + NA) {
            map[ii] = j;
            map[conn[ii]] = j;
            j++;
        }
    }

    if(zero) {
        if(ctf::get().set_zero_tensor(tcid) != DIST_TENSOR_SUCCESS) {
            throw ctf_error(g_ns, k_clazz, method, __FILE__, __LINE__,
                "set_zero_tensor");
        }
    }

    CTF_ctr_type_t ctrtyp;
    ctrtyp.tid_A = taid;
    ctrtyp.tid_B = tbid;
    ctrtyp.tid_C = tcid;
    ctrtyp.idx_map_A = &map[NC];
    ctrtyp.idx_map_B = &map[NC + NA];
    ctrtyp.idx_map_C = &map[0];
    if(ctf::get().contract(&ctrtyp, m_d, 1.0) != DIST_TENSOR_SUCCESS) {
        throw ctf_error(g_ns, k_clazz, method, __FILE__, __LINE__, "contract");
    }
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_CONTRACT2_IMPL

