#ifndef LIBTENSOR_CTF_TOD_CONTRACT2_IMPL
#define LIBTENSOR_CTF_TOD_CONTRACT2_IMPL

#include <libtensor/core/bad_dimensions.h>
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
    tCTF_Tensor<double> &dta = ca.req_ctf_tensor();
    tCTF_Tensor<double> &dtb = cb.req_ctf_tensor();
    tCTF_Tensor<double> &dtc = cc.req_ctf_tensor();

    char map[NC + NA + NB];
    const sequence<NA + NB + NC, size_t> &conn = m_contr.get_conn();
    for(size_t i = 0; i < NC; i++) {
        map[i] = i + 1;
        size_t ii = conn[NC - i - 1] - NC;
        if(ii < NA) {
            ii = NA - ii - 1;
            map[NC + ii] = i + 1;
        } else {
            ii = NB + NA - ii - 1;
            map[NC + NA + ii] = i + 1;
        }
    }
    for(size_t i = 0, j = NC; i < NA; i++) {
        size_t ii = NC + i;
        if(conn[ii] >= NC + NA) {
            size_t ia = NA - i - 1;
            size_t ib = NC + NA + NB - conn[ii] - 1;
            map[NC + ia] = j + 1;
            map[NC + NA + ib] = j + 1;
            j++;
        }
    }

    dtc.contract(m_d, dta, &map[NC], dtb, &map[NC + NA], zero ? 0.0 : 1.0,
        &map[0]);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_CONTRACT2_IMPL

