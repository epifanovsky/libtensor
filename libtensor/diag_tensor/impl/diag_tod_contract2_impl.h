#ifndef LIBTENSOR_DIAG_TOD_CONTRACT2_IMPL_H
#define LIBTENSOR_DIAG_TOD_CONTRACT2_IMPL_H

#include <map>
#include "../diag_tensor_ctrl.h"
#include "../diag_tod_set.h"
#include "diag_to_contract2_subspace.h"
#include "diag_tod_contract2_part.h"
#include "../diag_tod_contract2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *diag_tod_contract2<N, M, K>::k_clazz =
    "diag_tod_contract2<N, M, K>";


namespace diag_tod_contract2_ns {

struct schrec {
    size_t ssa, ssb;
    double d;
};

} // namespace diag_tod_contract2_ns
using diag_tod_contract2_ns::schrec;


template<size_t N, size_t M, size_t K>
void diag_tod_contract2<N, M, K>::perform(
    diag_tensor_wr_i<N + M, double> &dtc) {

    double d = 1.0;

    diag_tod_set<N + M>().perform(dtc);

    const diag_tensor_space<N + K> &dtsa = m_dta.get_space();
    const diag_tensor_space<M + K> &dtsb = m_dtb.get_space();
    const diag_tensor_space<N + M> &dtsc = dtc.get_space();

    std::vector<size_t> ssla, sslb, sslc;
    dtsa.get_all_subspaces(ssla);
    dtsb.get_all_subspaces(sslb);
    dtsc.get_all_subspaces(sslc);

    diag_tensor_rd_ctrl<N + K, double> ca(m_dta);
    diag_tensor_rd_ctrl<M + K, double> cb(m_dtb);
    diag_tensor_wr_ctrl<N + M, double> cc(dtc);

    std::multimap<size_t, schrec> sch;

    for(size_t isa = 0; isa < ssla.size(); isa++)
    for(size_t isb = 0; isb < sslb.size(); isb++) {
        diag_to_contract2_subspace<N, M, K> sscx(m_contr,
            dtsa.get_subspace(ssla[isa]), dtsb.get_subspace(sslb[isb]));
        bool found_exact = false;
        for(size_t isc = 0; isc < sslc.size(); isc++) {
            if(dtsc.get_subspace(sslc[isc]).equals(sscx.get_dtssc())) {
                schrec r;
                r.ssa = ssla[isa];
                r.ssb = sslb[isb];
                r.d = d;
                sch.insert(std::pair<size_t, schrec>(sslc[isc], r));
                found_exact = true;
                break;
            }
        }
        if(found_exact) continue;
        // ...
        throw 0;
    }

    for(std::multimap<size_t, schrec>::const_iterator i = sch.begin();
        i != sch.end(); ++i) {

        double *pc = cc.req_dataptr(i->first);
        const double *pa = ca.req_const_dataptr(i->second.ssa);
        const double *pb = cb.req_const_dataptr(i->second.ssb);
        diag_tod_contract2_part<N, M, K>(m_contr,
            dtsa.get_subspace(i->second.ssa), dtsa.get_dims(), pa,
            dtsb.get_subspace(i->second.ssb), dtsb.get_dims(), pb).
            perform(dtsc.get_subspace(i->first), dtsc.get_dims(), pc,
            i->second.d);
        cb.ret_const_dataptr(i->second.ssb, pb);
        ca.ret_const_dataptr(i->second.ssa, pa);
        cc.ret_dataptr(i->first, pc);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_CONTRACT2_IMPL_H
