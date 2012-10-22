#ifndef LIBTENSOR_DIAG_TO_CONTRACT2_SPACE_IMPL_H
#define LIBTENSOR_DIAG_TO_CONTRACT2_SPACE_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../diag_to_contract2_space.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *diag_to_contract2_space<N, M, K>::k_clazz =
    "diag_to_contract2_space<N, M, K>";


template<size_t N, size_t M, size_t K>
diag_to_contract2_space<N, M, K>::diag_to_contract2_space(
    const contraction2<N, M, K> &contr,
    const diag_tensor_space<N + K> &dtsa,
    const diag_tensor_space<M + K> &dtsb) :

    m_dimsc(contr, dtsa.get_dims(), dtsb.get_dims()),
    m_dtsc(m_dimsc.get_dims()) {

    std::vector<size_t> ssla, sslb;
    dtsa.get_all_subspaces(ssla);
    dtsb.get_all_subspaces(sslb);

    for(size_t i = 0; i < ssla.size(); i++) {
        for(size_t j = 0; j < sslb.size(); j++) {
            const diag_tensor_subspace<N + K> &ssa = dtsa.get_subspace(ssla[i]);
            const diag_tensor_subspace<M + K> &ssb = dtsb.get_subspace(sslb[j]);
            add_subspace(contr, ssa, ssb);
        }
    }
}


template<size_t N, size_t M, size_t K>
void diag_to_contract2_space<N, M, K>::add_subspace(
    const contraction2<N, M, K> &contr, const diag_tensor_subspace<N + K> &ssa,
    const diag_tensor_subspace<M + K> &ssb) {

    const sequence<2 * (N + M + K), size_t> &conn = contr.get_conn();

    size_t nda = ssa.get_ndiag(), ndb = ssb.get_ndiag(), ndc = 0;
    sequence< N + M, mask<N + M> > dc;

    //  Map i -> mask of diagonal
    sequence< N + K, mask<N + K> > mda;
    sequence< M + K, mask<M + K> > mdb;
    for(size_t i = 0; i < nda; i++) {
        const mask<N + K> &ma = ssa.get_diag_mask(i);
        for(size_t j = 0; j < N + K; j++) if(ma[j]) mda[j] = ma;
    }
    for(size_t i = 0; i < ndb; i++) {
        const mask<M + K> &mb = ssb.get_diag_mask(i);
        for(size_t j = 0; j < M + K; j++) if(mb[j]) mdb[j] = mb;
    }

    mask<N + K> tma; // Total mask in A to check for completion
    mask<M + K> tmb;

    //  First combine contracted diagonals together
    do {
        mask<N + K> ma0, ma;
        mask<M + K> mb;
        mask<N + M> mc0, mc;
        bool first = true;
        for(size_t ia = 0; ia < N + K; ia++) {
            if(conn[N + M + ia] < N + M) continue;
            if(tma[ia]) continue;
            size_t ib = conn[N + M + ia] - (N + K) - (N + M);
            if(first || ma[ia] || mb[ib]) {
                ma |= mda[ia];
                mb |= mdb[ib];
            }
            first = false;
        }
        //  We are done once all the contracted indexes have beed processed 
        if(ma.equals(ma0)) break;
        for(size_t ic = 0; ic < N + M; ic++) {
            if(conn[ic] < (N + K + N + M)) {
                size_t ia = conn[ic] - (N + M);
                mc[ic] = ma[ia];
            } else {
                size_t ib = conn[ic] - (N + K) - (N + M);
                mc[ic] = mb[ib];
            }
        }
        if(!mc.equals(mc0)) dc[ndc++] = mc;
        tma |= ma;
        tmb |= mb;
    } while(true);

    //  Then transfer uncontracted diagonals from A and B to C
    for(size_t ia = 0; ia < N + K; ia++) if(!tma[ia]) {
        mask<N + K> ma = mda[ia];
        mask<N + M> mc0, mc;
        for(size_t j = 0; j < N + K; j++) if(conn[N + M + j] < N + M) {
            mc[conn[N + M + j]] = ma[j];
        }
        if(!mc.equals(mc0)) dc[ndc++] = mc;
        tma |= ma;
    }
    for(size_t ib = 0; ib < M + K; ib++) if(!tmb[ib]) {
        mask<M + K> mb = mdb[ib];
        mask<N + M> mc0, mc;
        for(size_t j = 0; j < M + K; j++) if(conn[N + M + N + K + j] < N + M) {
            mc[conn[N + M + N + K + j]] = mb[j];
        }
        if(!mc.equals(mc0)) dc[ndc++] = mc;
        tmb |= mb;
    }

    diag_tensor_subspace<N + M> ssc(ndc);
    for(size_t i = 0; i < ndc; i++) ssc.set_diag_mask(i, dc[i]);
    if(!contains(m_dtsc, ssc)) m_dtsc.add_subspace(ssc);
}


template<size_t N, size_t M, size_t K>
bool diag_to_contract2_space<N, M, K>::contains(
    const diag_tensor_space<N + M> &dts,
    const diag_tensor_subspace<N + M> &ss) const {

    std::vector<size_t> ssl;
    dts.get_all_subspaces(ssl);
    for(size_t i = 0; i < ssl.size(); i++) {
        if(dts.get_subspace(ssl[i]).equals(ss)) return true;
    }
    return false;
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TO_CONTRACT2_SPACE_IMPL_H
