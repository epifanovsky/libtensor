#ifndef LIBTENSOR_DIAG_TO_CONTRACT2_SUBSPACE_H
#define LIBTENSOR_DIAG_TO_CONTRACT2_SUBSPACE_H

#include <libtensor/core/contraction2.h>
#include <libtensor/core/bad_dimensions.h>
#include "../diag_tensor_space.h"

namespace libtensor {


/** \brief Forms the subspace of the result of contraction from two diagonal
        tensor subspaces
    \tparam N Order of first tensor less contraction order.
    \tparam M Order of second tensor less contraction order.
    \tparam K Order of contraction.

    \ingroup libtensor_diag_tensor
 **/
template<size_t N, size_t M, size_t K>
class diag_to_contract2_subspace {
public:
    static const char *k_clazz; //!< Class name

private:
    diag_tensor_subspace<N + M> m_dtssc; //!< Subspace of result

public:
    /** \brief Builds the result subspace
     **/
    diag_to_contract2_subspace(const contraction2<N, M, K> &contr,
        const diag_tensor_subspace<N + K> &dtssa,
        const diag_tensor_subspace<M + K> &dtssb);

    /** \brief Returns the result subspace
     **/
    const diag_tensor_subspace<N + M> &get_dtssc() const {
        return m_dtssc;
    }

};


template<size_t N, size_t M, size_t K>
const char *diag_to_contract2_subspace<N, M, K>::k_clazz =
    "diag_to_contract2_subspace<N, M, K>";


template<size_t N, size_t M, size_t K>
diag_to_contract2_subspace<N, M, K>::diag_to_contract2_subspace(
    const contraction2<N, M, K> &contr,
    const diag_tensor_subspace<N + K> &dtssa,
    const diag_tensor_subspace<M + K> &dtssb) : m_dtssc(N + M) {

    const sequence<2 * (N + M + K), size_t> &conn = contr.get_conn();

    size_t nda = dtssa.get_ndiag(), ndb = dtssb.get_ndiag(), ndc = 0;
    sequence< N + M, mask<N + M> > dc;

    //  Map i -> mask of diagonal
    sequence< N + K, mask<N + K> > mda;
    sequence< M + K, mask<M + K> > mdb;
    for(size_t i = 0; i < nda; i++) {
        const mask<N + K> &ma = dtssa.get_diag_mask(i);
        for(size_t j = 0; j < N + K; j++) if(ma[j]) mda[j] = ma;
    }
    for(size_t i = 0; i < ndb; i++) {
        const mask<M + K> &mb = dtssb.get_diag_mask(i);
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

    for(size_t i = 0; i < ndc; i++) m_dtssc.set_diag_mask(i, dc[i]);
    m_dtssc.simplify();
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TO_CONTRACT2_SUBSPACE_H
