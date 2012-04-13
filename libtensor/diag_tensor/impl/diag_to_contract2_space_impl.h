#ifndef LIBTENSOR_DIAG_TO_CONTRACT2_SPACE_IMPL_H
#define LIBTENSOR_DIAG_TO_CONTRACT2_SPACE_IMPL_H

#include <libtensor/tod/bad_dimensions.h>
#include "diag_to_contract2_subspace.h"
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
    m_dtsc(m_dimsc.get_dimsc()) {

    std::vector<size_t> ssla, sslb;
    dtsa.get_all_subspaces(ssla);
    dtsb.get_all_subspaces(sslb);

    for(size_t i = 0; i < ssla.size(); i++)
    for(size_t j = 0; j < sslb.size(); j++) {
        const diag_tensor_subspace<N + K> &ssa = dtsa.get_subspace(ssla[i]);
        const diag_tensor_subspace<M + K> &ssb = dtsb.get_subspace(sslb[j]);
        add_subspace(contr, ssa, ssb);
    }
}


template<size_t N, size_t M, size_t K>
void diag_to_contract2_space<N, M, K>::add_subspace(
    const contraction2<N, M, K> &contr, const diag_tensor_subspace<N + K> &ssa,
    const diag_tensor_subspace<M + K> &ssb) {

    diag_to_contract2_subspace<N, M, K> sscx(contr, ssa, ssb);
    const diag_tensor_subspace<N + M> &ssc = sscx.get_dtssc();

    bool contains = false;
    std::vector<size_t> sslc;
    m_dtsc.get_all_subspaces(sslc);
    for(size_t i = 0; i < sslc.size(); i++) {
        if(m_dtsc.get_subspace(sslc[i]).equals(ssc)) {
            contains = true;
            break;
        }
    }
    if(!contains) m_dtsc.add_subspace(ssc);
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TO_CONTRACT2_SPACE_IMPL_H
