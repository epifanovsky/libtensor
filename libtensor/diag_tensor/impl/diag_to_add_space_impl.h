#ifndef LIBTENSOR_DIAG_TO_ADD_SPACE_IMPL_H
#define LIBTENSOR_DIAG_TO_ADD_SPACE_IMPL_H

#include <libtensor/tod/bad_dimensions.h>
#include "../diag_to_add_space.h"

namespace libtensor {


template<size_t N>
const char *diag_to_add_space<N>::k_clazz = "diag_to_add_space<N>";


template<size_t N>
diag_to_add_space<N>::diag_to_add_space(const diag_tensor_space<N> &dtsa,
    const diag_tensor_space<N> &dtsb) : m_dtsc(dtsa.get_dims()) {

    static const char  *method = "diag_to_add_space()";

    if(!dtsa.get_dims().equals(dtsb.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "a,b");
    }

    std::vector<size_t> ssla;
    dtsa.get_all_subspaces(ssla);
    for(size_t i = 0; i < ssla.size(); i++) {
        const diag_tensor_subspace<N> &ss = dtsa.get_subspace(ssla[i]);
        if(!contains(m_dtsc, ss)) m_dtsc.add_subspace(ss);
    }

    std::vector<size_t> sslb;
    dtsb.get_all_subspaces(sslb);
    for(size_t i = 0; i < sslb.size(); i++) {
        const diag_tensor_subspace<N> &ss = dtsb.get_subspace(sslb[i]);
        if(!contains(m_dtsc, ss)) m_dtsc.add_subspace(ss);
    }
}


template<size_t N>
bool diag_to_add_space<N>::contains(const diag_tensor_space<N> &dts,
    const diag_tensor_subspace<N> &ss) const {

    std::vector<size_t> ssl;
    dts.get_all_subspaces(ssl);
    for(size_t i = 0; i < ssl.size(); i++) {
        if(dts.get_subspace(ssl[i]).equals(ss)) return true;
    }
    return false;
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TO_ADD_SPACE_IMPL_H
