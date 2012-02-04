#ifndef LIBTENSOR_DIAG_TENSOR_SPACE_IMPL_H
#define LIBTENSOR_DIAG_TENSOR_SPACE_IMPL_H

#include <libtensor/exception.h>
#include "../diag_tensor_space.h"

namespace libtensor {


template<size_t N>
const char *diag_tensor_subspace<N>::k_clazz = "diag_tensor_subspace<N>";


template<size_t N>
diag_tensor_subspace<N>::diag_tensor_subspace(size_t n) :

    m_diag(n, mask<N>()) {

    static const char *method = "diag_tensor_subspace(size_t)";

#ifdef LIBTENSOR_DEBUG
    if(n > N) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "n");
    }
#endif // LIBTENSOR_DEBUG

}


template<size_t N>
const mask<N> &diag_tensor_subspace<N>::get_diag_mask(size_t n) const {

    static const char *method = "get_diag_mask(size_t)";

#ifdef LIBTENSOR_DEBUG
    if(n >= m_diag.size()) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__, "n");
    }
#endif // LIBTENSOR_DEBUG

    return m_diag[n];
}


template<size_t N>
void diag_tensor_subspace<N>::set_diag_mask(size_t n, const mask<N> &msk) {

    static const char *method = "set_diag_mask(size_t, const mask<N>&)";

#ifdef LIBTENSOR_DEBUG
    if(n >= m_diag.size()) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__, "n");
    }

    // Check that there is no overlap with existing masks
    // except the one being replaced
    mask<N> msk0;
    for(size_t i = 0; i < m_diag.size(); i++) {
        if(i == n) continue;
        mask<N> m(msk);
        m &= m_diag[i];
        if(!m.equals(msk0)) {
            throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "msk");
        }
    }
#endif // LIBTENSOR_DEBUG

    m_diag[n] = msk;

    mask<N> totmsk;
    for(size_t i = 0; i < m_diag.size(); i++) totmsk |= m_diag[i];
    m_msk = totmsk;
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TENSOR_SPACE_IMPL_H

