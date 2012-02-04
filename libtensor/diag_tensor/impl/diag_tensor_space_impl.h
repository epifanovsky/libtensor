#ifndef LIBTENSOR_DIAG_TENSOR_SPACE_IMPL_H
#define LIBTENSOR_DIAG_TENSOR_SPACE_IMPL_H

#include "../diag_tensor_space.h"

namespace libtensor {


template<size_t N>
diag_tensor_subspace<N>::diag_tensor_subspace(size_t n) :

    m_diag(n, mask<N>()) {

}


template<size_t N>
const mask<N> &diag_tensor_subspace<N>::get_diag_mask(size_t n) const {

#ifdef LIBTENSOR_DEBUG
    if(n >= m_diag.size()) throw 0;
#endif // LIBTENSOR_DEBUG

    return m_diag[n];
}


template<size_t N>
void diag_tensor_subspace<N>::set_diag_mask(size_t n, const mask<N> &msk) {

#ifdef LIBTENSOR_DEBUG
    mask<N> msk0;
    for(size_t i = 0; i < m_diag.size(); i++) {
        mask<N> m(msk);
        m &= m_diag[i];
        if(!m.equals(msk0)) throw 0;
    }
#endif // LIBTENSOR_DEBUG

    m_diag[n] = msk;

    mask<N> totmsk;
    for(size_t i = 0; i < m_diag.size(); i++) totmsk |= m_diag[i];
    m_msk = totmsk;
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TENSOR_SPACE_IMPL_H

