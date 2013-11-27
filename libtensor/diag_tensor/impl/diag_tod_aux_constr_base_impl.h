#ifndef LIBTENSOR_DIAG_TOD_AUX_CONSTR_BASE_IMPL_H
#define LIBTENSOR_DIAG_TOD_AUX_CONSTR_BASE_IMPL_H

#include "diag_tod_aux_constr_base.h"

namespace libtensor {


template<size_t N>
void diag_tod_aux_constr_base<N>::mark_diags(
    const mask<N> &m0,
    const diag_tensor_subspace<N> &ss,
    mask<N> &m1) const {

    //  Input: one or more bits set in m0
    //  Output: for each bit set in m0, the respective diagonal is marked in m1

    size_t ndiag = ss.get_ndiag();
    for(size_t i = 0; i < ndiag; i++) {
        const mask<N> &m = ss.get_diag_mask(i);
        for(size_t j = 0; j < N; j++) if(m0[j] && m[j]) {
            m1 |= m;
            break;
        }
    }
    m1 |= m0;
}


template<size_t N>
size_t diag_tod_aux_constr_base<N>::get_increment(
    const dimensions<N> &dims,
    const diag_tensor_subspace<N> &ss,
    const mask<N> &m) const {

    //  Build new dimensions in which only the primary index
    //  of each diagonal exists

    index<N> i1, i2;
    mask<N> mm;

    size_t ndiag = ss.get_ndiag(); // Total number of diagonals
    const mask<N> &totm = ss.get_total_mask();
    for(size_t i = 0; i < ndiag; i++) {
        const mask<N> &dm = ss.get_diag_mask(i);
        for(size_t j = 0; j < N; j++) if(dm[j]) {
            i2[j] = dims[j] - 1;
            if(m[j]) mm[j] = true;
            break;
        }
    }
    for(size_t j = 0; j < N; j++) if(!totm[j]) {
        i2[j] = dims[j] - 1;
        mm[j] = m[j];
    }

    dimensions<N> dims2(index_range<N>(i1, i2));

    //  Now compute and return increment

    size_t inc = 0;
    for(size_t j = 0; j < N; j++) if(mm[j]) inc += dims2.get_increment(j);
    return inc;
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_AUX_CONSTR_BASE_IMPL_H
