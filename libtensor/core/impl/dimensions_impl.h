#ifndef LIBTENSOR_DIMENSIONS_IMPL_H
#define LIBTENSOR_DIMENSIONS_IMPL_H

#include "../dimensions.h"

namespace libtensor {


template<size_t N>
dimensions<N>::dimensions(const index_range<N> &ir) {

    const index<N> &i0 = ir.get_begin(), &i1 = ir.get_end();
    for(size_t i = 0; i < N; i++) {
        m_dims[i] = i1[i] - i0[i] + 1;
    }
    update_increments();
}


template<size_t N>
dimensions<N>::dimensions(const dimensions<N> &d) :

    m_dims(d.m_dims), m_incs(d.m_incs), m_size(d.m_size) {

}


template<size_t N>
bool dimensions<N>::contains(const index<N> &idx) const {

    for(size_t i = 0; i < N; i++) {
        if(idx[i] >= m_dims[i]) return false;
    }
    return true;
}


template<size_t N>
bool dimensions<N>::equals(const dimensions<N> &d) const {

    for(size_t i = 0; i < N; i++) {
        if(m_dims[i] != d.m_dims[i]) return false;
    }
    return true;
}


template<size_t N>
dimensions<N> &dimensions<N>::permute(const permutation<N> &p) {

    p.apply(m_dims);
    update_increments();
    return *this;
}


template<size_t N>
void dimensions<N>::update_increments() {

    size_t sz = 1;
    size_t i = N;
    while(i != 0) {
        i--;
        m_incs[i] = sz;
        sz *= m_dims[i];
    }
    m_size = sz;
}


} // namespace libtensor

#endif // LIBTENSOR_DIMENSIONS_IMPL_H
