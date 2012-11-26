#ifndef LIBTENSOR_ABS_INDEX_IMPL_H
#define LIBTENSOR_ABS_INDEX_IMPL_H

#include <libtensor/core/out_of_bounds.h>
#include "magic_dimensions_impl.h"
#include "../abs_index.h"

namespace libtensor {


template<size_t N>
const char *abs_index<N>::k_clazz = "abs_index<N>";


template<size_t N>
abs_index<N>::abs_index(const dimensions<N> &dims) :

    m_dims(dims), m_aidx(0) {

}


template<size_t N>
abs_index<N>::abs_index(const index<N> &idx, const dimensions<N> &dims) :

    m_dims(dims), m_idx(idx) {

    m_aidx = get_abs_index(m_idx, m_dims);
}


template<size_t N>
abs_index<N>::abs_index(size_t aidx, const dimensions<N> &dims) :

    m_dims(dims), m_aidx(aidx) {

    get_index(m_aidx, m_dims, m_idx);
}


template<size_t N>
abs_index<N>::abs_index(const abs_index<N> &other) :

    m_dims(other.m_dims), m_idx(other.m_idx), m_aidx(other.m_aidx) {

}


template<size_t N>
bool abs_index<N>::inc() {

    if(is_last()) return false;

    size_t n = N - 1;
    bool done = false, ok = false;
    do {
        if(m_idx[n] < m_dims[n] - 1) {
            m_idx[n]++;
            for(register size_t i = n + 1; i < N; i++) m_idx[i] = 0;
            done = true;
            ok = true;
        } else {
            if(n == 0) done = true;
            else n--;
        }
    } while(!done);
    if(ok) m_aidx++;
    return ok;
}


template<size_t N>
size_t abs_index<N>::get_abs_index(const index<N> &idx,
    const dimensions<N> &dims) {

#ifdef LIBTENSOR_DEBUG
    static const char *method =
        "get_abs_index(const index<N>&, const dimensions<N>&)";

    for(size_t i = 0; i < N; i++) {
        if(idx[i] >= dims[i]) {
            throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
                "idx");
        }
    }
#endif // LIBTENSOR_DEBUG

    size_t aidx = 0;
    for(register size_t i = 0; i < N; i++) {
        aidx += dims.get_increment(i) * idx[i];
    }

    return aidx;
}


template<size_t N>
void abs_index<N>::get_index(size_t aidx, const dimensions<N> &dims,
    index<N> &idx) {

#ifdef LIBTENSOR_DEBUG
    static const char *method =
        "get_index(size_t, const dimensions<N>&, index<N>&)";

    if(aidx >= dims.get_size()) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__, "aidx");
    }
#endif // LIBTENSOR_DEBUG

    size_t a = aidx;
    size_t imax = N - 1;
    for(register size_t i = 0; i < imax; i++) {
        idx[i] = a / dims.get_increment(i);
        a %= dims.get_increment(i);
    }
    idx[N - 1] = a;
}


template<size_t N>
void abs_index<N>::get_index(size_t aidx, const magic_dimensions<N> &mdims,
    index<N> &idx) {

#ifdef LIBTENSOR_DEBUG
    static const char *method =
        "get_index(size_t, const dimensions<N>&, index<N>&)";

    if(aidx >= mdims.get_dims().get_size()) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__, "aidx");
    }
#endif // LIBTENSOR_DEBUG

    uint64_t a = aidx;
    for(register size_t i = 0; i < N - 1; i++) {
        idx[i] = mdims.divide(a, i);
        a -= idx[i] * mdims.get_dims().get_increment(i);
    }
    idx[N - 1] = a;
}


} // namespace libtensor

#endif // LIBTENSOR_ABS_INDEX_IMPL_H
