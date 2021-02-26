#ifndef LIBTENSOR_MAGIC_DIMENSIONS_IMPL_H
#define LIBTENSOR_MAGIC_DIMENSIONS_IMPL_H

#include <stdint.h> // required for libdivide
#include "libdivide.h"
#include "../magic_dimensions.h"

namespace libtensor {


template<size_t N>
magic_dimensions<N>::magic_dimensions(const dimensions<N> &dims, bool incs) :

    m_dims(dims), m_incs(incs), m_magic(0) {

    make_magic();
}


template<size_t N>
magic_dimensions<N>::magic_dimensions(const magic_dimensions<N> &mdims) :

    m_dims(mdims.m_dims), m_incs(mdims.m_incs), m_magic(0) {

    typedef sequence<N, libdivide::libdivide_u64_t> seq_t;

    const seq_t &other_magic = *reinterpret_cast<const seq_t*>(mdims.m_magic);
    m_magic = new seq_t(other_magic);
}


template<size_t N>
magic_dimensions<N>::~magic_dimensions() {

    typedef sequence<N, libdivide::libdivide_u64_t> seq_t;

    seq_t *pmagic = reinterpret_cast<seq_t*>(m_magic);
    delete pmagic;
}


template<size_t N>
void magic_dimensions<N>::permute(const permutation<N> &p) {

    typedef sequence<N, libdivide::libdivide_u64_t> seq_t;

    seq_t *pmagic = reinterpret_cast<seq_t*>(m_magic);
    delete pmagic;
    m_magic = 0;

    m_dims.permute(p);
    make_magic();
}


template<size_t N>
void magic_dimensions<N>::divide(const index<N> &i1, index<N> &i2) const {

    typedef sequence<N, libdivide::libdivide_u64_t> seq_t;

    const seq_t &magic = *reinterpret_cast<const seq_t*>(m_magic);
    for(size_t i = 0; i < N; i++) {
        i2[i] = libdivide::libdivide_u64_do(i1[i], &magic[i]);
    }
}


template<size_t N>
size_t magic_dimensions<N>::divide(size_t n, size_t i) const {

    typedef sequence<N, libdivide::libdivide_u64_t> seq_t;

    const seq_t &magic = *reinterpret_cast<const seq_t*>(m_magic);
    return libdivide::libdivide_u64_do(n, &magic[i]);
}


template<size_t N>
void magic_dimensions<N>::make_magic() {

    typedef sequence<N, libdivide::libdivide_u64_t> seq_t;

    seq_t *pmagic = new seq_t;
    seq_t &magic = *pmagic;

    if(m_incs) {
        for(size_t i = 0; i < N; i++) {
            magic[i] = libdivide::libdivide_u64_gen(m_dims.get_increment(i));
        }
    } else {
        for(size_t i = 0; i < N; i++) {
            magic[i] = libdivide::libdivide_u64_gen(m_dims.get_dim(i));
        }
    }

    m_magic = pmagic;
}


} // namespace libtensor

#endif // LIBTENSOR_MAGIC_DIMENSIONS_IMPL_H
