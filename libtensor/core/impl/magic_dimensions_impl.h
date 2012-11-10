#ifndef LIBTENSOR_MAGIC_DIMENSIONS_IMPL_H
#define LIBTENSOR_MAGIC_DIMENSIONS_IMPL_H

#include "../magic_dimensions.h"

namespace libtensor {


template<size_t N>
magic_dimensions<N>::magic_dimensions(const dimensions<N> &dims, bool incs) :

    m_dims(dims) {

    if(incs) {
        for(size_t i = 0; i < N; i++) {
            m_magic[i] = libdivide::divider<uint64_t>(m_dims.get_increment(i));
        }
    } else {
        for(size_t i = 0; i < N; i++) {
            m_magic[i] = libdivide::divider<uint64_t>(m_dims.get_dim(i));
        }
    }
}


template<size_t N>
inline const libdivide::divider<uint64_t> &
magic_dimensions<N>::operator[](size_t i) const {

    return m_magic[i];
}


} // namespace libtensor

#endif // LIBTENSOR_MAGIC_DIMENSIONS_IMPL_H
