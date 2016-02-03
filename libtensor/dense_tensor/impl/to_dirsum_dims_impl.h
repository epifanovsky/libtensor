#ifndef LIBTENSOR_TO_DIRSUM_DIMS_IMPL_H
#define LIBTENSOR_TO_DIRSUM_DIMS_IMPL_H

#include "../to_dirsum_dims.h"

namespace libtensor {


template<size_t N, size_t M>
const char to_dirsum_dims<N, M>::k_clazz[] = "to_dirsum_dims<N, M>";


namespace {

template<size_t N, size_t M>
dimensions<N + M> make_to_dirsum_dims(
    const dimensions<N> &dimsa, const dimensions<M> &dimsb) {

    enum {
        NA = to_dirsum_dims<N, M>::NA,
        NB = to_dirsum_dims<N, M>::NB,
        NC = to_dirsum_dims<N, M>::NC
    };

    index<NC> i1, i2;
    for(size_t i = 0; i < NA; i++) i2[i] = dimsa[i] - 1;
    for(size_t i = 0; i < NB; i++) i2[NA + i] = dimsb[i] - 1;

    return dimensions<NC>(index_range<NC>(i1, i2));
}

} // unnamed namespace


template<size_t N, size_t M>
to_dirsum_dims<N, M>::to_dirsum_dims(const dimensions<NA> &dimsa,
    const dimensions<NB> &dimsb, const permutation<NC> &permc) :

    m_dimsc(make_to_dirsum_dims<N, M>(dimsa, dimsb)) {

    m_dimsc.permute(permc);
}


} // namespace libtensor

#endif // LIBTENSOR_TO_DIRSUM_DIMS_IMPL_H
