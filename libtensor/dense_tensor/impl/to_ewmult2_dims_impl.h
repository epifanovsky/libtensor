#ifndef LIBTENSOR_TO_EWMULT2_DIMS_IMPL_H
#define LIBTENSOR_TO_EWMULT2_DIMS_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../to_ewmult2_dims.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char to_ewmult2_dims<N, M, K>::k_clazz[] = "to_ewmult2_dims<N, M, K>";


namespace {

template<size_t N, size_t M, size_t K>
dimensions<N + M + K> make_to_ewmult2_dims(
    const dimensions<N + K> &dimsa, const permutation<N + K> &perma,
    const dimensions<M + K> &dimsb, const permutation<M + K> &permb) {

    static const char method[] = "make_to_ewmult2_dims()";

    enum {
        NA = to_ewmult2_dims<N, M, K>::NA,
        NB = to_ewmult2_dims<N, M, K>::NB,
        NC = to_ewmult2_dims<N, M, K>::NC
    };

    dimensions<NA> dimsa1(dimsa);
    dimsa1.permute(perma);
    dimensions<NB> dimsb1(dimsb);
    dimsb1.permute(permb);

    for(size_t i = 0; i < K; i++) {
        if(dimsa1[N + i] != dimsb1[M + i]) {
            throw bad_dimensions(g_ns, to_ewmult2_dims<N, M, K>::k_clazz,
                method, __FILE__, __LINE__, "dimsa,dimsb");
        }
    }

    index<NC> i1, i2;
    for(size_t i = 0; i != N; i++) i2[i] = dimsa1[i] - 1;
    for(size_t i = 0; i != NB; i++) i2[N + i] = dimsb1[i] - 1;

    return dimensions<NC>(index_range<NC>(i1, i2));
}

} // unnamed namespace


template<size_t N, size_t M, size_t K>
to_ewmult2_dims<N, M, K>::to_ewmult2_dims(
    const dimensions<NA> &dimsa, const permutation<NA> &perma,
    const dimensions<NB> &dimsb, const permutation<NB> &permb,
    const permutation<NC> &permc) :

    m_dimsc(make_to_ewmult2_dims<N, M, K>(dimsa, perma, dimsb, permb)) {

    m_dimsc.permute(permc);
}


} // namespace libtensor

#endif // LIBTENSOR_TO_DIRSUM_DIMS_IMPL_H
