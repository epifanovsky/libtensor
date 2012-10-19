#ifndef LIBTENSOR_TO_CONTRACT2_DIMS_IMPL_H
#define LIBTENSOR_TO_CONTRACT2_DIMS_IMPL_H

#include <libtensor/tod/bad_dimensions.h>
#include "../to_contract2_dims.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *to_contract2_dims<N, M, K>::k_clazz = "to_contract2_dims<N, M, K>";


template<size_t N, size_t M, size_t K>
dimensions<N + M> to_contract2_dims<N, M, K>::make_dimsc(
    const contraction2<N, M, K> &contr, const dimensions<N + K> &dimsa,
    const dimensions<M + K> &dimsb) {

    static const char *method = "make_dimsc()";

    const sequence<2 * (N + M + K), size_t> &conn = contr.get_conn();

#ifdef LIBTENSOR_DEBUG
    for(size_t i = 0; i < N + K; i++) {
        if(conn[i + N + M] >= 2 * N + M + K) {
            size_t j = conn[i + N + M] - 2 * N - M - K;
            if(dimsa[i] != dimsb[j]) {
                throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "ta,tb");
            }
        }
    }
#endif // LIBTENSOR_DEBUG

    index<N + M> i1, i2;
    for(size_t i = 0; i < N + M; i++) {
        size_t j = conn[i] - N - M;
        if(j < N + K) i2[i] = dimsa[j] - 1;
        else i2[i] = dimsb[j - N - K] - 1;
    }

    return dimensions<N + M>(index_range<N + M>(i1, i2));
}


} // namespace libtensor

#endif // LIBTENSOR_TO_CONTRACT2_DIMS_IMPL_H

