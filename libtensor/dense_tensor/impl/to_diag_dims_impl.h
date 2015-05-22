#ifndef LIBTENSOR_TO_DIAG_DIMS_IMPL_H
#define LIBTENSOR_TO_DIAG_DIMS_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../to_diag_dims.h"

namespace libtensor {


template<size_t N, size_t M>
const char to_diag_dims<N, M>::k_clazz[] = "to_diag_dims<N, M>";


namespace {

template<size_t N, size_t M>
dimensions<M> make_to_diag_dims(
    const dimensions<N> &dimsa, const sequence<N, size_t> &msk) {

    static const char method[] =
        "make_to_diag_dims(const dimensions<N>&, const sequence<N, size_t>&)";

    enum {
        NA = to_diag_dims<N, M>::NA,
        NB = to_diag_dims<N, M>::NB
    };

    index<NB> i1, i2;

    bool bad_dims = false;
    sequence<NB + 1, size_t> d(0);
    size_t m = 0;
    for(size_t i = 0; i < NA; i++) {
        if(msk[i] != 0) {
            if(d[msk[i]] == 0) {
                d[msk[i]] = dimsa[i];
                i2[m++] = dimsa[i] - 1;
            } else {
                bad_dims = bad_dims || d[msk[i]] != dimsa[i];
            }
        } else {
            if(!bad_dims) i2[m++] = dimsa[i] - 1;
        }
    }
    if(m != M) {
        throw bad_parameter(g_ns, to_diag_dims<N, M>::k_clazz, method,
            __FILE__, __LINE__, "m");
    }
    if(bad_dims) {
        throw bad_dimensions(g_ns, to_diag_dims<N, M>::k_clazz, method,
            __FILE__, __LINE__, "t");
    }
    return dimensions<NB>(index_range<NB>(i1, i2));
}

} // unnamed namespace


template<size_t N, size_t M>
to_diag_dims<N, M>::to_diag_dims(const dimensions<NA> &dimsa,
    const sequence<NA, size_t> &m, const permutation<NB> &permb) :

    m_dimsb(make_to_diag_dims<N, M>(dimsa, m)) {

    m_dimsb.permute(permb);
}


} // namespace libtensor

#endif // LIBTENSOR_TO_DIAG_DIMS_IMPL_H
