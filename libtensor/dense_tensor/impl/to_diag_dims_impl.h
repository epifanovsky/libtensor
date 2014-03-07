#ifndef LIBTENSOR_TO_DIAG_DIMS_IMPL_H
#define LIBTENSOR_TO_DIAG_DIMS_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../to_diag_dims.h"

namespace libtensor {


template<size_t N, size_t M>
const char to_diag_dims<N, M>::k_clazz[] = "to_diag_dims<N, M>";


namespace {

template<size_t N, size_t M>
dimensions<N + 1 - M> make_to_diag_dims(
    const dimensions<N> &dimsa, const mask<N> &m) {

    static const char method[] =
        "make_to_diag_dims(const dimensions<N>&, const mask<N>&)";

    enum {
        NA = to_diag_dims<N, M>::NA,
        NB = to_diag_dims<N, M>::NB
    };

    size_t nd = 0;
    for(size_t i = 0; i < NA; i++) if(m[i]) nd++;
    if(nd != M) {
        throw bad_parameter(g_ns, to_diag_dims<N, M>::k_clazz, method,
            __FILE__, __LINE__, "m");
    }

    index<NB> i1, i2;
    size_t dimd = 0;
    for(size_t i = 0, j = 0; i < NA; i++) {
        if(m[i]) {
            if(dimd == 0) {
                dimd = dimsa[i];
                i2[j++] = dimd - 1;
            } else {
                if(dimsa[i] != dimd) {
                    throw bad_dimensions(g_ns, to_diag_dims<N, M>::k_clazz,
                        method, __FILE__, __LINE__, "dimsa");
                }
            }
        } else {
            i2[j++] = dimsa[i] - 1;
        }
    }

    return dimensions<NB>(index_range<NB>(i1, i2));
}

} // unnamed namespace


template<size_t N, size_t M>
to_diag_dims<N, M>::to_diag_dims(const dimensions<NA> &dimsa,
    const mask<NA> &m, const permutation<NB> &permb) :

    m_dimsb(make_to_diag_dims<N, M>(dimsa, m)) {

    m_dimsb.permute(permb);
}


} // namespace libtensor

#endif // LIBTENSOR_TO_DIAG_DIMS_IMPL_H
