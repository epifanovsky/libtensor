#ifndef LIBTENSOR_BTO_DOTPROD_IMPL_H
#define LIBTENSOR_BTO_DOTPROD_IMPL_H

#include <libtensor/core/scalar_transf_double.h>
#include "../bto_dotprod.h"

namespace libtensor {


template<size_t N, typename T>
const char *bto_dotprod<N, T>::k_clazz = "bto_dotprod<N, T>";


template<size_t N, typename T>
void bto_dotprod<N, T>::add_arg(
        block_tensor_rd_i<N, T> &bt1,
        block_tensor_rd_i<N, T> &bt2) {

    m_gbto.add_arg(bt1, tensor_transf<N, T>(),
            bt2, tensor_transf<N, T>());
}


template<size_t N, typename T>
void bto_dotprod<N, T>::add_arg(
        block_tensor_rd_i<N, T> &bt1,
        const permutation<N> &perm1,
        block_tensor_rd_i<N, T> &bt2,
        const permutation<N> &perm2) {

    m_gbto.add_arg(bt1, tensor_transf<N, T>(perm1),
            bt2, tensor_transf<N, T>(perm2));
}


template<size_t N, typename T>
T bto_dotprod<N, T>::calculate() {

    std::vector<T> v(1);
    m_gbto.calculate(v);
    return v[0];
}


template<size_t N, typename T>
void bto_dotprod<N, T>::calculate(std::vector<T> &v) {

    m_gbto.calculate(v);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_DOTPROD_IMPL_H
