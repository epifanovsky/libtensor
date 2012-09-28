#ifndef LIBTENSOR_BTOD_DOTPROD_IMPL_H
#define LIBTENSOR_BTOD_DOTPROD_IMPL_H

#include "../btod_dotprod.h"

namespace libtensor {


template<size_t N>
const char *btod_dotprod<N>::k_clazz = "btod_dotprod<N>";


template<size_t N>
void btod_dotprod<N>::add_arg(
        block_tensor_rd_i<N, double> &bt1,
        block_tensor_rd_i<N, double> &bt2) {

    m_gbto.add_arg(bt1, tensor_transf<N, double>(),
            bt2, tensor_transf<N, double>());
}


template<size_t N>
void btod_dotprod<N>::add_arg(
        block_tensor_rd_i<N, double> &bt1,
        const permutation<N> &perm1,
        block_tensor_rd_i<N, double> &bt2,
        const permutation<N> &perm2) {

    m_gbto.add_arg(bt1, tensor_transf<N, double>(perm1),
            bt2, tensor_transf<N, double>(perm2));
}


template<size_t N>
double btod_dotprod<N>::calculate() {

    std::vector<double> v(1);
    m_gbto.calculate(v);
    return v[0];
}


template<size_t N>
void btod_dotprod<N>::calculate(std::vector<double> &v) {

    m_gbto.calculate(v);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_DOTPROD_IMPL_H
