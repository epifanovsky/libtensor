#ifndef LIBTENSOR_DIAG_BTOD_DOTPROD_IMPL_H
#define LIBTENSOR_DIAG_BTOD_DOTPROD_IMPL_H

#include "../diag_btod_dotprod.h"

namespace libtensor {


template<size_t N>
const char diag_btod_dotprod<N>::k_clazz[] = "diag_btod_dotprod<N>";


template<size_t N>
void diag_btod_dotprod<N>::add_arg(
    diag_block_tensor_rd_i<N, double> &bt1,
    diag_block_tensor_rd_i<N, double> &bt2) {

    m_gbto.add_arg(bt1, tensor_transf<N, double>(),
        bt2, tensor_transf<N, double>());
}


template<size_t N>
void diag_btod_dotprod<N>::add_arg(
    diag_block_tensor_rd_i<N, double> &bt1, const permutation<N> &perm1,
    diag_block_tensor_rd_i<N, double> &bt2, const permutation<N> &perm2) {

    m_gbto.add_arg(bt1, tensor_transf<N, double>(perm1),
        bt2, tensor_transf<N, double>(perm2));
}


template<size_t N>
double diag_btod_dotprod<N>::calculate() {

    std::vector<double> v(1);
    m_gbto.calculate(v);
    return v[0];
}


template<size_t N>
void diag_btod_dotprod<N>::calculate(std::vector<double> &v) {

    m_gbto.calculate(v);
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_BTOD_DOTPROD_IMPL_H
