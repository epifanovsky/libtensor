#ifndef LIBTENSOR_BTOD_EXPORT_ARMA_IMPL_H
#define LIBTENSOR_BTOD_EXPORT_ARMA_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include <libtensor/block_tensor/btod_export.h>
#include "../btod_export_arma.h"

namespace libtensor {


template<size_t N>
const char btod_export_arma<N>::k_clazz[] = "btod_export_arma<N>";


template<size_t N>
btod_export_arma<N>::btod_export_arma(block_tensor_rd_i<N, double> &bt) :
    m_bt(bt) {

}


template<size_t N>
void btod_export_arma<N>::perform(arma::Mat<double> &m) {

    static const char method[] = "perform(arma::Mat<double>&)";

    const dimensions<N> &dims = m_bt.get_bis().get_dims();
    if(dims.get_size() != m.n_rows * m.n_cols) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "m");
    }

    btod_export<N>(m_bt).perform(m.memptr());
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EXPORT_ARMA_IMPL_H

