#ifndef LIBTENSOR_BTO_EXPORT_ARMA_IMPL_H
#define LIBTENSOR_BTO_EXPORT_ARMA_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include <libtensor/block_tensor/bto_export.h>
#include "../bto_export_arma.h"

namespace libtensor {


template<size_t N, typename T>
const char bto_export_arma<N, T>::k_clazz[] = "bto_export_arma<N, T>";


template<size_t N, typename T>
bto_export_arma<N, T>::bto_export_arma(block_tensor_rd_i<N, T> &bt) :
    m_bt(bt) {

}


template<size_t N, typename T>
void bto_export_arma<N, T>::perform(arma::Mat<T> &m) {

    static const char method[] = "perform(arma::Mat<T>&)";

    const dimensions<N> &dims = m_bt.get_bis().get_dims();
    if(dims.get_size() != m.n_rows * m.n_cols) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "m");
    }

    bto_export<N, T>(m_bt).perform(m.memptr());
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_EXPORT_ARMA_IMPL_H

