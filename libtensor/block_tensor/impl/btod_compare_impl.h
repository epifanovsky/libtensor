#ifndef LIBTENSOR_BTOD_COMPARE_IMPL_H
#define LIBTENSOR_BTOD_COMPARE_IMPL_H

#include <cmath> // for fabs
#include "../btod_compare.h"

namespace libtensor {


template<size_t N>
const char *btod_compare<N>::k_clazz = "btod_compare<N>";


template<size_t N>
btod_compare<N>::btod_compare(
        block_tensor_rd_i<N, double> &bt1,
        block_tensor_rd_i<N, double> &bt2,
        double thresh, bool strict) :

    m_gbto(bt1, bt2, fabs(thresh), strict) {

}


template<size_t N>
bool btod_compare<N>::compare() {

    return m_gbto.compare();
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_COMPARE_IMPL_H
