#ifndef LIBTENSOR_BTO_COMPARE_IMPL_H
#define LIBTENSOR_BTO_COMPARE_IMPL_H

#include <cmath> // for fabs
#include "../bto_compare.h"

namespace libtensor {


template<size_t N, typename T>
const char *bto_compare<N, T>::k_clazz = "bto_compare<N, T>";


template<size_t N, typename T>
bto_compare<N, T>::bto_compare(
        block_tensor_rd_i<N, T> &bt1,
        block_tensor_rd_i<N, T> &bt2,
        T thresh, bool strict) :

    m_gbto(bt1, bt2, fabs(thresh), strict) {

}


template<size_t N, typename T>
bool bto_compare<N, T>::compare() {

    return m_gbto.compare();
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_COMPARE_IMPL_H
