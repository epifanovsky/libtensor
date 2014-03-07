#ifndef LIBTENSOR_BTOD_SELECT_IMPL_H
#define LIBTENSOR_BTOD_SELECT_IMPL_H

#include <libtensor/gen_block_tensor/impl/gen_bto_select_impl.h>
#include "../btod_select.h"

namespace libtensor {


template<size_t N, typename ComparePolicy>
const char *btod_select<N, ComparePolicy>::k_clazz =
        "btod_select<N, ComparePolicy>";


template<size_t N, typename ComparePolicy>
btod_select<N, ComparePolicy>::btod_select(
        block_tensor_rd_i<N, double> &bt, compare_type cmp) :
    m_gbto(bt, cmp) {

}

template<size_t N, typename ComparePolicy>
btod_select<N, ComparePolicy>::btod_select(block_tensor_rd_i<N, double> &bt,
        const symmetry<N, double> &sym, compare_type cmp) :
    m_gbto(bt, sym, cmp) {

}


template<size_t N, typename ComparePolicy>
void btod_select<N, ComparePolicy>::perform(list_type &li, size_t n) {

    m_gbto.perform(li, n);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SELECT_H
