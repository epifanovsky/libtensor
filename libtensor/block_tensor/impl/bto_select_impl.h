#ifndef LIBTENSOR_BTO_SELECT_IMPL_H
#define LIBTENSOR_BTO_SELECT_IMPL_H

#include <libtensor/gen_block_tensor/impl/gen_bto_select_impl.h>
#include "../bto_select.h"

namespace libtensor {


template<size_t N, typename T, typename ComparePolicy>
const char *bto_select<N, T, ComparePolicy>::k_clazz =
        "bto_select<N, T, ComparePolicy>";


template<size_t N, typename T, typename ComparePolicy>
bto_select<N, T, ComparePolicy>::bto_select(
        block_tensor_rd_i<N, T> &bt, compare_type cmp) :
    m_gbto(bt, cmp) {

}

template<size_t N, typename T, typename ComparePolicy>
bto_select<N, T, ComparePolicy>::bto_select(block_tensor_rd_i<N, T> &bt,
        const symmetry<N, T> &sym, compare_type cmp) :
    m_gbto(bt, sym, cmp) {

}


template<size_t N, typename T, typename ComparePolicy>
void bto_select<N, T, ComparePolicy>::perform(list_type &li, size_t n) {

    m_gbto.perform(li, n);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_SELECT_H
