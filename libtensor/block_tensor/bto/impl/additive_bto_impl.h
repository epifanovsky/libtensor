#ifndef LIBTENSOR_ADDITIVE_BTO_IMPL_H
#define LIBTENSOR_ADDITIVE_BTO_IMPL_H

#include <libtensor/symmetry/so_copy.h>
#include <libtensor/symmetry/so_dirsum.h>
#include <libtensor/symmetry/so_merge.h>
#include "../additive_bto.h"

namespace libtensor {


template<size_t N, typename Traits>
void additive_bto<N, Traits>::compute_block(additive_bto<N, Traits> &op,
    bool zero, block_t &blk, const index<N> &i,
    const tensor_transf<N, element_t> &tr, const element_t &c) {

    op.compute_block(zero, blk, i, tr, c);
}


} // namespace libtensor

#endif // LIBTENSOR_ADDITIVE_BTOD_IMPL_H
