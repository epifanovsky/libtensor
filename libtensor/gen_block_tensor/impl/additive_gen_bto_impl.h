#ifndef LIBTENSOR_ADDITIVE_GEN_BTO_IMPL_H
#define LIBTENSOR_ADDITIVE_GEN_BTO_IMPL_H

#include <libtensor/symmetry/so_copy.h>
#include <libtensor/symmetry/so_dirsum.h>
#include <libtensor/symmetry/so_merge.h>
#include "../additive_gen_bto.h"

namespace libtensor {


template<size_t N, typename BtiTraits>
void additive_gen_bto<N, BtiTraits>::compute_block(
        additive_gen_bto<N, BtiTraits> &op,
        bool zero,
        const index<N> &idx,
        const tensor_transf_type &tr,
        wr_block_type &blk) {

    op.compute_block(zero, idx, tr, blk);
}


} // namespace libtensor

#endif // LIBTENSOR_ADDITIVE_GEN_BTOD_IMPL_H
