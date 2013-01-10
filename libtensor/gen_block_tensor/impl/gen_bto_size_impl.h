#ifndef LIBTENSOR_GEN_BTO_SIZE_IMPL_H
#define LIBTENSOR_GEN_BTO_SIZE_IMPL_H

#include <libtensor/core/abs_index.h>
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_size.h"

namespace libtensor {


template<size_t N, typename Traits>
size_t gen_bto_size<N, Traits>::get_size(
    gen_block_tensor_rd_i<N, bti_traits> &bt) {

    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;
    typedef typename Traits::template to_size_type<N>::type to_size;

    dimensions<N> bidims = bt.get_bis().get_block_index_dims();
    gen_block_tensor_rd_ctrl<N, bti_traits> ctrl(bt);

    size_t sz = 0;

    std::vector<size_t> blst;
    ctrl.req_nonzero_blocks(blst);
    for(typename std::vector<size_t>::const_iterator i = blst.begin();
        i != blst.end(); ++i) {

        index<N> idx;
        abs_index<N>::get_index(*i, bidims, idx);
        rd_block_type &blk = ctrl.req_const_block(idx);
        sz += to_size().get_size(blk);
        ctrl.ret_const_block(idx);
    }

    return sz;
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SIZE_IMPL_H
