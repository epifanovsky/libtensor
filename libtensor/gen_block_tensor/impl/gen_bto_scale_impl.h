#ifndef LIBTENSOR_GEN_BTO_SCALE_IMPL_H
#define LIBTENSOR_GEN_BTO_SCALE_IMPL_H

#include <libtensor/core/abs_index.h>
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_scale.h"

namespace libtensor {


template<size_t N, typename Traits, typename Timed>
const char gen_bto_scale<N, Traits, Timed>::k_clazz[] =
    "gen_bto_scale<N, Traits, Timed>";


template<size_t N, typename Traits, typename Timed>
void gen_bto_scale<N, Traits, Timed>::perform() {

    typedef typename Traits::template to_scale_type<N>::type to_scale_type;

    gen_bto_scale::start_timer();

    try {

        dimensions<N> bidims = m_bt.get_bis().get_block_index_dims();

        gen_block_tensor_ctrl<N, bti_traits> ctrl(m_bt);

        std::vector<size_t> nzblk;
        ctrl.req_nonzero_blocks(nzblk);

        for(size_t i = 0; i < nzblk.size(); i++) {

            index<N> idx;
            abs_index<N>::get_index(nzblk[i], bidims, idx);

            if(m_c.is_zero()) {
                ctrl.req_zero_block(idx);
            } else {
                wr_block_type &blk = ctrl.req_block(idx);
                to_scale_type(m_c).perform(blk);
                ctrl.ret_block(idx);
            }
        }

    } catch(...) {
        gen_bto_scale::stop_timer();
        throw;
    }

    gen_bto_scale::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SCALE_IMPL_H
