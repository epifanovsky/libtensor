#ifndef LIBTENSOR_GEN_BTO_SET_IMPL_H
#define LIBTENSOR_GEN_BTO_SET_IMPL_H

#include <libtensor/core/orbit_list.h>
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_set.h"

namespace libtensor {


template<size_t N, typename Traits, typename Timed>
void gen_bto_set<N, Traits, Timed>::perform(
    gen_block_tensor_wr_i<N, bti_traits> &bta) {

    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;
    typedef typename Traits::template to_set_type<N>::type to_set;

    gen_bto_set::start_timer();

    try {

        gen_block_tensor_wr_ctrl<N, bti_traits> ca(bta);

        if(Traits::is_zero(m_v)) {

            ca.req_zero_all_blocks();

        } else {

            orbit_list<N, element_type> ol(ca.req_const_symmetry());
            for(typename orbit_list<N, element_type>::iterator io = ol.begin();
                io != ol.end(); ++io) {

                index<N> bi;
                ol.get_index(io, bi);
                wr_block_type &blk = ca.req_block(bi);
                to_set(m_v).perform(true, blk);
                ca.ret_block(bi);
            }

        }

    } catch(...) {
        gen_bto_set::stop_timer();
        throw;
    }

    gen_bto_set::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SET_IMPL_H
