#ifndef LIBTENSOR_GEN_BTO_VMPRIORITY_IMPL_H
#define LIBTENSOR_GEN_BTO_VMPRIORITY_IMPL_H

#include <libtensor/core/orbit_list.h>
#include "../gen_bto_vmpriority.h"

namespace libtensor {


template<size_t N, typename Traits>
void gen_bto_vmpriority<N, Traits>::set_priority() {

    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;
    typedef typename Traits::template to_vmpriority_type<N>::type to_vmpriority;

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrl(m_bt);

    orbit_list<N, element_type> ol(ctrl.req_const_symmetry());
    for(typename orbit_list<N, element_type>::iterator io = ol.begin();
        io != ol.end(); ++io) {

        index<N> bi(ol.get_index(io));
        if(ctrl.req_is_zero_block(bi)) continue;

        rd_block_type &blk = ctrl.req_const_block(bi);
        to_vmpriority(blk).set_priority();
        ctrl.ret_const_block(bi);
    }
}


template<size_t N, typename Traits>
void gen_bto_vmpriority<N, Traits>::unset_priority() {

    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;
    typedef typename Traits::template to_vmpriority_type<N>::type to_vmpriority;

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrl(m_bt);

    orbit_list<N, element_type> ol(ctrl.req_const_symmetry());

    for(typename orbit_list<N, element_type>::iterator io = ol.begin();
            io != ol.end(); ++io) {

        index<N> bi(ol.get_index(io));
        if(ctrl.req_is_zero_block(bi)) continue;

        rd_block_type &blk = ctrl.req_const_block(bi);
        to_vmpriority(blk).unset_priority();
        ctrl.ret_const_block(bi);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_VMPRIORITY_IMPL_H
