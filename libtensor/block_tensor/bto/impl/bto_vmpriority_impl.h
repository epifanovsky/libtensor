#ifndef LIBTENSOR_BTO_VMPRIORITY_IMPL_H
#define LIBTENSOR_BTO_VMPRIORITY_IMPL_H

#include <libtensor/core/orbit_list.h>
#include "../bto_vmpriority.h"

namespace libtensor {


template<size_t N, typename Traits>
void bto_vmpriority<N, Traits>::set_priority() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;
    typedef typename Traits::template block_type<N>::type block_type;
    typedef typename Traits::template to_vmpriority_type<N>::type to_vmpriority;

    block_tensor_ctrl_type ctrl(m_bt);

    orbit_list<N, element_type> ol(ctrl.req_symmetry());

    for(typename orbit_list<N, element_type>::iterator io = ol.begin();
        io != ol.end(); ++io) {

        index<N> bi(ol.get_index(io));
        block_type &blk = ctrl.req_block(bi);
        to_vmpriority(blk).set_priority();
        ctrl.ret_block(bi);
    }
}


template<size_t N, typename Traits>
void bto_vmpriority<N, Traits>::unset_priority() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;
    typedef typename Traits::template block_type<N>::type block_type;
    typedef typename Traits::template to_vmpriority_type<N>::type to_vmpriority;

    block_tensor_ctrl_type ctrl(m_bt);

    orbit_list<N, element_type> ol(ctrl.req_symmetry());

    for(typename orbit_list<N, element_type>::iterator io = ol.begin();
        io != ol.end(); ++io) {

        index<N> bi(ol.get_index(io));
        block_type &blk = ctrl.req_block(bi);
        to_vmpriority(blk).unset_priority();
        ctrl.ret_block(bi);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_VMPRIORITY_IMPL_H
