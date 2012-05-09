#ifndef LIBTENSOR_BTO_SET_IMPL_H
#define LIBTENSOR_BTO_SET_IMPL_H

#include <libtensor/core/orbit_list.h>

namespace libtensor {


template<size_t N, typename Traits>
void bto_set<N, Traits>::perform(block_tensor_type &bt) {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;
    typedef typename Traits::template block_type<N>::type block_type;
    typedef typename Traits::template to_set_type<N>::type to_set;

    block_tensor_ctrl_type ctrl(bt);

    orbit_list<N, element_type> ol(ctrl.req_symmetry());

    for(typename orbit_list<N, element_type>::iterator io = ol.begin();
        io != ol.end(); ++io) {

        index<N> bi(ol.get_index(io));
        if(Traits::is_zero(m_v)) {
            ctrl.req_zero_block(bi);
        } else {
            block_type &blk = ctrl.req_block(bi);
            to_set(m_v).perform(blk);
            ctrl.ret_block(bi);
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_SET_IMPL_H
