#ifndef LIBTENSOR_BTO_AUX_COPY_IMPL_H
#define LIBTENSOR_BTO_AUX_COPY_IMPL_H

#include <libtensor/symmetry/so_copy.h>
#include "../bto_aux_copy.h"

namespace libtensor {


template<size_t N, typename Traits>
bto_aux_copy<N, Traits>::bto_aux_copy(const symmetry_type &sym,
    block_tensor_type &bt) :

    m_sym(sym.get_bis()), m_bt(bt), m_ctrl(m_bt), m_open(false) {

    so_copy<N, element_type>(sym).perform(m_sym);
}


template<size_t N, typename Traits>
bto_aux_copy<N, Traits>::~bto_aux_copy() {

    if(m_open) close();
}


template<size_t N, typename Traits>
void bto_aux_copy<N, Traits>::open() {

    if(!m_open) {
        m_ctrl.req_zero_all_blocks();
        so_copy<N, element_type>(m_sym).perform(m_ctrl.req_symmetry());
        m_ctrl.req_sync_on();
        m_open = true;
    }
}


template<size_t N, typename Traits>
void bto_aux_copy<N, Traits>::close() {

    if(m_open) {
        m_ctrl.req_sync_off();
        m_open = false;
    }
}


template<size_t N, typename Traits>
void bto_aux_copy<N, Traits>::put(const index<N> &idx, block_type &blk,
    const tensor_transf_type &tr) {

    typedef typename Traits::template to_copy_type<N>::type to_copy_type;

    block_type &blk_tgt = m_ctrl.req_block(idx);
    to_copy_type(blk, tr).perform(true, Traits::identity(), blk_tgt);
    m_ctrl.ret_block(idx);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_AUX_COPY_IMPL_H
