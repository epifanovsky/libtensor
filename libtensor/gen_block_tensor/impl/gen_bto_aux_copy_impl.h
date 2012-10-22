#ifndef LIBTENSOR_GEN_BTO_AUX_COPY_IMPL_H
#define LIBTENSOR_GEN_BTO_AUX_COPY_IMPL_H

#include <libtensor/symmetry/so_copy.h>
#include "../gen_bto_aux_copy.h"

namespace libtensor {


template<size_t N, typename Traits>
gen_bto_aux_copy<N, Traits>::gen_bto_aux_copy(
    const symmetry<N, element_type> &sym,
    gen_block_tensor_wr_i<N, bti_traits> &bt) :

    m_sym(sym.get_bis()), m_bt(bt), m_ctrl(m_bt), m_open(false) {

    so_copy<N, element_type>(sym).perform(m_sym);
}


template<size_t N, typename Traits>
gen_bto_aux_copy<N, Traits>::~gen_bto_aux_copy() {

    if(m_open) close();
}


template<size_t N, typename Traits>
void gen_bto_aux_copy<N, Traits>::open() {

    if(!m_open) {
        m_ctrl.req_zero_all_blocks();
        so_copy<N, element_type>(m_sym).perform(m_ctrl.req_symmetry());
        m_open = true;
    }
}


template<size_t N, typename Traits>
void gen_bto_aux_copy<N, Traits>::close() {

    if(m_open) {
        m_open = false;
    }
}


template<size_t N, typename Traits>
void gen_bto_aux_copy<N, Traits>::put(
    const index<N> &idx,
    rd_block_type &blk,
    const tensor_transf<N, element_type> &tr) {

    typedef typename Traits::template to_copy_type<N>::type to_copy_type;

    wr_block_type &blk_tgt = m_ctrl.req_block(idx);
    to_copy_type(blk, tr).perform(true, blk_tgt);
    m_ctrl.ret_block(idx);
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_AUX_COPY_IMPL_H
