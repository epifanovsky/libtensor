#ifndef LIBTENSOR_BTO_AUX_ADD_IMPL_H
#define LIBTENSOR_BTO_AUX_ADD_IMPL_H

#include <libtensor/symmetry/so_copy.h>
#include <libtensor/symmetry/so_dirsum.h>
#include <libtensor/symmetry/so_merge.h>
#include "../bto_aux_add.h"

namespace libtensor {


template<size_t N, typename Traits>
bto_aux_add<N, Traits>::bto_aux_add(const symmetry_type &syma,
    block_tensor_type &btb) :

    m_syma(syma.get_bis()), m_btb(btb), m_cb(m_btb), m_open(false) {

    so_copy<N, element_type>(syma).perform(m_syma);
}


template<size_t N, typename Traits>
bto_aux_add<N, Traits>::~bto_aux_add() {

    close();
}


template<size_t N, typename Traits>
void bto_aux_add<N, Traits>::open() {

    if(m_open) return;

    //  Compute the symmetry of the result of the addition

    symmetry_type symcopy(m_sym.get_bis());
    so_copy<N, element_type>(m_sym).perform(symcopy);

    permutation<N + N> p0;
    block_index_space_product_builder<N, N> bbx(m_syma.get_bis(),
        m_btb.get_bis(), p0);
    symmetry<N + N, element_type> symx(bbx.get_bis());
    so_dirsum<N, N, element_type>(symcopy, m_syma, p0).perform(symx);
    mask<N + N> msk;
    sequence<N + N, size_t> seq(0);
    for(size_t i = 0; i < N; i++) {
        msk[i] = msk[i + N] = true;
        seq[i] = seq[i + N] = i;
    }
    so_merge<N + N, N, element_type>(symx, msk, seq).
        perform(m_cb.req_symmetry());

    m_cb.req_sync_on();
    m_open = true;
}


template<size_t N, typename Traits>
void bto_aux_add<N, Traits>::close() {

    if(!m_open) return;

    m_cb.req_sync_off();
    m_open = false;
}


template<size_t N, typename Traits>
void bto_aux_add<N, Traits>::put(const index<N> &idx, block_type &blk,
    const tensor_transf_type &tr) {

    typedef typename Traits::template to_copy_type<N>::type to_copy_type;

    block_type &blk_tgt = m_ctrl.req_block(idx);
    to_copy_type(blk, tr).perform(true, Traits::identity(), blk_tgt);
    m_ctrl.ret_block(idx);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_AUX_ADD_IMPL_H
