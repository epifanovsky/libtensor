#ifndef LIBTENSOR_BTO_AUX_SYMMETRIZE_IMPL_H
#define LIBTENSOR_BTO_AUX_SYMMETRIZE_IMPL_H

#include <libtensor/core/orbit.h>
#include <libtensor/symmetry/so_copy.h>
#include "../bto_aux_symmetrize.h"

namespace libtensor {


template<size_t N, typename Traits>
bto_aux_symmetrize<N, Traits>::bto_aux_symmetrize(const symmetry_type &sym,
    bto_stream_i<N, Traits> &out) :

    m_sym(sym.get_bis()), m_out(out), m_open(false) {

    so_copy<N, element_type>(sym).perform(m_sym);
}


template<size_t N, typename Traits>
bto_aux_symmetrize<N, Traits>::~bto_aux_symmetrize() {

    if(m_open) close();
}


template<size_t N, typename Traits>
void bto_aux_symmetrize<N, Traits>::open() {

    if(!m_open) {
        m_out.open();
        m_open = true;
    }
}


template<size_t N, typename Traits>
void bto_aux_symmetrize<N, Traits>::close() {

    if(m_open) {
        m_out.close();
        m_open = false;
    }
}


template<size_t N, typename Traits>
void bto_aux_symmetrize<N, Traits>::put(const index<N> &idx, block_type &blk,
    const tensor_transf_type &tr) {

    orbit<N, element_type> o(m_sym, idx);

    tensor_transf_type tr1(tr);
    tr1.transform(tensor_transf_type(o.get_transf(idx), true));

    m_out.put(o.get_cindex(), blk, tr1);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_AUX_SYMMETRIZE_IMPL_H
