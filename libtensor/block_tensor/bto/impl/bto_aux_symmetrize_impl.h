#ifndef LIBTENSOR_BTO_AUX_SYMMETRIZE_IMPL_H
#define LIBTENSOR_BTO_AUX_SYMMETRIZE_IMPL_H

#include <libtensor/core/orbit.h>
#include <libtensor/symmetry/so_copy.h>
#include "../bto_aux_symmetrize.h"

namespace libtensor {


template<size_t N, typename Traits>
bto_aux_symmetrize<N, Traits>::bto_aux_symmetrize(const symmetry_type &syma,
    const symmetry_type &symb, gen_block_stream_i<N, bti_traits> &out) :

    m_syma(syma.get_bis()), m_symb(symb.get_bis()), m_olb(symb), m_out(out),
    m_open(false) {

    so_copy<N, element_type>(syma).perform(m_syma);
    so_copy<N, element_type>(symb).perform(m_symb);
}


template<size_t N, typename Traits>
bto_aux_symmetrize<N, Traits>::~bto_aux_symmetrize() {

    if(m_open) close();
}


template<size_t N, typename Traits>
void bto_aux_symmetrize<N, Traits>::add_transf(const tensor_transf_type &tr) {

    m_trlst.push_back(tr);
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
        m_trlst.clear();
        m_open = false;
    }
}


template<size_t N, typename Traits>
void bto_aux_symmetrize<N, Traits>::put(const index<N> &idxa, block_type &blk,
    const tensor_transf_type &tr) {

    orbit<N, element_type> oa(m_syma, idxa, false);
    tensor_transf_type tra0inv(oa.get_transf(idxa), true);
    dimensions<N> bidims = m_syma.get_bis().get_block_index_dims();

    for(typename orbit<N, element_type>::iterator i = oa.begin();
        i != oa.end(); ++i) {

        const tensor_transf_type &tra1 = oa.get_transf(i);
        for(typename std::list<tensor_transf_type>::const_iterator j =
            m_trlst.begin(); j != m_trlst.end(); ++j) {

            index<N> idxb;
            abs_index<N>::get_index(oa.get_abs_index(i), bidims, idxb);
            j->apply(idxb);
            if(!m_olb.contains(idxb)) continue;

            tensor_transf<N, double> trb(tr);
            trb.transform(tra0inv).transform(tra1).transform(*j);
            m_out.put(idxb, blk, trb);
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_AUX_SYMMETRIZE_IMPL_H
