#ifndef LIBTENSOR_GEN_BTO_UNFOLD_BLOCK_LIST_IMPL_H
#define LIBTENSOR_GEN_BTO_UNFOLD_BLOCK_LIST_IMPL_H

#include <libtensor/core/orbit.h>
#include "gen_bto_unfold_block_list.h"

namespace libtensor {


template<size_t N, typename Traits>
void gen_bto_unfold_block_list<N, Traits>::build(block_list<N> &blstx) {

    for(typename block_list<N>::iterator i = m_blst.begin();
        i != m_blst.end(); ++i) {

        orbit<N, element_type> o(m_sym, m_blst.get_abs_index(i), false);
        for(typename orbit<N, element_type>::iterator j = o.begin();
            j != o.end(); ++j) blstx.add(o.get_abs_index(j));
    }
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_UNFOLD_BLOCK_LIST_IMPL_H
