#ifndef LIBTENSOR_GEN_BTO_AUX_SYMMETRIZE_IMPL_H
#define LIBTENSOR_GEN_BTO_AUX_SYMMETRIZE_IMPL_H

#include <libtensor/core/orbit.h>
#include <libtensor/symmetry/so_copy.h>
#include "../block_stream_exception.h"
#include "../gen_bto_aux_symmetrize.h"

namespace libtensor {


template<size_t N, typename Traits>
const char gen_bto_aux_symmetrize<N, Traits>::k_clazz[] =
    "gen_bto_aux_symmetrize<N, Traits>";


template<size_t N, typename Traits>
gen_bto_aux_symmetrize<N, Traits>::gen_bto_aux_symmetrize(
    const symmetry_type &syma,
    const symmetry_type &symb,
    gen_block_stream_i<N, bti_traits> &out) :

    m_syma(syma.get_bis()), m_symb(symb.get_bis()), m_out(out),
    m_open(false) {

    so_copy<N, element_type>(syma).perform(m_syma);
    so_copy<N, element_type>(symb).perform(m_symb);
}


template<size_t N, typename Traits>
gen_bto_aux_symmetrize<N, Traits>::~gen_bto_aux_symmetrize() {

    if(m_open) close();
}


template<size_t N, typename Traits>
void gen_bto_aux_symmetrize<N, Traits>::add_transf(
    const tensor_transf_type &tr) {

    m_trlst.push_back(tr);
}


template<size_t N, typename Traits>
void gen_bto_aux_symmetrize<N, Traits>::open() {

    if(m_open) {
        throw block_stream_exception(g_ns, k_clazz, "open()",
            __FILE__, __LINE__, "Stream is already open.");
    }

    m_open = true;
}


template<size_t N, typename Traits>
void gen_bto_aux_symmetrize<N, Traits>::close() {

    if(!m_open) {
        throw block_stream_exception(g_ns, k_clazz, "close()",
            __FILE__, __LINE__, "Stream is already closed.");
    }

    m_trlst.clear();
    m_open = false;
}


template<size_t N, typename Traits>
void gen_bto_aux_symmetrize<N, Traits>::put(
    const index<N> &idxa,
    rd_block_type &blk,
    const tensor_transf_type &tr) {

    if(!m_open) {
        throw block_stream_exception(g_ns, k_clazz, "put()",
            __FILE__, __LINE__, "Stream is not ready.");
    }

    dimensions<N> bidimsa = m_syma.get_bis().get_block_index_dims();
    dimensions<N> bidimsb = m_symb.get_bis().get_block_index_dims();

    orbit<N, element_type> oa(m_syma, idxa, false);
#ifdef LIBTENSOR_DEBUG
    if(!oa.get_cindex().equals(idxa)) {
        throw symmetry_violation(g_ns, k_clazz, "put()", __FILE__, __LINE__,
            "Expected canonical index idxa");
    }
#endif // LIBTENSOR_DEBUG

    std::multimap<size_t, tensor_transf_type> symap;

    for(typename orbit<N, element_type>::iterator i = oa.begin();
        i != oa.end(); ++i) {

        const tensor_transf_type &tra1 = oa.get_transf(i);
        for(typename std::list<tensor_transf_type>::const_iterator j =
            m_trlst.begin(); j != m_trlst.end(); ++j) {

            index<N> idxb;
            abs_index<N>::get_index(oa.get_abs_index(i), bidimsa, idxb);
            j->apply(idxb);
            size_t aidxb = abs_index<N>::get_abs_index(idxb, bidimsb);

            tensor_transf<N, element_type> trb(tr);
            trb.transform(tra1).transform(*j);
            // There is a bug with icc 13 and -O3, need to break it down
            //symap.insert(std::make_pair(aidxb, trb));
            std::pair<size_t, tensor_transf_type> p;
            p.first = aidxb;
            typename std::multimap<size_t, tensor_transf_type>::iterator jjj =
                symap.insert(p);
            jjj->second.transform(trb);
        }
    }

    typedef typename std::multimap<size_t, tensor_transf_type>::iterator
        symap_iterator;

    while(!symap.empty()) {

        size_t aidxb = symap.begin()->first;
        index<N> idxb;
        abs_index<N>::get_index(aidxb, bidimsb, idxb);
        std::pair<symap_iterator, symap_iterator> irange =
            symap.equal_range(aidxb);
        
        std::multimap<size_t, tensor_transf_type> symap2;
        symap2.insert(irange.first, irange.second);
        while(!symap2.empty()) {
            symap_iterator i = symap2.begin();
            permutation<N> perm(i->second.get_perm());
            scalar_transf_sum<element_type> sum;
            while(i != symap2.end()) {
                if(perm.equals(i->second.get_perm())) {
                    sum.add(i->second.get_scalar_tr());
                    symap_iterator j = i;
                    ++i;
                    symap2.erase(j);
                } else {
                    ++i;
                }
            }
            if(!sum.is_zero()) {
                tensor_transf<N, element_type> tr(perm, sum.get_transf());
                m_out.put(idxb, blk, tr);
            }
        }

        orbit<N, element_type> ob(m_symb, idxb, false);
        for(typename orbit<N, element_type>::iterator i = ob.begin();
            i != ob.end(); ++i) {
            symap.erase(ob.get_abs_index(i));
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_AUX_SYMMETRIZE_IMPL_H
