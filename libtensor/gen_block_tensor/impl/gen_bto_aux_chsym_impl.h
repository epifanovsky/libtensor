#ifndef LIBTENSOR_GEN_BTO_AUX_CHSYM_IMPL_H
#define LIBTENSOR_GEN_BTO_AUX_CHSYM_IMPL_H

#include <set>
#include <libtensor/core/orbit.h>
#include <libtensor/symmetry/so_copy.h>
#include "../block_stream_exception.h"
#include "../gen_bto_aux_chsym.h"

namespace libtensor {


template<size_t N, typename Traits>
const char gen_bto_aux_chsym<N, Traits>::k_clazz[] =
    "gen_bto_aux_chsym<N, Traits>";


template<size_t N, typename Traits>
gen_bto_aux_chsym<N, Traits>::gen_bto_aux_chsym(
    const symmetry_type &syma,
    const symmetry_type &symb,
    gen_block_stream_i<N, bti_traits> &out) :

    m_syma(syma.get_bis()), m_symb(symb.get_bis()), m_out(out), m_open(false) {

    so_copy<N, element_type>(syma).perform(m_syma);
    so_copy<N, element_type>(symb).perform(m_symb);
}


template<size_t N, typename Traits>
gen_bto_aux_chsym<N, Traits>::~gen_bto_aux_chsym() {

    if(m_open) close();
}


template<size_t N, typename Traits>
void gen_bto_aux_chsym<N, Traits>::open() {

    if(m_open) {
        throw block_stream_exception(g_ns, k_clazz, "open()",
            __FILE__, __LINE__, "Stream is already open.");
    }

    m_open = true;
}


template<size_t N, typename Traits>
void gen_bto_aux_chsym<N, Traits>::close() {

    if(!m_open) {
        throw block_stream_exception(g_ns, k_clazz, "close()",
            __FILE__, __LINE__, "Stream is already closed.");
    }

    m_open = false;
}


template<size_t N, typename Traits>
void gen_bto_aux_chsym<N, Traits>::put(
    const index<N> &idxa,
    rd_block_type &blk,
    const tensor_transf_type &tr) {

    if(!m_open) {
        throw block_stream_exception(g_ns, k_clazz, "put()",
            __FILE__, __LINE__, "Stream is not ready.");
    }

    std::set<size_t> orb;

    orbit<N, element_type> oa(m_syma, idxa, false);
    for(typename orbit<N, element_type>::iterator i = oa.begin();
        i != oa.end(); ++i) orb.insert(oa.get_abs_index(i));

    while(!orb.empty()) {

        size_t aidxb = *orb.begin();
        orbit<N, element_type> ob(m_symb, aidxb, false);

        tensor_transf_type tr1(tr);
        tr1.transform(oa.get_transf(ob.get_acindex()));
        m_out.put(ob.get_cindex(), blk, tr1);

        for(typename orbit<N, element_type>::iterator i = ob.begin();
            i != ob.end(); ++i) orb.erase(ob.get_abs_index(i));
    }
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_AUX_CHSYM_IMPL_H
