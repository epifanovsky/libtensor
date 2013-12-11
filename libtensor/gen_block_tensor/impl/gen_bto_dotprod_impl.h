#ifndef LIBTENSOR_GEN_BTO_DOTPROD_IMPL_H
#define LIBTENSOR_GEN_BTO_DOTPROD_IMPL_H

#include <libtensor/core/bad_block_index_space.h>
#include "../gen_bto_dotprod.h"
#include "gen_bto_aux_dotprod_impl.h"
#include "gen_bto_copy_impl.h"

namespace libtensor {


template<size_t N, typename Traits, typename Timed>
const char gen_bto_dotprod<N, Traits, Timed>::k_clazz[] =
    "gen_bto_dotprod<N, Traits, Timed>";


template<size_t N, typename Traits, typename Timed>
gen_bto_dotprod<N, Traits, Timed>::gen_bto_dotprod(
    gen_block_tensor_rd_i<N, bti_traits> &bt1,
    const tensor_transf_type &tr1,
    gen_block_tensor_rd_i<N, bti_traits> &bt2,
    const tensor_transf_type &tr2) :

    m_bis(bt1.get_bis()) {

    m_bis.match_splits();
    m_bis.permute(tr1.get_perm());
    add_arg(bt1, tr1, bt2, tr2);
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_dotprod<N, Traits, Timed>::add_arg(
    gen_block_tensor_rd_i<N, bti_traits> &bt1,
    const tensor_transf_type &tr1,
    gen_block_tensor_rd_i<N, bti_traits> &bt2,
    const tensor_transf_type &tr2) {

    static const char method[] = "add_arg()";

    block_index_space<N> bis1(bt1.get_bis()), bis2(bt2.get_bis());
    bis1.match_splits();
    bis2.match_splits();
    bis1.permute(tr1.get_perm());
    bis2.permute(tr2.get_perm());
    if(!m_bis.equals(bis1)) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
            "bt1");
    }
    if(!m_bis.equals(bis2)) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
            "bt2");
    }

    m_args.push_back(arg(bt1, tr1, bt2, tr2));
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_dotprod<N, Traits, Timed>::calculate(
    std::vector<element_type> &v) {

    static const char method[] = "calculate(std::vector<element_type>&)";

    if(v.size() != m_args.size()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "v");
    }

    gen_bto_dotprod::start_timer();

    try {

        typename std::list<arg>::const_iterator iarg = m_args.begin();
        for(size_t i = 0; iarg != m_args.end(); i++, ++iarg) {

            gen_block_tensor_rd_i<N, bti_traits> &bta = iarg->bt1;
            gen_block_tensor_rd_i<N, bti_traits> &btb = iarg->bt2;
            const tensor_transf<N, element_type> &tra = iarg->tr1;
            const tensor_transf<N, element_type> &trb = iarg->tr2;

            gen_block_tensor_rd_ctrl<N, bti_traits> cb(btb);
            const symmetry<N, element_type> &symb = cb.req_const_symmetry();

            gen_bto_aux_dotprod<N, Traits> out(bta, tra, symb);
            out.open();
            gen_bto_copy<N, Traits, Timed>(btb, trb).perform(out);
            out.close();
            v[i] = out.get_d();
        }

    } catch(...) {
        gen_bto_dotprod::stop_timer();
        throw;
    }

    gen_bto_dotprod::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_DOTPROD_IMPL_H
