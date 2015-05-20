#ifndef LIBTENSOR_GEN_BTO_SET_ELEM_IMPL_H
#define LIBTENSOR_GEN_BTO_SET_ELEM_IMPL_H

#include <libtensor/exception.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/orbit.h>
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_set_elem.h"

namespace libtensor {


template<size_t N, typename Traits>
const char *gen_bto_set_elem<N, Traits>::k_clazz =
        "gen_bto_set_elem<N, Traits>";


template<size_t N, typename Traits>
void gen_bto_set_elem<N, Traits>::perform(
        gen_block_tensor_i<N, bti_traits> &bt, const index<N> &bidx,
        const index<N> &idx, const element_type &d) {

    typedef typename Traits::template to_set_type<N>::type to_set;
    typedef typename Traits::template to_set_elem_type<N>::type to_set_elem;

    static const char *method = "perform("
            "gen_block_tensor_wr_i<N, bti_traits> &, const index<N> &, "
            "const index<N> &, const element_type &)";

    gen_block_tensor_ctrl<N, bti_traits> ctrl(bt);

    dimensions<N> bidims(bt.get_bis().get_block_index_dims());
    orbit<N, element_type> o(ctrl.req_const_symmetry(), bidx);

    if (! o.is_allowed())
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Block index not allowed by symmetry.");

    const tensor_transf<N, element_type> &tr = o.get_transf(bidx);
    abs_index<N> abidx(o.get_acindex(), bidims);

    bool zero = ctrl.req_is_zero_block(abidx.get_index());
    wr_block_type &blk = ctrl.req_block(abidx.get_index());

    if(zero) to_set().perform(zero, blk);

    permutation<N> perm(tr.get_perm(), true);
    index<N> idx1(idx); idx1.permute(perm);
    scalar_transf<element_type> trx(tr.get_scalar_tr());
    trx.invert();
    element_type d1(d);
    trx.apply(d1);

    transf_map_t trmap;
    tensor_transf<N, element_type> tr0;
    make_transf_map(ctrl.req_const_symmetry(), bidims, abidx.get_index(),
        tr0, trmap);
    typename transf_map_t::iterator ilst =
        trmap.find(abidx.get_abs_index());
    for(typename transf_list_t::iterator itr = ilst->second.begin();
        itr != ilst->second.end(); itr++) {

        index<N> idx2(idx1);
        idx2.permute(itr->get_perm());
        element_type d2(d1);
        itr->get_scalar_tr().apply(d2);
        to_set_elem().perform(blk, idx2, d2);
    }

    ctrl.ret_block(abidx.get_index());
}


template<size_t N, typename Traits>
bool gen_bto_set_elem<N, Traits>::make_transf_map(
        const symmetry<N, element_type> &sym, const dimensions<N> &bidims,
        const index<N> &idx, const tensor_transf<N, element_type> &tr,
        transf_map_t &alltransf) {

    size_t absidx = abs_index<N>::get_abs_index(idx, bidims);
    typename transf_map_t::iterator ilst = alltransf.find(absidx);
    if(ilst == alltransf.end()) {
        ilst = alltransf.insert(std::pair<size_t, transf_list_t>(
            absidx, transf_list_t())).first;
    }
    typename transf_list_t::iterator itr = ilst->second.begin();
    bool done = false;
    for(; itr != ilst->second.end(); itr++) {
        if(*itr == tr) {
            done = true;
            break;
        }
    }
    if(done) return true;
    ilst->second.push_back(tr);

    bool allowed = true;
    for(typename symmetry<N, element_type>::iterator iset = sym.begin();
        iset != sym.end(); iset++) {

        const symmetry_element_set<N, element_type> &eset =
            sym.get_subset(iset);
        for(typename symmetry_element_set<N, element_type>::const_iterator
            ielem = eset.begin(); ielem != eset.end(); ielem++) {

            const symmetry_element_i<N, element_type> &elem =
                eset.get_elem(ielem);
            index<N> idx2(idx);
            tensor_transf<N, element_type> tr2(tr);
            if(elem.is_allowed(idx2)) {
                elem.apply(idx2, tr2);
                allowed = make_transf_map(sym, bidims,
                    idx2, tr2, alltransf);
            } else {
                allowed = false;
            }
        }
    }
    return allowed;
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SET_ELEM_IMPL_H
