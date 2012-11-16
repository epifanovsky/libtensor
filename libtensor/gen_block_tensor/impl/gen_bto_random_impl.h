#ifndef LIBTENSOR_GEN_BTO_RANDOM_IMPL_H
#define LIBTENSOR_GEN_BTO_RANDOM_IMPL_H

#include <libtensor/core/abs_index.h>
#include <libtensor/core/orbit_list.h>
#include "../gen_bto_random.h"

namespace libtensor {


template<size_t N, typename Traits, typename Timed>
const char *gen_bto_random<N, Traits, Timed>::k_clazz =
        "gen_bto_random<N, Traits, Timed>";


template<size_t N, typename Traits, typename Timed>
void gen_bto_random<N, Traits, Timed>::perform(
        gen_block_tensor_wr_i<N, bti_traits> &bt) throw(exception) {

    gen_bto_random::start_timer();

    dimensions<N> bidims(bt.get_bis().get_block_index_dims());
    gen_block_tensor_wr_ctrl<N, bti_traits> ctrl(bt);

    orbit_list<N, element_type> orblist(ctrl.req_symmetry());
    typename orbit_list<N, element_type>::iterator iorbit = orblist.begin();
    for(; iorbit != orblist.end(); iorbit++) {
        index<N> idx;
        orblist.get_index(iorbit, idx);
        make_random_blk(ctrl, bidims, idx);
    }

    gen_bto_random::stop_timer();
}

template<size_t N, typename Traits, typename Timed>
void gen_bto_random<N, Traits, Timed>::perform(
        gen_block_tensor_wr_i<N, bti_traits> &bt, const index<N> &idx)
        throw(exception) {

    gen_bto_random::start_timer();

    dimensions<N> bidims(bt.get_bis().get_block_index_dims());
    gen_block_tensor_wr_ctrl<N, bti_traits> ctrl(bt);
    make_random_blk(ctrl, bidims, idx);

    gen_bto_random::stop_timer();
}


template<size_t N, typename Traits, typename Timed>
bool gen_bto_random<N, Traits, Timed>::make_transf_map(
        const symmetry<N, element_type> &sym, const dimensions<N> &bidims,
        const index<N> &idx, const tensor_transf_type &tr,
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
            tensor_transf_type tr2(tr);
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


template<size_t N, typename Traits, typename Timed>
void gen_bto_random<N, Traits, Timed>::make_random_blk(
        gen_block_tensor_wr_ctrl<N, bti_traits> &ctrl,
        const dimensions<N> &bidims, const index<N> &idx) {

    typedef typename Traits::template to_add_type<N>::type to_add;
    typedef typename Traits::template to_copy_type<N>::type to_copy;
    typedef typename Traits::template to_random_type<N>::type to_random;
    typedef typename Traits::template temp_block_tensor_type<N>::type
            temp_block_tensor_type;

    const symmetry<N, element_type> &sym = ctrl.req_const_symmetry();
    size_t absidx = abs_index<N>::get_abs_index(idx, bidims);

    to_random randop;

    tensor_transf_type tr0;
    transf_map_t transf_map;

    bool allowed = make_transf_map(sym, bidims, idx, tr0, transf_map);
    typename transf_map_t::iterator ilst = transf_map.find(absidx);
    if(!allowed || ilst == transf_map.end()) {
        ctrl.req_zero_block(idx);
        return;
    }


    typename transf_list_t::iterator itr = ilst->second.begin();
    if(itr == ilst->second.end()) {
        wr_block_type &blk = ctrl.req_block(idx);
        gen_bto_random::start_timer("randop");
        randop.perform(true, blk);
        gen_bto_random::stop_timer("randop");
        ctrl.ret_block(idx);
    } else {
        temp_block_tensor_type btrnd(sym.get_bis()), btsymrnd(sym.get_bis());
        gen_block_tensor_ctrl<N, bti_traits> crnd(btrnd), csymrnd(btsymrnd);

        {
        wr_block_type &rnd = crnd.req_block(idx);

        gen_bto_random::start_timer("randop");
        randop.perform(true, rnd);
        gen_bto_random::stop_timer("randop");

        crnd.ret_block(idx);
        }

        scalar_transf_sum<element_type> tottr;
        {
        rd_block_type &rnd = crnd.req_const_block(idx);
        wr_block_type &symrnd = csymrnd.req_block(idx);

        tottr.add(itr->get_scalar_tr());
        to_add symop(rnd, *itr);

        for(itr++; itr != ilst->second.end(); itr++) {
            symop.add_op(rnd, *itr);
            tottr.add(itr->get_scalar_tr());
        }
        gen_bto_random::start_timer("symop");
        symop.perform(true, symrnd);
        gen_bto_random::stop_timer("symop");

        crnd.ret_const_block(idx);
        csymrnd.ret_block(idx);
        }
        crnd.req_zero_block(idx);

        {
        rd_block_type &symrnd = csymrnd.req_const_block(idx);
        wr_block_type &blk = ctrl.req_block(idx);

        scalar_transf<element_type> str(tottr.get_transf());
        if (str.is_zero()) str = scalar_transf<element_type>();
        else str.invert();
        tensor_transf<N, element_type> tr(permutation<N>(), str);

        gen_bto_random::start_timer("copy");
        to_copy(symrnd, tr).perform(true, blk);
        gen_bto_random::stop_timer("copy");

        csymrnd.ret_const_block(idx);
        ctrl.ret_block(idx);
        }

        csymrnd.req_zero_block(idx);
    }

}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_RANDOM_H
