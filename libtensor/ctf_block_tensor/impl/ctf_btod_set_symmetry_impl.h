#ifndef LIBTENSOR_CTF_BTOD_SET_SYMMETRY_IMPL_H
#define LIBTENSOR_CTF_BTOD_SET_SYMMETRY_IMPL_H

#include <algorithm>
#include <libtensor/ctf_dense_tensor/ctf_tod_set_symmetry.h>
#include "../ctf_symmetry_builder.h"
#include "ctf_btod_set_symmetry.h"

namespace libtensor {


template<size_t N>
const char ctf_btod_set_symmetry<N>::k_clazz[] = "ctf_btod_set_symmetry<N>";


template<size_t N>
void ctf_btod_set_symmetry<N>::perform(
    const std::vector<size_t> &blst,
    gen_block_tensor_i<N, bti_traits> &bt) {

    dimensions<N> bidims = bt.get_bis().get_block_index_dims();
    gen_block_tensor_ctrl<N, bti_traits> ctrl(bt);
    const symmetry<N, double> &sym = ctrl.req_const_symmetry();

    for(size_t i = 0; i < blst.size(); i++) {
        index<N> ii;
        abs_index<N>::get_index(blst[i], bidims, ii);
        transf_list<N, double> trl(sym, ii);
        bool zero = ctrl.req_is_zero_block(ii);
        ctf_dense_tensor_i<N, double> &blk = ctrl.req_block(ii);
        ctf_symmetry_builder<N, double> symbld(trl);
        ctf_tod_set_symmetry<N>(symbld.get_symmetry()).perform(zero, blk);
        ctrl.ret_block(ii);
    }
}


template<size_t N>
void ctf_btod_set_symmetry<N>::perform(
    const assignment_schedule<N, double> &sch,
    gen_block_tensor_i<N, bti_traits> &bt) {

    std::vector<size_t> blst;
    for(typename assignment_schedule<N, double>::iterator i = sch.begin();
        i != sch.end(); ++i) {
        blst.push_back(sch.get_abs_index(i));
    }

    perform(blst, bt);
}


template<size_t N>
void ctf_btod_set_symmetry<N>::perform(
    const addition_schedule<N, ctf_btod_traits> &asch,
    gen_block_tensor_i<N, bti_traits> &bt) {

    std::vector<size_t> blst;

    typedef typename addition_schedule<N, ctf_btod_traits>::node node;

    for(typename addition_schedule<N, ctf_btod_traits>::iterator igrp =
        asch.begin(); igrp != asch.end(); ++igrp) {

        const typename addition_schedule<N, ctf_btod_traits>::schedule_group
            &grp = asch.get_node(igrp);
        for(typename std::list<node>::const_iterator i = grp.begin();
            i != grp.end(); ++i) {
            blst.push_back(i->cic);
        }
    }
    std::sort(blst.begin(), blst.end());
    blst.resize(std::distance(blst.begin(),
        std::unique(blst.begin(), blst.end())));

    perform(blst, bt);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_SET_SYMMETRY_IMPL_H
