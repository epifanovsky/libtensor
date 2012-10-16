#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_SYM_IMPL_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_SYM_IMPL_H

#include <libtensor/core/permutation_builder.h>
#include <libtensor/symmetry/so_dirprod.h>
#include <libtensor/symmetry/so_reduce.h>
#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include "gen_bto_contract2_sym.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename Traits>
gen_bto_contract2_sym<N, M, K, Traits>::gen_bto_contract2_sym(
    const contraction2<N, M, K> &contr,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    gen_block_tensor_rd_i<NB, bti_traits> &btb) :

    m_bis(contr, bta.get_bis(), btb.get_bis()), m_sym(m_bis.get_bis()) {

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(bta);
    gen_block_tensor_rd_ctrl<NB, bti_traits> cb(btb);

    make_symmetry(contr, ca.req_const_symmetry(), cb.req_const_symmetry());
}


template<size_t N, size_t M, size_t K, typename Traits>
gen_bto_contract2_sym<N, M, K, Traits>::gen_bto_contract2_sym(
    const contraction2<N, M, K> &contr,
    const symmetry<NA, element_type> &syma,
    const symmetry<NB, element_type> &symb) :

    m_bis(contr, syma.get_bis(), symb.get_bis()),
    m_sym(m_bis.get_bis()) {

    make_symmetry(contr, syma, symb);
}


template<size_t N, size_t M, size_t K, typename Traits>
void gen_bto_contract2_sym<N, M, K, Traits>::make_symmetry(
    const contraction2<N, M, K> &contr,
    const symmetry<NA, element_type> &syma,
    const symmetry<NB, element_type> &symb) {

    contraction2<NA, NB, 0> contr2;
    gen_bto_contract2_bis<NA, NB, 0> xbis0(contr2,
            syma.get_bis(), symb.get_bis());
    block_index_space<NA + NB> xbis(xbis0.get_bis());

    const sequence<NA + NB + NC, size_t> &conn = contr.get_conn();

    sequence<NA + NB, size_t> seq1(0), seq2(0), seq(0);
    mask<NA + NB> msk;
    for(size_t i = 0, k = 0; i < NA + NB; i++) {
        seq1[i] = i;
        if(conn[i + NC] < NC) { // remaining indexes
            seq2[conn[i + NC]] = i;
        } else if(i < NA) { // contracted indexes
            size_t j = N + M + 2 * k;
            msk[j] = msk[j + 1] = true;
            seq[j] = seq[j + 1] = k;
            seq2[j] = i;
            seq2[j + 1] = conn[i + NC] - NC;
            k++;
        }
    }
    permutation_builder<NA + NB> pb(seq2, seq1);
    xbis.permute(pb.get_perm());
    symmetry<NA + NB, element_type> xsymab(xbis);

    so_dirprod<NA, NB, element_type>(syma, symb, pb.get_perm()).perform(xsymab);

    dimensions<NA + NB> bidims = xbis.get_block_index_dims();
    index<NA + NB> bia, bib, ia, ib;
    for(size_t i = 0; i < NA + NB; i++) bib[i] = bidims[i] - 1;
    dimensions<NA + NB> bdims = xbis.get_block_dims(bib);
    for(size_t i = 0; i < NA + NB; i++) ib[i] = bdims[i] - 1;

    index_range<NA + NB> ir(ia, ib), bir(bia, bib);
    so_reduce<NA + NB, 2 * K, element_type>(xsymab, msk, seq, bir, ir).
        perform(m_sym);
}


template<size_t N, size_t K, typename Traits>
gen_bto_contract2_sym<N, N, K, Traits>::gen_bto_contract2_sym(
    const contraction2<N, N, K> &contr,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    gen_block_tensor_rd_i<NB, bti_traits> &btb) :

    m_bis(contr, bta.get_bis(), btb.get_bis()), m_sym(m_bis.get_bis()) {

    if(&bta == &btb) {
        gen_block_tensor_rd_ctrl<NA, bti_traits> ca(bta);
        make_symmetry(contr, ca.req_const_symmetry(),
            ca.req_const_symmetry(), true);
    } else {
        gen_block_tensor_rd_ctrl<NB, bti_traits> ca(bta), cb(btb);
        make_symmetry(contr, ca.req_const_symmetry(),
            cb.req_const_symmetry(), false);
    }
}


template<size_t N, size_t K, typename Traits>
gen_bto_contract2_sym<N, N, K, Traits>::gen_bto_contract2_sym(
    const contraction2<N, N, K> &contr,
    const symmetry<NA, element_type> &syma,
    const symmetry<NB, element_type> &symb,
    bool self) :

    m_bis(contr, syma.get_bis(), symb.get_bis()),
    m_sym(m_bis.get_bis()) {

    make_symmetry(contr, syma, symb, self);
}


template<size_t N, size_t K, typename Traits>
void gen_bto_contract2_sym<N, N, K, Traits>::make_symmetry(
    const contraction2<N, N, K> &contr,
    const symmetry<NA, element_type> &syma,
    const symmetry<NB, element_type> &symb,
    bool self) {

    contraction2<NA, NB, 0> contr2;
    gen_bto_contract2_bis<NA, NB, 0> xbis0(contr2,
            syma.get_bis(), symb.get_bis());
    block_index_space<NA + NB> xbis(xbis0.get_bis());

    const sequence<NA + NB + NC, size_t> &conn = contr.get_conn();

    sequence<NA + NB, size_t> seq1(0), seq2(0), seq(0);
    mask<NA + NB> msk;
    for(size_t i = 0, k = 0; i < NA + NB; i++) {
        seq1[i] = i;
        if(conn[i + NC] < NC) { // remaining indexes
            seq2[conn[i + NC]] = i;
        } else if(i < NA) { // contracted indexes
            size_t j = 2 * (N + k);
            msk[j] = msk[j + 1] = true;
            seq[j] = seq[j + 1] = k;
            seq2[j] = i;
            seq2[j + 1] = conn[i + NC] - NC;
            k++;
        }
    }
    permutation_builder<NA + NB> pb(seq2, seq1);
    xbis.permute(pb.get_perm());
    symmetry<NA + NB, element_type> xsymab(xbis);

    so_dirprod<NA, NB, element_type>(syma, symb, pb.get_perm()).perform(xsymab);

    //  When a tensor is contracted with itself, there is additional
    //  perm symmetry

    if(self) {
        permutation<NA + NB> permab(pb.get_perm(), true);
        for(size_t i = 0; i < NB; i++) permab.permute(i, NA + i);
        permab.permute(pb.get_perm());
        if(!permab.is_identity()) {
            scalar_transf<element_type> tr;
            xsymab.insert(se_perm<NA + NB, element_type>(permab, tr));
        }
    }

    dimensions<NA + NB> bidims = xbis.get_block_index_dims();
    index<NA + NB> bia, bib, ia, ib;
    for(size_t i = 0; i < NA + NB; i++) bib[i] = bidims[i] - 1;
    dimensions<NA + NB> bdims = xbis.get_block_dims(bib);
    for(size_t i = 0; i < NA + NB; i++) ib[i] = bdims[i] - 1;

    index_range<NA + NB> ir(ia, ib), bir(bia, bib);
    so_reduce<NA + NB, 2 * K, element_type>(xsymab, msk, seq, bir, ir).
        perform(m_sym);
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_SYM_IMPL_H
