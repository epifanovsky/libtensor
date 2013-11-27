#ifndef LIBTENSOR_GEN_BTO_SYMCONTRACT2_SYM_IMPL_H
#define LIBTENSOR_GEN_BTO_SYMCONTRACT2_SYM_IMPL_H

#include <libtensor/symmetry/so_symmetrize.h>
#include "gen_bto_symcontract2_sym.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename Traits>
const char gen_bto_symcontract2_sym<N, M, K, Traits>::k_clazz[] =
    "gen_bto_symcontract2_sym<N, M, K, Traits>";

template<size_t N, size_t M, size_t K, typename Traits>
gen_bto_symcontract2_sym<N, M, K, Traits>::gen_bto_symcontract2_sym(
    const contraction2<N, M, K> &contr,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    gen_block_tensor_rd_i<NB, bti_traits> &btb,
    const permutation<NC> &perm,
    bool symm) :

    m_symbld(contr, bta, btb), m_sym(m_symbld.get_bis()) {

    make_symmetry(perm, symm);
}


template<size_t N, size_t M, size_t K, typename Traits>
gen_bto_symcontract2_sym<N, M, K, Traits>::gen_bto_symcontract2_sym(
    const contraction2<N, M, K> &contr,
    const symmetry<NA, element_type> &syma,
    const symmetry<NB, element_type> &symb,
    const permutation<NC> &perm,
    bool symm) :

    m_symbld(contr, syma, symb), m_sym(m_symbld.get_bis()) {

    make_symmetry(perm, symm);
}


template<size_t N, size_t M, size_t K, typename Traits>
void gen_bto_symcontract2_sym<N, M, K, Traits>::make_symmetry(
    const permutation<NC> &perm,
    bool symm) {

    static const char method[] = "make_symmetry()";

    permutation<NC> p1(perm); p1.permute(perm);
    if(perm.is_identity() || !p1.is_identity()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "perm");
    }

    sequence<NC, size_t> seq2(0), idxgrp(0), symidx(0);
    for(size_t i = 0; i < NC; i++) seq2[i] = i;
    perm.apply(seq2);

    size_t idx = 1;
    for(size_t i = 0; i < NC; i++) {
        if(seq2[i] <= i) continue;
        idxgrp[i] = 1;
        idxgrp[seq2[i]] = 2;
        symidx[i] = symidx[seq2[i]] = idx++;
    }
    scalar_transf<element_type> tr(symm ? 1.0 : -1.0);
    so_symmetrize<NC, element_type>(m_symbld.get_symmetry(), idxgrp, symidx,
        tr, tr).perform(m_sym);
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SYMCONTRACT2_SYM_IMPL_H
