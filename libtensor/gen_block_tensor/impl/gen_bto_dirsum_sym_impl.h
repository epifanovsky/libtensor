#ifndef LIBTENSOR_GEN_BTO_DIRSUM_SYM_IMPL_H
#define LIBTENSOR_GEN_BTO_DIRSUM_SYM_IMPL_H

#include <libtensor/symmetry/so_dirsum.h>
#include "gen_bto_dirsum_sym.h"

namespace libtensor {


template<size_t N, size_t M, typename Traits>
gen_bto_dirsum_sym<N, M, Traits>::gen_bto_dirsum_sym(
        gen_block_tensor_rd_i<N, bti_traits> &bta,
        const scalar_transf<element_type> &ka,
        gen_block_tensor_rd_i<M, bti_traits> &btb,
        const scalar_transf<element_type> &kb,
        const permutation<NC> &permc) :
        m_bisc(bta.get_bis(), btb.get_bis(), permc), m_symc(m_bisc.get_bis()) {

    gen_block_tensor_rd_ctrl<N, bti_traits> ca(bta);
    gen_block_tensor_rd_ctrl<M, bti_traits> cb(btb);

    so_dirsum<N, M, element_type>(
            ca.req_const_symmetry(),
            cb.req_const_symmetry(), permc).perform(m_symc);
}


template<size_t N, size_t M, typename Traits>
gen_bto_dirsum_sym<N, M, Traits>::gen_bto_dirsum_sym(
        block_index_space<N> &bisa,
        const symmetry<N, element_type> &syma,
        const scalar_transf<element_type> &ka,
        block_index_space<M> &bisb,
        const symmetry<M, element_type> &symb,
        const scalar_transf<element_type> &kb,
        const permutation<NC> &permc) :
        m_bisc(bisa, bisb, permc), m_symc(m_bisc.get_bis()) {

    so_dirsum<N, M, element_type>(syma, symb, permc).perform(m_symc);
}


template<size_t N, typename Traits>
gen_bto_dirsum_sym<N, N, Traits>::gen_bto_dirsum_sym(
        gen_block_tensor_rd_i<N, bti_traits> &bta,
        const scalar_transf<element_type> &ka,
        gen_block_tensor_rd_i<N, bti_traits> &btb,
        const scalar_transf<element_type> &kb,
        const permutation<NC> &permc) :
        m_bisc(bta.get_bis(), btb.get_bis(), permc),
        m_symc(m_bisc.get_bis()) {

    gen_block_tensor_rd_ctrl<N, bti_traits> ca(bta), cb(btb);

    make_symmetry(ca.req_const_symmetry(), ka, cb.req_const_symmetry(), kb,
            permc, (&bta == &btb));
}


template<size_t N, typename Traits>
gen_bto_dirsum_sym<N, N, Traits>::gen_bto_dirsum_sym(
        block_index_space<N> &bisa,
        const symmetry<N, element_type> &syma,
        const scalar_transf<element_type> &ka,
        block_index_space<N> &bisb,
        const symmetry<N, element_type> &symb,
        const scalar_transf<element_type> &kb,
        const permutation<NC> &permc) :
        m_bisc(bisa, bisb, permc),
        m_symc(m_bisc.get_bis()) {

    make_symmetry(syma, ka, symb, kb, permc, self);
}

template<size_t N, typename Traits>
void gen_bto_dirsum_sym<N, N, Traits>::make_symmetry(
        const symmetry<N, element_type> &syma,
        const scalar_transf<element_type> &ka,
        const symmetry<N, element_type> &symb,
        const scalar_transf<element_type> &kb,
        const permutation<NC> &permc, bool self) {

    so_dirsum<N, N, element_type>(syma, symb, permc).perform(m_symc);

    if (self) {

        scalar_transf<element_type> dk(kb), ka_inv(ka);
        ka_inv.invert();
        dk.transform(ka_inv);

        scalar_transf<element_type> dk2(dk);
        dk2.transform(dk);

        if (! dk2.is_identity()) return;

        permutation<NC> perm;
        for (size_t i = 0; i < N; i++) perm.permute(i, i + N);

        sequence<N + N, size_t> seq1(0), seq2(0);
        for (size_t i = 0; i < N; i++) seq1[i] = seq2[i + N] = i;
        for (size_t i = 0; i < N; i++) seq1[i + N] = seq2[i] = i + N;
        permc.apply(seq1);
        permc.apply(seq2);

        permutation_builder<N + N> pb(seq2, seq1);
        se_perm<N + N, element_type> sp(pb.get_perm(), dk);

        m_symc.insert(sp);
    }
}


} // namespace libtensor


#endif // LIBTENOSR_GEN_BTO_DIRSUM_IMPL_H
