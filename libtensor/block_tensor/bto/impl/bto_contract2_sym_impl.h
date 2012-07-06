#ifndef LIBTENSOR_BTO_CONTRACT2_SYM_IMPL_H
#define LIBTENSOR_BTO_CONTRACT2_SYM_IMPL_H

#include <libtensor/core/permutation_builder.h>
#include <libtensor/symmetry/so_dirprod.h>
#include <libtensor/symmetry/so_reduce.h>
#include "../bto_contract2_sym.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename T>
bto_contract2_sym<N, M, K, T>::bto_contract2_sym(
    const contraction2<N, M, K> &contr, const block_index_space<N + K> &bisa,
    const symmetry<N + K, T> &syma, const block_index_space<M + K> &bisb,
    const symmetry<M + K, T> &symb) :

    m_bisc(contr, bisa, bisb), m_symc(m_bisc.get_bisc()) {

    contraction2<N + K, M + K, 0> contr2;
    bto_contract2_bis<N + K, M + K, 0> xbis0(contr2, bisa, bisb);
    block_index_space<N + M + 2 * K> xbis(xbis0.get_bisc());

    const sequence<2 * (N + M + K), size_t> &conn = contr.get_conn();

    sequence<N + M + 2 * K, size_t> seq1(0), seq2(0), seq(0);
    mask<N + M + 2 * K> msk;
    for (size_t i = 0, k = 0; i < N + M + 2 * K; i++) {
        seq1[i] = i;
        if (conn[i + N + M] < N + M) { // remaining indexes
            seq2[conn[i + N + M]] = i;
        }
        else if (i < N + K) { // contracted indexes
            size_t j = N + M + 2 * k;
            msk[j] = msk[j + 1] = true;
            seq[j] = seq[j + 1] = k;
            seq2[j] = i;
            seq2[j + 1] = conn[i + N + M] - (N + M);
            k++;
        }
    }
    permutation_builder<N + M + 2 * K> pb(seq2, seq1);
    xbis.permute(pb.get_perm());
    symmetry<N + M + 2 * K, T> xsymab(xbis);

    so_dirprod<N + K, M + K, T>(syma, symb, pb.get_perm()).perform(xsymab);

    dimensions<N + M + 2 * K> bidims = xbis.get_block_index_dims();
    index<N + M + 2 * K> bia, bib, ia, ib;
    for(size_t i = 0; i < N + M + 2 * K; i++) bib[i] = bidims[i] - 1;
    dimensions<N + M + 2 * K> bdims = xbis.get_block_dims(bib);
    for(size_t i = 0; i < N + M + 2 * K; i++) ib[i] = bdims[i] - 1;

    index_range<N + M + 2 * K> ir(ia, ib), bir(bia, bib);
    so_reduce<N + M + 2 * K, 2 * K, T>(xsymab, msk,
            seq, bir, ir).perform(m_symc);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_CONTRACT2_SYM_IMPL_H
