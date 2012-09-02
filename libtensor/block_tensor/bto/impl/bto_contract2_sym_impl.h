#ifndef LIBTENSOR_BTO_CONTRACT2_SYM_IMPL_H
#define LIBTENSOR_BTO_CONTRACT2_SYM_IMPL_H

#include <libtensor/core/permutation_builder.h>
#include <libtensor/symmetry/so_dirprod.h>
#include <libtensor/symmetry/so_reduce.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include "../bto_contract2_sym.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename T>
bto_contract2_sym<N, M, K, T>::bto_contract2_sym(
    const contraction2<N, M, K> &contr, block_tensor_i<N + K, T> &bta,
    block_tensor_i<M + K, T> &btb) :

    m_bisc(contr, bta.get_bis(), btb.get_bis()), m_symc(m_bisc.get_bisc()) {

    block_tensor_ctrl<N + K, T> ca(bta);
    block_tensor_ctrl<M + K, T> cb(btb);

    make_symmetry(contr, bta.get_bis(), ca.req_const_symmetry(),
        btb.get_bis(), cb.req_const_symmetry());
}


template<size_t N, size_t M, size_t K, typename T>
bto_contract2_sym<N, M, K, T>::bto_contract2_sym(
    const contraction2<N, M, K> &contr, const block_index_space<N + K> &bisa,
    const symmetry<N + K, T> &syma, const block_index_space<M + K> &bisb,
    const symmetry<M + K, T> &symb) :

    m_bisc(contr, bisa, bisb), m_symc(m_bisc.get_bisc()) {

    make_symmetry(contr, bisa, syma, bisb, symb);
}


template<size_t N, size_t M, size_t K, typename T>
void bto_contract2_sym<N, M, K, T>::make_symmetry(
    const contraction2<N, M, K> &contr, const block_index_space<N + K> &bisa,
    const symmetry<N + K, T> &syma, const block_index_space<M + K> &bisb,
    const symmetry<M + K, T> &symb) {

    contraction2<N + K, M + K, 0> contr2;
    bto_contract2_bis<N + K, M + K, 0> xbis0(contr2, bisa, bisb);
    block_index_space<N + M + 2 * K> xbis(xbis0.get_bisc());

    const sequence<2 * (N + M + K), size_t> &conn = contr.get_conn();

    sequence<N + M + 2 * K, size_t> seq1(0), seq2(0), seq(0);
    mask<N + M + 2 * K> msk;
    for(size_t i = 0, k = 0; i < N + M + 2 * K; i++) {
        seq1[i] = i;
        if(conn[i + N + M] < N + M) { // remaining indexes
            seq2[conn[i + N + M]] = i;
        } else if(i < N + K) { // contracted indexes
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
    so_reduce<N + M + 2 * K, 2 * K, T>(xsymab, msk, seq, bir, ir).
        perform(m_symc);
}


template<size_t N, size_t K, typename T>
bto_contract2_sym<N, N, K, T>::bto_contract2_sym(
    const contraction2<N, N, K> &contr, block_tensor_i<N + K, T> &bta,
    block_tensor_i<N + K, T> &btb) :

    m_bisc(contr, bta.get_bis(), btb.get_bis()), m_symc(m_bisc.get_bisc()) {

    if(&bta == &btb) {
        block_tensor_ctrl<N + K, T> ca(bta);
        make_symmetry(contr, bta.get_bis(), ca.req_const_symmetry(),
            bta.get_bis(), ca.req_const_symmetry(), true);
    } else {
        block_tensor_ctrl<N + K, T> ca(bta), cb(btb);
        make_symmetry(contr, bta.get_bis(), ca.req_const_symmetry(),
            btb.get_bis(), cb.req_const_symmetry(), false);
    }
}


template<size_t N, size_t K, typename T>
bto_contract2_sym<N, N, K, T>::bto_contract2_sym(
    const contraction2<N, N, K> &contr, const block_index_space<N + K> &bisa,
    const symmetry<N + K, T> &syma, const block_index_space<N + K> &bisb,
    const symmetry<N + K, T> &symb, bool self) :

    m_bisc(contr, bisa, bisb), m_symc(m_bisc.get_bisc()) {

    make_symmetry(contr, bisa, syma, bisb, symb, self);
}


template<size_t N, size_t K, typename T>
void bto_contract2_sym<N, N, K, T>::make_symmetry(
    const contraction2<N, N, K> &contr, const block_index_space<N + K> &bisa,
    const symmetry<N + K, T> &syma, const block_index_space<N + K> &bisb,
    const symmetry<N + K, T> &symb, bool self) {

    contraction2<N + K, N + K, 0> contr2;
    bto_contract2_bis<N + K, N + K, 0> xbis0(contr2, bisa, bisb);
    block_index_space<N + N + 2 * K> xbis(xbis0.get_bisc());

    const sequence<2 * (N + N + K), size_t> &conn = contr.get_conn();

    sequence<2 * (N + K), size_t> seq1(0), seq2(0), seq(0);
    mask<2 * (N + K)> msk;
    for(size_t i = 0, k = 0; i < 2 * (N + K); i++) {
        seq1[i] = i;
        if(conn[i + 2 * N] < 2 * N) { // remaining indexes
            seq2[conn[i + 2 * N]] = i;
        } else if(i < N + K) { // contracted indexes
            size_t j = 2 * (N + k);
            msk[j] = msk[j + 1] = true;
            seq[j] = seq[j + 1] = k;
            seq2[j] = i;
            seq2[j + 1] = conn[i + 2 * N] - 2 * N;
            k++;
        }
    }
    permutation_builder<2 * (N + K)> pb(seq2, seq1);
    xbis.permute(pb.get_perm());
    symmetry<2 * (N + K), double> xsymab(xbis);

    so_dirprod<N + K, N + K, T>(syma, symb, pb.get_perm()).perform(xsymab);

    //  When a tensor is contracted with itself, there is additional
    //  perm symmetry

    if(self) {
        permutation<2 * (N + K)> permab(pb.get_perm(), true);
        for(size_t i = 0; i < N + K; i++) permab.permute(i, N + K + i);
        permab.permute(pb.get_perm());
        if(!permab.is_identity()) {
            scalar_transf<double> tr;
            xsymab.insert(se_perm<2 * (N + K), double>(permab, tr));
        }
    }

    dimensions<2 * (N + K)> bidims = xbis.get_block_index_dims();
    index<2 * (N + K)> bia, bib, ia, ib;
    for(size_t i = 0; i < 2 * (N + K); i++) bib[i] = bidims[i] - 1;
    dimensions<2 * (N + K)> bdims = xbis.get_block_dims(bib);
    for(size_t i = 0; i < 2 * (N + K); i++) ib[i] = bdims[i] - 1;

    index_range<2 * (N + K)> ir(ia, ib), bir(bia, bib);
    so_reduce<2 * (N + K), 2 * K, T>(xsymab, msk, seq, bir, ir).
        perform(m_symc);
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_CONTRACT2_SYM_IMPL_H
