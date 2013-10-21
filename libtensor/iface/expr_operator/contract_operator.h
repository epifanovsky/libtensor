#ifndef LIBTENSOR_IFACE_CONTRACT_OPERATOR_H
#define LIBTENSOR_IFACE_CONTRACT_OPERATOR_H

#include "../expr_core/contract2_core.h"

namespace libtensor {
namespace iface {


/** \brief Contraction of two expressions over multiple indexes
    \tparam K Number of contracted indexes.
    \tparam N Order of the first %tensor.
    \tparam M Order of the second %tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t K, size_t N, size_t M, typename T>
expr_rhs<N + M - 2 * K, T> contract(
    const letter_expr<K> contr,
    expr_rhs<N, T> bta,
    expr_rhs<M, T> btb) {

    sequence<2 * K, size_t> cseq(0);
    std::vector<const letter *> label;
    for (size_t i = 0, j = 0; i < N; i++) {
        const letter &l = bta.letter_at(i);
        if (contr.contains(l)) { cseq[contr.index_of(l)] = i; }
        else { label.push_back(&l); }
    }
    for (size_t i = 0, j = N - K; i < M; i++) {
        const letter &l = btb.letter_at(i);
        if (contr.contains(l)) { cseq[contr.index_of(l) + K] = i; }
        else { label.push_back(&l); }
    }

    expr_core_ptr<N + M - 2 * K, T> core(
            new contract2_core<N, M, K>(cseq, bta.get_core(), btb.get_core()));
    return expr_rhs<N + M - 2 * K, T>(core, letter_expr<N + M - 2 * K>(label));
}

/** \brief Contraction of two expressions over one index
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N + M - 2, T> contract(
    const letter &let,
    expr_rhs<N, T> bta,
    expr_rhs<M, T> btb) {

    return contract(letter_expr<1>(let), bta, btb);
}

#if 0
template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
expr_rhs<N1 + N2 + N3 - 2 * K1 - 2 * K2, T> contract(
    const letter_expr<K1> contr1,
    expr_rhs<N1, T> bta,
    expr_rhs<N2, T> btb,
    const letter_expr<K2> contr2,
    expr_rhs<N3, T> btc) {

    sequence<2 * K1, size_t> cs1(0);
    sequence<2 * K2, size_t> cs1(0);

    std::vector<const letter *> label(N + M - 2 * K);
    for (size_t i = 0, j = 0; i < N; i++) {
        const letter &l = bta.letter_at(i);
        if (contr.contains(l)) { cseq[contr.index_of(l)] = i; }
        else { label[j++] = &l; }
    }
    for (size_t i = 0, j = N - K; i < M; i++) {
        const letter &l = btb.letter_at(i);
        if (contr.contains(l)) { cseq[contr.index_of(l) + K] = i; }
        else { label[j++] = &l; }
    }

    expr_core_ptr<N1 + N2 + N3 - 2 * K1 - 2 * K2, T> core(
            new contract_core<N, M, K>(cseq, bta.get_core(), btb.get_core()));
    return expr_rhs<N + M - 2 * K, T>(core, letter_expr<N + M - 2 * K>(label));

}


template<size_t N1, size_t N2, size_t N3, typename T, bool A1, bool A2, bool A3>
expr_rhs<N1 + N2 + N3 - 4, T> contract(
    const letter &let1,
    expr_rhs<N1, T, A1> bta,
    expr_rhs<N2, T, A2> btb,
    const letter &let2,
    expr_rhs<N3, T, A3> btc) {

    return contract(
        letter_expr<1>(let1),
        expr_rhs<N1, T>(ident_core<N1, T, A1>(bta)),
        expr_rhs<N2, T>(ident_core<N2, T, A2>(btb)),
        letter_expr<1>(let2),
        expr_rhs<N2, T>(ident_core<N3, T, A3>(btc)));
}
#endif


} // namespace iface

using iface::contract;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_CONTRACT_OPERATOR_H
