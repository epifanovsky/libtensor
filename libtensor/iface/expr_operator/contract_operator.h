#ifndef LIBTENSOR_IFACE_CONTRACT_OPERATOR_H
#define LIBTENSOR_IFACE_CONTRACT_OPERATOR_H

#include <libtensor/expr/node_contract.h>
#include <libtensor/expr/node_transform.h>

namespace libtensor {
namespace iface {


/** \brief Contraction of two expressions over multiple indices
    \tparam K Number of contracted indices.
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_iface
 **/
template<size_t K, size_t N, size_t M, typename T>
expr_rhs<N + M - 2 * K, T> contract(
    const letter_expr<K> &contr,
    const expr_rhs<N, T> &a,
    const expr_rhs<M, T> &b) {

    const expr_tree &ea = a.get_expr(), &eb = b.get_expr();
    tensor_list tl(ea.get_tensors());
    tl.merge(eb.get_tensors());

    std::map<size_t, size_t> cseq;
    std::vector<const letter*> label;
    for(size_t i = 0; i < N; i++) {
        const letter &l = a.letter_at(i);
        if(contr.contains(l)) cseq[i] = N + b.index_of(l);
        else label.push_back(&l);
    }
    for(size_t i = 0; i < M; i++) {
        const letter &l = b.letter_at(i);
        if(!contr.contains(l)) label.push_back(&l);
    }

    enum {
        NC = N + M - 2 * K
    };

    expr::node_contract ncon(ea.get_nodes(), eb.get_nodes(), cseq);
    return expr_rhs<NC, T>(expr_tree(ncon, tl), letter_expr<NC>(label));
}


/** \brief Contraction of two expressions over one index
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_iface
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N + M - 2, T> contract(
    const letter &let,
    const expr_rhs<N, T> &a,
    const expr_rhs<M, T> &b) {

    return contract(letter_expr<1>(let), a, b);
}


#if 0
template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
expr_rhs<N1 + N2 + N3 - 2 * K1 - 2 * K2, T> contract(
    const letter_expr<K1> contr1,
    expr_rhs<N1, T> bta,
    expr_rhs<N2, T> btb,
    const letter_expr<K2> contr2,
    expr_rhs<N3, T> btc) {

    const expr_tree &exa = bta.get_expr();
    const expr_tree &exb = btb.get_expr();
    const expr_tree &exc = btc.get_expr();

    tensor_list tl(exa.get_tensors());
    tl.merge(exb.get_tensors());
    tl.merge(exc.get_tensors());

    std::map<size_t, size_t> cseq;
    std::vector<const letter *> label;
    for (size_t i = 0; i < N1; i++) {
        const letter &l = bta.letter_at(i);
        if (contr1.contains(l)) {
            cseq[i] = N1 + btb.index_of(l); continue;
        }
        if (contr2.contains(l)) {
            cseq[i] = N1 + N2 + btc.index_of(l); continue;
        }
        label.push_back(&l);
    }
    for (size_t i = 0; i < N2; i++) {
        const letter &l = btb.letter_at(i);
        if (contr1.contains(l)) continue;

        if (contr2.contains(l)) {
            cseq[N1 + i] = N1 + N2 + btc.index_of(l); continue;
        }
        label.push_back(&l);
    }
    for (size_t i = 0; i < N3; i++) {
        const letter &l = btc.letter_at(i);
        if (contr1.contains(l) || contr2.contains(l)) continue;

        label.push_back(&l);
    }

    return expr_rhs<N1 + N2 + N3 - 2 * K1 - 2 * K2, T>(
            expr_tree(expr::node_contract(exa.get_nodes(),
                    exb.get_nodes(), exb.get_nodes(), cseq), tl),
            letter_expr<N1 + N2 + N3 - 2 * K1 - 2 * K2, T>(label));
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
