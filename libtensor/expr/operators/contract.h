#ifndef LIBTENSOR_EXPR_OPERATORS_CONTRACT_H
#define LIBTENSOR_EXPR_OPERATORS_CONTRACT_H

#include <libtensor/expr/dag/node_contract.h>
#include <libtensor/expr/dag/node_transform.h>

namespace libtensor {
namespace expr {


/** \brief Contraction of two expressions over multiple indices
    \tparam K Number of contracted indices.
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_expr_operators
 **/
template<size_t K, size_t N, size_t M, typename T>
expr_rhs<N + M - 2 * K, T> contract(
    const label<K> &contr,
    const expr_rhs<N, T> &a,
    const expr_rhs<M, T> &b) {

    std::multimap<size_t, size_t> cseq;
    std::vector<const letter*> lab;
    for(size_t i = 0; i < N; i++) {
        const letter &l = a.letter_at(i);
        if(contr.contains(l))
            cseq.insert(std::pair<size_t, size_t>(i, N + b.index_of(l)));
        else lab.push_back(&l);
    }
    for(size_t i = 0; i < M; i++) {
        const letter &l = b.letter_at(i);
        if(!contr.contains(l)) lab.push_back(&l);
    }

    enum {
        NC = N + M - 2 * K
    };

    expr_tree e(expr::node_contract(NC, cseq, true));
    expr_tree::node_id_t id = e.get_root();
    e.add(id, a.get_expr());
    e.add(id, b.get_expr());

    return expr_rhs<NC, T>(e, label<NC>(lab));
}


/** \brief Contraction of two expressions over one index
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_expr_operators
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N + M - 2, T> contract(
    const letter &let,
    const expr_rhs<N, T> &a,
    const expr_rhs<M, T> &b) {

    return contract(label<1>(let), a, b);
}


/** \brief Contraction of three expressions over multiple indices
    \tparam N1 Order of the first expression.
    \tparam N2 Order of the second expression.
    \tparam N3 Order of the third expression.
    \tparam K1 Number of contracted indices of first two expressions.
    \tparam K2 Number of contracted indices of the remaining expressions.
    \tparam T Tensor element type.

    \ingroup libtensor_expr_operators
 **/
template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2, typename T>
expr_rhs<N1 + N2 + N3 - 2 * K1 - 2 * K2, T> contract(
    const label<K1> contr1,
    expr_rhs<N1, T> bta,
    expr_rhs<N2, T> btb,
    const label<K2> contr2,
    expr_rhs<N3, T> btc) {

    typedef std::multimap<size_t, size_t>::value_type value_type;

    std::multimap<size_t, size_t> cseq;
    std::vector<const letter *> lab;
    for (size_t i = 0; i < N1; i++) {
        const letter &l = bta.letter_at(i);
        if (contr1.contains(l)) {
            cseq.insert(value_type(i, N1 + btb.index_of(l)));
            continue;
        }
        if (contr2.contains(l)) {
            cseq.insert(value_type(i, N1 + N2 + btc.index_of(l)));
            continue;
        }
        lab.push_back(&l);
    }
    for (size_t i = 0; i < N2; i++) {
        const letter &l = btb.letter_at(i);
        if (contr1.contains(l)) continue;

        if (contr2.contains(l)) {
            cseq.insert(value_type(N1 + i, N1 + N2 + btc.index_of(l)));
            continue;
        }
        lab.push_back(&l);
    }
    for (size_t i = 0; i < N3; i++) {
        const letter &l = btc.letter_at(i);
        if (contr1.contains(l) || contr2.contains(l)) continue;

        lab.push_back(&l);
    }

    enum {
        N = N1 + N2 + N3 - 2 * K1 - 2 * K2
    };

    expr_tree e(node_contract(N, cseq, true));
    expr_tree::node_id_t id = e.get_root();
    e.add(id, bta.get_expr());
    e.add(id, btb.get_expr());
    e.add(id, btc.get_expr());

    return expr_rhs<N, T>(e, label<N>(lab));
}


/** \brief Contraction of three expressions over two indices
    \tparam N1 Order of the first expression.
    \tparam N2 Order of the second expression.
    \tparam N3 Order of the third expression.
    \tparam T Tensor element type.

    \ingroup libtensor_expr_operators
 **/
template<size_t N1, size_t N2, size_t N3, typename T>
expr_rhs<N1 + N2 + N3 - 4, T> contract(
    const letter &let1,
    expr_rhs<N1, T> bta,
    expr_rhs<N2, T> btb,
    const letter &let2,
    expr_rhs<N3, T> btc) {

    return contract(label<1>(let1), bta, btb, label<1>(let2), btc);
}


} // namespace expr
} // namespace libtensor


namespace libtensor {

using expr::contract;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_OPERATORS_CONTRACT_H
