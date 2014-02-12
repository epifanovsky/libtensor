#ifndef LIBTENSOR_EXPR_OPERATORS_EWMULT_H
#define LIBTENSOR_EXPR_OPERATORS_EWMULT_H

#include <libtensor/expr/dag/node_contract.h>
#include <libtensor/expr/dag/node_div.h>
#include <libtensor/expr/dag/node_transform.h>
#include <libtensor/expr/expr_exception.h>

namespace libtensor {
namespace expr {


/** \brief Element-wise multiplication of two expressions

    \ingroup libtensor_expr_operators
 **/
template<size_t N, typename T>
expr_rhs<N, T> mult(
    const expr_rhs<N, T> &lhs,
    const expr_rhs<N, T> &rhs) {

    std::multimap<size_t, size_t> map;

    permutation<N> p = lhs.get_label().permutation_of(rhs.get_label());
    for (size_t i = 0; i < N; i++) {
        map.insert(std::pair<size_t, size_t>(i, p[i]));
    }

    expr_tree e(node_contract(N, map, false));
    expr_tree::node_id_t id = e.get_root();
    e.add(id, lhs.get_expr());
    e.add(id, rhs.get_expr());

    return expr_rhs<N, T>(e, lhs.get_label());
}


/** \brief Element-wise division of two expressions

    \ingroup libtensor_expr_operators
 **/
template<size_t N, typename T>
expr_rhs<N, T> div(
    const expr_rhs<N, T> &lhs,
    const expr_rhs<N, T> &rhs) {

    node_div n1(N);
    expr_tree e(n1);
    expr_tree::node_id_t id = e.get_root();
    e.add(id, lhs.get_expr());

    permutation<N> px = lhs.get_label().permutation_of(rhs.get_label());
    if (! px.is_identity()) {
        std::vector<size_t> perm(N);
        for (size_t i = 0; i < N; i++) perm[i] = px[i];

        e.add(id, node_transform<T>(perm, scalar_transf<T>()));
        id = e.get_edges_out(id).back();
    }

    e.add(id, rhs.get_expr());
    return expr_rhs<N, T>(e, lhs.get_label());
}


/** \brief Element-wise multiplication of two expressions

    \ingroup libtensor_expr_operators
 **/
template<size_t N, size_t M, size_t K, typename T>
expr_rhs<N + M - K, T> ewmult(
    const label<K> &ewidx,
    const expr_rhs<N, T> &lhs,
    const expr_rhs<M, T> &rhs) {

    static const char method[] = "ewmult(const letter_expr<K> &, "
            "const expr_rhs<N, T> &, const expr_rhs<M, T> &)";

    std::multimap<size_t, size_t> map;
    std::vector<const letter *> lab;
    for(size_t i = 0; i < K; i++) {
        const letter &l = ewidx.letter_at(i);
        if(!lhs.contains(l) || !rhs.contains(l)) {
            throw expr_exception(g_ns, "", method, __FILE__, __LINE__,
                    "Letter not found.");
        }
        map.insert(std::pair<size_t, size_t>(
                lhs.index_of(l), rhs.index_of(l)));
    }

    enum {
        NC = N + M - K
    };

    for(size_t i = 0; i < N; i++) lab.push_back(&lhs.letter_at(i));
    for(size_t i = 0; i < M; i++) {
        const letter &l = rhs.letter_at(i);
        if(!ewidx.contains(l)) lab.push_back(&l);
    }

    expr_tree e(node_contract(NC, map, false));
    expr_tree::node_id_t id = e.get_root();
    e.add(id, lhs.get_expr());
    e.add(id, rhs.get_expr());

    return expr_rhs<NC, T>(e, label<NC>(lab));

}


/** \brief Element-wise multiplication of two expressions

    \ingroup libtensor_expr_operators
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N + M - 1, T> ewmult(
    const letter &l,
    const expr_rhs<N, T> &bta,
    const expr_rhs<M, T> &btb) {

    return ewmult(label<1>(l), bta, btb);
}


} // namespace expr
} // namespace libtensor


namespace libtensor {

using expr::div;
using expr::ewmult;
using expr::mult;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_OPERATORS_EWMULT_H
