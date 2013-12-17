#ifndef LIBTENSOR_IFACE_EWMULT_OPERATOR_H
#define LIBTENSOR_IFACE_EWMULT_OPERATOR_H

#include <libtensor/expr/node_contract.h>
#include <libtensor/expr/node_div.h>
#include <libtensor/expr/node_transform.h>

namespace libtensor {
namespace iface {


/** \brief Element-wise multiplication of two expressions

    \ingroup libtensor_iface
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

    expr::expr_tree e(expr::node_contract(N, map, false));
    expr::expr_tree::node_id_t id = e.get_root();
    e.add(id, lhs.get_expr());
    e.add(id, rhs.get_expr());

    return expr_rhs<N, T>(e, lhs.get_label());
}


/** \brief Element-wise division of two expressions

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
expr_rhs<N, T> div(
    const expr_rhs<N, T> &lhs,
    const expr_rhs<N, T> &rhs) {

    expr::node_div n1(N);
    expr::expr_tree e(n1);
    expr::expr_tree::node_id_t id = e.get_root();
    e.add(id, lhs.get_expr());

    permutation<N> px = lhs.get_label().permutation_of(rhs.get_label());
    if (! px.is_identity()) {
        std::vector<size_t> perm(N);
        for (size_t i = 0; i < N; i++) perm[i] = px[i];

        e.add(id, expr::node_transform<T>(perm, scalar_transf<T>()));
        id = e.get_edges_out(id).back();
    }

    e.add(id, rhs.get_expr());
    return expr_rhs<N, T>(e, lhs.get_label());
}


/** \brief Element-wise multiplication of two expressions

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, size_t K, typename T>
expr_rhs<N + M - K, T> ewmult(
    const letter_expr<K> &ewidx,
    const expr_rhs<N, T> &lhs,
    const expr_rhs<M, T> &rhs) {

    static const char method[] = "ewmult(const letter_expr<K> &, "
            "const expr_rhs<N, T> &, const expr_rhs<M, T> &)";

    std::multimap<size_t, size_t> map;
    std::vector<const letter *> label;
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

    for(size_t i = 0; i < N; i++) label.push_back(&lhs.letter_at(i));
    for(size_t i = 0; i < M; i++) {
        const letter &l = rhs.letter_at(i);
        if(!ewidx.contains(l)) label.push_back(&l);
    }

    expr::expr_tree e(expr::node_contract(NC, map, false));
    expr::expr_tree::node_id_t id = e.get_root();
    e.add(id, lhs.get_expr());
    e.add(id, rhs.get_expr());

    return expr_rhs<NC, T>(e, letter_expr<NC>(label));

}


/** \brief Element-wise multiplication of two expressions

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N + M - 1, T> ewmult(
    const letter &l,
    const expr_rhs<N, T> &bta,
    const expr_rhs<M, T> &btb) {

    return ewmult(letter_expr<1>(l), bta, btb);
}


} // namespace iface

using iface::div;
using iface::ewmult;
using iface::mult;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_EWMULT_OPERATOR_H
