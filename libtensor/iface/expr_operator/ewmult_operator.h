#ifndef LIBTENSOR_IFACE_EWMULT_OPERATOR_H
#define LIBTENSOR_IFACE_EWMULT_OPERATOR_H

#include <libtensor/expr/node_ewmult.h>
#include <libtensor/expr/node_mult.h>
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

    const expr_tree &le = lhs.get_expr(), &re = rhs.get_expr();
    tensor_list tl(le.get_tensors());
    tl.merge(re.get_tensors());

    permutation<N> px = lhs.get_label().permutation_of(rhs.get_label());
    if (px.is_identity()) {
        return expr_rhs<N, T>(expr_tree(expr::node_mult(le.get_nodes(),
                re.get_nodes(), false), tl), lhs.get_label());
    }
    else {
        std::vector<size_t> perm(N);
        for (size_t i = 0; i < N; i++) perm[i] = px[i];

        expr::node_transform<T> ntr(re.get_nodes(), perm, scalar_transf<T>());
        return expr_rhs<N, T>(
            expr_tree(expr::node_mult(le.get_nodes(), ntr, false), tl),
            lhs.get_label());
    }

}


/** \brief Element-wise division of two expressions

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
expr_rhs<N, T> div(
    const expr_rhs<N, T> &lhs,
    const expr_rhs<N, T> &rhs) {

    const expr_tree &le = lhs.get_expr(), &re = rhs.get_expr();
    tensor_list tl(le.get_tensors());
    tl.merge(re.get_tensors());

    permutation<N> px = lhs.get_label().permutation_of(rhs.get_label());
    if (px.is_identity()) {
        return expr_rhs<N, T>(expr_tree(expr::node_mult(le.get_nodes(),
                re.get_nodes(), true), tl), lhs.get_label());
    }
    else {
        std::vector<size_t> perm(N);
        for (size_t i = 0; i < N; i++) perm[i] = px[i];

        expr::node_transform<T> ntr(re.get_nodes(), perm, scalar_transf<T>());
        return expr_rhs<N, T>(
            expr_tree(expr::node_mult(le.get_nodes(), ntr, true), tl),
            lhs.get_label());
    }
}


/** \brief Element-wise multiplication of two expressions

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, size_t K, typename T>
expr_rhs<N + M - K, T> ewmult(
    const letter_expr<K> &ewidx,
    const expr_rhs<N, T> &lhs,
    const expr_rhs<M, T> &rhs) {

    static const char *method = "ewmult(const letter_expr<K> &, "
            "const expr_rhs<N, T> &, const expr_rhs<M, T> &)";

    const expr_tree &le = lhs.get_expr(), &re = rhs.get_expr();
    tensor_list tl(le.get_tensors());
    tl.merge(re.get_tensors());

    std::map<size_t, size_t> multmap;
    std::vector<const letter *> label;
    for (size_t i = 0; i < K; i++) {
        const letter &l = ewidx.letter_at(i);
        if(!lhs.contains(l) || !rhs.contains(l)) {
            throw expr_exception(g_ns, "", method, __FILE__, __LINE__,
                    "Letter not found.");
        }
        multmap[lhs.index_of(l)] = rhs.index_of(l);
    }

    enum {
        NC = N + M - K
    };

    expr::node_ewmult nmul(le.get_nodes(), re.get_nodes(), multmap);
    return expr_rhs<NC, T>(expr_tree(nmul, tl), letter_expr<NC>(label));
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
