#ifndef LIBTENSOR_EXPR_OPERATORS_DOT_PRODUCT_H
#define LIBTENSOR_EXPR_OPERATORS_DOT_PRODUCT_H

#include <libtensor/expr/dag/node_dot_product.h>
#include <libtensor/expr/dag/node_scalar.h>
#include <libtensor/expr/iface/expr_rhs.h>
#include <libtensor/expr/eval/eval.h>

namespace libtensor {
namespace expr {


/** \brief Dot product

    \ingroup libtensor_expr_operators
 **/
template<size_t N, typename T>
T dot_product(
    const expr_rhs<N, T> &lhs,
    const expr_rhs<N, T> &rhs) {

    std::vector<size_t> idxa(N), idxb(N);
    for(size_t i = 0; i < N; i++) {
        const letter &l = lhs.letter_at(i);
        idxa[i] = i;
        idxb[rhs.index_of(l)] = i;
    }

    T d;

    node_assign n1(0);
    expr_tree e(node_assign(0));
    expr_tree::node_id_t id_res = e.add(e.get_root(), node_scalar<T>(d));
    expr_tree::node_id_t id_dot =
        e.add(e.get_root(), node_dot_product(idxa, idxb));
    e.add(id_dot, lhs.get_expr());
    e.add(id_dot, rhs.get_expr());

    eval().evaluate(e);

    return d;
}


} // namespace expr
} // namespace libtensor


namespace libtensor {

using expr::dot_product;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_OPERATORS_DOT_PRODUCT_H
