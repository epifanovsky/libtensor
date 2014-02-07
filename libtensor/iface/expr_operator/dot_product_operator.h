#ifndef LIBTENSOR_IFACE_DOT_PRODUCT_OPERATOR_H
#define LIBTENSOR_IFACE_DOT_PRODUCT_OPERATOR_H

#include <libtensor/expr/node_dot_product.h>
#include <libtensor/expr/dag/node_scalar.h>
#include <libtensor/iface/expr_rhs.h>
#include <libtensor/iface/eval/eval.h>

namespace libtensor {
namespace iface {


/** \brief Dot product

    \ingroup libtensor_iface
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

    expr::node_assign n1(0);
    expr::expr_tree e(expr::node_assign(0));
    expr::expr_tree::node_id_t id_res =
        e.add(e.get_root(), expr::node_scalar<T>(d));
    expr::expr_tree::node_id_t id_dot =
        e.add(e.get_root(), expr::node_dot_product(idxa, idxb));
    e.add(id_dot, lhs.get_expr());
    e.add(id_dot, rhs.get_expr());

    eval().evaluate(e);

    return d;
}


} // namespace iface

using iface::dot_product;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_DOT_PRODUCT_OPERATOR_H
