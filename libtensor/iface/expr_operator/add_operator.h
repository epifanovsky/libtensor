#ifndef LIBTENSOR_IFACE_ADD_OPERATOR_H
#define LIBTENSOR_IFACE_ADD_OPERATOR_H

#include <libtensor/core/scalar_transf.h>
#include <libtensor/expr/node_add.h>
#include <libtensor/expr/node_transform.h>
#include "mul_operator.h"

namespace libtensor {
namespace iface {


/** \brief Addition of two expressions

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator+(
    const expr_rhs<N, T> &lhs,
    const expr_rhs<N, T> &rhs) {

    const expr_tree &le = lhs.get_expr(), &re = rhs.get_expr();

    tensor_list tl(le.get_tensors());
    tl.merge(re.get_tensors());

    // TODO: remap tensors in rhs

    permutation<N> px = match(lhs.get_label(), rhs.get_label());
    if(px.is_identity()) {

        return expr_rhs<N, T>(
            expr_tree(expr::node_add(le.get_nodes(), re.get_nodes()), tl),
            lhs.get_label());

    } else {

        std::vector<size_t> perm(N);
        for(size_t i = 0; i < N; i++) perm[i] = px[i];

        expr::node_transform<T> ntr(re.get_nodes(), perm, scalar_transf<T>());
        return expr_rhs<N, T>(
           expr_tree(expr::node_add(le.get_nodes(), ntr), tl),
           lhs.get_label());
    }
}


/** \brief Subtraction of an expression from an expression

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator-(
    const expr_rhs<N, T> &lhs,
    const expr_rhs<N, T> &rhs) {

    const expr_tree &le = lhs.get_expr(), &re = rhs.get_expr();

    tensor_list tl(le.get_tensors());
    tl.merge(re.get_tensors());

    // TODO: remap tensors in rhs

    permutation<N> px = match(lhs.get_label(), rhs.get_label());
    std::vector<size_t> perm(N);
    for(size_t i = 0; i < N; i++) perm[i] = px[i];

    expr::node_transform<T> ntr(re.get_nodes(), perm, scalar_transf<T>(-1));
    return expr_rhs<N, T>(
        expr_tree(expr::node_add(le.get_nodes(), ntr), tl),
        lhs.get_label());
}


/** \brief Do nothing to an expression

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
const expr_rhs<N, T> &operator+(
    const expr_rhs<N, T> &rhs) {

    return rhs;
}


/** \brief Change of the sign of an expression

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator-(
    const expr_rhs<N, T> &rhs) {

    return T(-1) * rhs;
}


} // namespace iface

using iface::operator+;
using iface::operator-;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_ADD_OPERATOR_H
