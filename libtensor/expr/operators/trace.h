#ifndef LIBTENSOR_EXPR_OPERATORS_TRACE_H
#define LIBTENSOR_EXPR_OPERATORS_TRACE_H

#include <libtensor/expr/dag/node_scalar.h>
#include <libtensor/expr/dag/node_trace.h>
#include <libtensor/expr/iface/expr_rhs.h>

namespace libtensor {
namespace expr {


/** \brief Trace of a tensor expression

    \ingroup libtensor_expr_operators
 **/
template<size_t N, size_t N2, typename T>
T trace(
    const label<N> le1,
    const label<N> le2,
    const expr_rhs<N2, T> &rhs) {

    std::vector<size_t> idx(N2), cidx(N);
    for(size_t i = 0; i < N; i++) cidx[i] = i;
    for(size_t i = 0; i < N2; i++) {
        const letter &l = rhs.letter_at(i);
        if(le1.contains(l)) {
            idx[i] = le1.index_of(l);
        } else if(le2.contains(l)) {
            idx[i] = le2.index_of(l);
        } else {
            throw 0;
        }
    }

    T d;

    node_assign n1(0, false);
    expr_tree e(node_assign(0, false));
    expr_tree::node_id_t id_res = e.add(e.get_root(), node_scalar<T>(d));
    expr_tree::node_id_t id_trace = e.add(e.get_root(), node_trace(idx, cidx));
    e.add(id_trace, rhs.get_expr());

    eval().evaluate(e);

    return d;
}


/** \brief Trace of a matrix expression

    \ingroup libtensor_expr_operators
 **/
template<typename T>
T trace(
    const letter &l1,
    const letter &l2,
    const expr_rhs<2, T> &expr) {

    return trace(label<1>(l1), label<1>(l2), expr);
}


} // namespace expr
} // namespace libtensor


namespace libtensor {

using expr::trace;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_OPERATORS_TRACE_H
