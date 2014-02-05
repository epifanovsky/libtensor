#ifndef LIBTENSOR_IFACE_TRACE_OPERATOR_H
#define LIBTENSOR_IFACE_TRACE_OPERATOR_H

#include <libtensor/expr/node_scalar.h>
#include <libtensor/expr/node_trace.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/block_tensor/btod_trace.h>
#include "../expr_rhs.h"

namespace libtensor {
namespace iface {


/** \brief Trace of a tensor expression

    \ingroup libtensor_iface
 **/
template<size_t N, size_t N2, typename T>
T trace(
    const letter_expr<N> le1,
    const letter_expr<N> le2,
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

    expr::node_assign n1(0);
    expr::expr_tree e(expr::node_assign(0));
    expr::expr_tree::node_id_t id_res =
        e.add(e.get_root(), expr::node_scalar<T>(d));
    expr::expr_tree::node_id_t id_trace =
        e.add(e.get_root(), expr::node_trace(idx, cidx));
    e.add(id_trace, rhs.get_expr());

    eval().evaluate(e);

    return d;
}


/** \brief Trace of a matrix expression

    \ingroup libtensor_iface
 **/
template<typename T>
T trace(
    const letter &l1,
    const letter &l2,
    const expr_rhs<2, T> &expr) {

    return trace(letter_expr<1>(l1), letter_expr<1>(l2), expr);
}


} // namespace iface

using iface::trace;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_TRACE_OPERATOR_H
