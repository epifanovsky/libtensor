#ifndef LIBTENSOR_IFACE_TRACE_OPERATOR_H
#define LIBTENSOR_IFACE_TRACE_OPERATOR_H

#include <libtensor/core/permutation_builder.h>
#include <libtensor/block_tensor/btod_trace.h>
#include "../expr_rhs.h"

namespace libtensor {
namespace iface {


/** \brief Trace of a matrix expression

    \ingroup libtensor_iface
 **/
template<typename T>
double trace(
    const letter &l1,
    const letter &l2,
    const expr_rhs<2, T> &expr) {

    return 0;
//    letter_expr<2> le(l1|l2);
//    anon_eval<2, T> eval(expr, le);
//    eval.evaluate();
//    return btod_trace<1>(eval.get_btensor()).calculate();
}


/** \brief Trace of a tensor expression

    \ingroup libtensor_iface
 **/
template<size_t N, size_t N2, typename T>
double trace(
    const letter_expr<N> le1,
    const letter_expr<N> le2,
    expr_rhs<N2, T> expr) {

    return 0;
//    trace_subexpr_label_builder<N> lb(le1, le2);
//    anon_eval<2 * N, T> eval(expr, lb.get_label());
//    eval.evaluate();
//    return btod_trace<N>(eval.get_btensor()).calculate();
}


} // namespace labeled_btensor_expr

using iface::trace;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_TRACE_OPERATOR_H
