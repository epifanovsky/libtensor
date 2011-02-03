#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATOR_DIV_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATOR_DIV_H

#include "../ident/core_ident.h"
#include "core_mult.h"
#include "eval_mult.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Element-wise division of two expressions

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename E1, typename E2>
expr<N, T, core_mult<N, T, expr<N, T, E1>, expr<N, T, E2>, true> >
inline div(expr<N, T, E1> lhs, expr<N, T, E2> rhs) {

	typedef expr<N, T, E1> expr1_t;
	typedef expr<N, T, E2> expr2_t;
	typedef core_mult<N, T, expr1_t, expr2_t, true> core_t;
	return expr<N, T, core_t>(core_t(lhs, rhs));
}


/**	\brief Element-wise division of an expression and a %tensor

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename E1, bool A2>
expr<N, T, core_mult<N, T,
	expr<N, T, E1>,
	expr<N, T, core_ident<N, T, A2> >,
	true
> >
inline div(
	expr<N, T, E1> lhs,
	labeled_btensor<N, T, A2> rhs) {

	typedef expr< N, T, core_ident<N, T, A2> > expr2_t;
	return div(lhs, expr2_t(rhs));
}


/**	\brief Element-wise divison of a %tensor and an expression

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1, typename E2>
expr<N, T, core_mult<N, T,
	expr< N, T, core_ident<N, T, A1> >,
	expr< N, T, E2>,
	true
> >
inline div(
	labeled_btensor<N, T, A1> lhs,
	expr<N, T, E2> rhs) {

	typedef expr<N, T, core_ident< N, T, A1> > expr1_t;
	return div(expr1_t(lhs), rhs);
}


/**	\brief Element-wise division of two %tensors

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1, bool A2>
expr<N, T, core_mult<N, T,
	expr< N, T, core_ident<N, T, A1> >,
	expr< N, T, core_ident<N, T, A2> >,
	true
> >
inline div(
	labeled_btensor<N, T, A1> lhs,
	labeled_btensor<N, T, A2> rhs) {

	typedef expr< N, T, core_ident<N, T, A1> > expr1_t;
	typedef expr< N, T, core_ident<N, T, A2> > expr2_t;
	return div(expr1_t(lhs), expr2_t(rhs));
}



} // namespace labeled_btensor_expr

using labeled_btensor_expr::div;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATOR_DIV_H
