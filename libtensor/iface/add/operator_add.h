#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATOR_ADD_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATOR_ADD_H

#include "../ident/core_ident.h"
#include "core_add.h"
#include "eval_add.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Addition of two expressions

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename E1, typename E2>
expr<N, T, core_add<N, T,
	expr<N, T, E1>,
	expr<N, T, E2>
> >
inline operator+(
	expr<N, T, E1> lhs,
	expr<N, T, E2> rhs) {

	typedef expr<N, T, E1> expr1_t;
	typedef expr<N, T, E2> expr2_t;
	typedef core_add<N, T, expr1_t, expr2_t> add_t;
	return expr<N, T, add_t>(add_t(lhs, rhs));
}


/**	\brief Addition of a %tensor and an expression

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1, typename E2>
expr<N, T, core_add<N, T,
	expr< N, T, core_ident<N, T, A1> >,
	expr< N, T, E2 >
> >
inline operator+(
	labeled_btensor<N, T, A1> lhs,
	expr<N, T, E2> rhs) {

	typedef expr< N, T, core_ident<N, T, A1> > expr1_t;
	return expr1_t(lhs) + rhs;
}


/**	\brief Addition of an expression and a %tensor

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename E1, bool A2>
expr<N, T, core_add<N, T,
	expr< N, T, E1 >,
	expr< N, T, core_ident<N, T, A2> >
> >
inline operator+(
	expr<N, T, E1> lhs,
	labeled_btensor<N, T, A2> rhs) {

	typedef expr< N, T, core_ident<N, T, A2> > expr2_t;
	return lhs + expr2_t(rhs);
}


/**	\brief Addition of two tensors

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1, bool A2>
expr<N, T, core_add<N, T,
	expr< N, T, core_ident<N, T, A1> >,
	expr< N, T, core_ident<N, T, A2> >
> >
inline operator+(
	labeled_btensor<N, T, A1> lhs,
	labeled_btensor<N, T, A2> rhs) {

	typedef expr< N, T, core_ident<N, T, A1> > expr1_t;
	typedef expr< N, T, core_ident<N, T, A2> > expr2_t;
	return expr1_t(lhs) + expr2_t(rhs);
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::operator+;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATOR_ADD_H
