#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATOR_SUB_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATOR_SUB_H

#include "../scale/core_scale.h"
#include "operator_add.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Subtraction of an expression from an expression

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename E1, typename E2>
expr<N, T, core_add<N, T,
	expr< N, T, E1 >,
	expr< N, T, core_scale< N, T, expr<N, T, E2> > >
> >
inline operator-(
	expr<N, T, E1> lhs,
	expr<N, T, E2> rhs) {

	typedef core_scale< N, T, expr<N, T, E2> > core2_t;
	typedef expr<N, T, core2_t> sexpr2_t;
	return lhs + sexpr2_t(core2_t(-1, rhs));
}


/**	\brief Subtraction of a %tensor from an expression

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename E1, bool A2>
expr<N, T, core_add<N, T,
	expr< N, T, E1 >,
	expr< N, T, core_scale< N, T, expr< N, T, core_ident<N, T, A2> > > >
> >
inline operator-(
	expr<N, T, E1> lhs,
	labeled_btensor<N, T, A2> rhs) {

	typedef expr< N, T, core_ident<N, T, A2> > expr2_t;
	return lhs - expr2_t(rhs);
}


/**	\brief Subtraction of an expression from a %tensor

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1, typename E2>
expr<N, T, core_add<N, T,
	expr< N, T, core_ident<N, T, 1> >,
	expr< N, T, core_scale< N, T, expr<N, T, E2> > >
> >
inline operator-(
	labeled_btensor<N, T, A1> lhs,
	expr<N, T, E2> rhs) {

	typedef expr<N, T, core_ident< N, T, A1> > expr1_t;
	return expr1_t(lhs) - rhs;
}


/**	\brief Subtraction of a %tensor from a %tensor

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1, bool A2>
expr<N, T, core_add<N, T,
	expr< N, T, core_ident<N, T, A1> >,
	expr< N, T, core_scale<N, T,
		expr< N, T, core_ident<N, T, A2> >
	> >
> >
inline operator-(
	labeled_btensor<N, T, A1> lhs,
	labeled_btensor<N, T, A2> rhs) {

	typedef expr< N, T, core_ident<N, T, A1> > expr1_t;
	typedef expr< N, T, core_ident<N, T, A2> > expr2_t;
	return expr1_t(lhs) - expr2_t(rhs);
}


/**	\brief Unary minus (expression), effectively multiplies by -1

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename E>
expr<N, T, core_scale<N, T, expr<N, T, E> > >
inline operator-(
	expr<N, T, E> e) {

	typedef expr<N, T, E> opexpr_t;
	typedef core_scale<N, T, opexpr_t> scale_expr_t;
	typedef expr<N, T, scale_expr_t> expr_t;
	return expr_t(scale_expr_t(-1, opexpr_t(e)));
}


/**	\brief Unary minus on a %tensor, effectively multiplies by -1

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr< N, T, core_scale< N, T, expr< N, T, core_ident<N, T, A> > > >
inline operator-(
	labeled_btensor<N, T, A> t) {

	typedef expr< N, T, core_ident<N, T, A> > expr_t;
	return -expr_t(t);
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::operator-;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATOR_SUB_H
