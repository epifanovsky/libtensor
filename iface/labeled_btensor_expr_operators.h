#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATORS_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATORS_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor_expr.h"
#include "labeled_btensor_expr_add.h"
#include "labeled_btensor_expr_ident.h"
#include "labeled_btensor_expr_scale.h"

/**	\defgroup libtensor_btensor_expr_op Overloaded operators
	\ingroup libtensor_btensor_expr
 **/

namespace libtensor {

namespace labeled_btensor_expr {

/**	\brief Unary minus (plain), effectively multiplies by -1

	Creates expr< scale< -1.0, expr< ident > > >

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool Assignable, typename Label>
expr<N, T, core_scale<N, T, expr<N, T, core_ident<N, T, Assignable, Label> > > >
inline operator-(
	labeled_btensor<N, T, Assignable, Label> t) {

	typedef core_ident<N, T, Assignable, Label> id_t;
	typedef expr<N, T, id_t> expr_id_t;
	typedef core_scale<N, T, expr_id_t> scale_expr_t;
	typedef expr<N, T, scale_expr_t> expr_t;
	return expr_t(scale_expr_t(-1, expr_id_t(id_t(t))));
}

/**	\brief Unary minus (expression), effectively multiplies by -1

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename Expr>
expr<N, T, core_scale<N, T, expr<N, T, Expr> > >
inline operator-(
	expr<N, T, Expr> t) {

	typedef expr<N, T, Expr> opexpr_t;
	typedef core_scale<N, T, opexpr_t> scale_expr_t;
	typedef expr<N, T, scale_expr_t> expr_t;
	return expr_t(scale_expr_t(-1, opexpr_t(t)));
}

/**	\brief Addition of two tensors (plain + plain)

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool AssignableL, typename LabelL,
	bool AssignableR, typename LabelR>
expr<N, T, core_add<N, T,
	expr<N, T, core_ident<N, T, AssignableL, LabelL> >,
	expr<N, T, core_ident<N, T, AssignableR, LabelR> >
> >
inline operator+(
	labeled_btensor<N, T, AssignableL, LabelL> lhs,
	labeled_btensor<N, T, AssignableR, LabelR> rhs) {

	typedef expr<N, T, core_ident<N, T, AssignableL, LabelL> > exprl_t;
	typedef expr<N, T, core_ident<N, T, AssignableR, LabelR> > exprr_t;
	typedef core_add<N, T, exprl_t, exprr_t> add_t;
	return expr<N, T, add_t>(add_t(exprl_t(lhs), exprr_t(rhs)));
}

/**	\brief Subtraction of tensors (plain - plain)

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool AssignableL, typename LabelL,
	bool AssignableR, typename LabelR>
expr<N, T, core_add<N, T,
	expr<N, T, core_ident<N, T, AssignableL, LabelL> >,
	expr<N, T, core_scale<N, T,
		expr<N, T, core_ident<N, T, AssignableR, LabelR> >
	> >
> >
inline operator-(
	labeled_btensor<N, T, AssignableL, LabelL> lhs,
	labeled_btensor<N, T, AssignableR, LabelR> rhs) {

	typedef expr<N, T, core_ident<N, T, AssignableL, LabelL> > exprl_t;
	typedef expr<N, T, core_ident<N, T, AssignableR, LabelR> > exprr_t;
	typedef core_scale<N, T, exprr_t> scale_exprr_t;
	typedef expr<N, T, scale_exprr_t> scaled_exprr_t;
	typedef core_add<N, T, exprl_t, scaled_exprr_t> add_t;
	return expr<N, T, add_t>(add_t(exprl_t(lhs),
		scaled_exprr_t(scale_exprr_t(-1, exprr_t(rhs)))));
}

/**	\brief Addition of two tensors (plain + expression)

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool AssignableL, typename LabelL,
	typename ExprR>
expr<N, T, core_add<N, T,
	expr<N, T, core_ident<N, T, AssignableL, LabelL> >,
	expr<N, T, ExprR>
> >
inline operator+(
	labeled_btensor<N, T, AssignableL, LabelL> lhs,
	expr<N, T, ExprR> rhs) {

	typedef expr<N, T, core_ident<N, T, AssignableL, LabelL> > exprl_t;
	typedef expr<N, T, ExprR> exprr_t;
	typedef core_add<N, T, exprl_t, exprr_t> add_t;
	return expr<N, T, add_t>(add_t(exprl_t(lhs), rhs));
}

/**	\brief Subtraction of tensors (plain - expression)

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool AssignableL, typename LabelL,
	typename ExprR>
expr<N, T, core_add<N, T,
	expr<N, T, core_ident<N, T, AssignableL, LabelL> >,
	expr<N, T, core_scale<N, T, expr<N, T, ExprR> > >
> >
inline operator-(
	labeled_btensor<N, T, AssignableL, LabelL> lhs,
	expr<N, T, ExprR> rhs) {

	typedef expr<N, T, core_ident<N, T, AssignableL, LabelL> > exprl_t;
	typedef expr<N, T, ExprR> exprr_t;
	typedef core_scale<N, T, exprr_t> scale_exprr_t;
	typedef expr<N, T, scale_exprr_t> scaled_exprr_t;
	typedef core_add<N, T, exprl_t, scaled_exprr_t> add_t;
	return expr<N, T, add_t>(add_t(exprl_t(lhs),
		scaled_exprr_t(scale_exprr_t(-1, rhs))));
}

/**	\brief Addition of two tensors (expression + plain)

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename ExprL, bool AssignableR,
	typename LabelR>
expr<N, T, core_add<N, T,
	expr<N, T, ExprL>,
	expr<N, T, core_ident<N, T, AssignableR, LabelR> >
> >
inline operator+(
	expr<N, T, ExprL> lhs,
	labeled_btensor<N, T, AssignableR, LabelR> rhs) {

	typedef expr<N, T, ExprL> exprl_t;
	typedef expr<N, T, core_ident<N, T, AssignableR, LabelR> > exprr_t;
	typedef core_add<N, T, exprl_t, exprr_t> add_t;
	return expr<N, T, add_t>(add_t(lhs, exprr_t(rhs)));
}

/**	\brief Subtraction of tensors (expression - plain)

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename ExprL, bool AssignableR,
	typename LabelR>
expr<N, T, core_add<N, T,
	expr<N, T, ExprL>,
	expr<N, T, core_scale<N, T,
		expr<N, T, core_ident<N, T, AssignableR, LabelR> >
	> >
> >
inline operator-(
	expr<N, T, ExprL> lhs,
	labeled_btensor<N, T, AssignableR, LabelR> rhs) {

	typedef expr<N, T, ExprL> exprl_t;
	typedef expr<N, T, core_ident<N, T, AssignableR, LabelR> > exprr_t;
	typedef core_scale<N, T, exprr_t> scale_exprr_t;
	typedef expr<N, T, scale_exprr_t> scaled_exprr_t;
	typedef core_add<N, T, exprl_t, scaled_exprr_t> add_t;
	return expr<N, T, add_t>(add_t(lhs,
		scaled_exprr_t(scale_exprr_t(-1, exprr_t(rhs)))));
}

/**	\brief Addition of two tensors (expression + expression)

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename ExprL, typename ExprR>
expr<N, T, core_add<N, T,
	expr<N, T, ExprL>,
	expr<N, T, ExprR>
> >
inline operator+(
	expr<N, T, ExprL> lhs,
	expr<N, T, ExprR> rhs) {

	typedef expr<N, T, ExprL> exprl_t;
	typedef expr<N, T, ExprR> exprr_t;
	typedef core_add<N, T, exprl_t, exprr_t> add_t;
	return expr<N, T, add_t>(add_t(lhs, rhs));
}

/**	\brief Subtraction of tensors (expression - expression)

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename ExprL, typename ExprR>
expr<N, T, core_add<N, T,
	expr<N, T, ExprL>,
	expr<N, T, core_scale<N, T, expr<N, T, ExprR> > >
> >
inline operator-(
	expr<N, T, ExprL> lhs,
	expr<N, T, ExprR> rhs) {

	typedef expr<N, T, ExprL> exprl_t;
	typedef expr<N, T, ExprR> exprr_t;
	typedef core_scale<N, T, exprr_t> scale_exprr_t;
	typedef expr<N, T, scale_exprr_t> scaled_exprr_t;
	typedef core_add<N, T, exprl_t, scaled_exprr_t> add_t;
	return expr<N, T, add_t>(add_t(lhs,
		scaled_exprr_t(scale_exprr_t(-1, rhs))));
}

/**	\brief Multiplication of a tensor (rhs) by a scalar (lhs)

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool Assignable, typename Label>
expr<N, T, core_scale<N, T, expr<N, T, core_ident<N, T, Assignable, Label> > > >
inline operator*(
	T lhs,
	labeled_btensor<N, T, Assignable, Label> rhs) {

	typedef core_ident<N, T, Assignable, Label> rhs_expr_id_t;
	typedef expr<N, T, rhs_expr_id_t> rhs_expr_t;
	typedef core_scale<N, T, rhs_expr_t> scale_expr_t;
	typedef expr<N, T, scale_expr_t> expr_t;
	return expr_t(scale_expr_t(lhs, rhs_expr_t(rhs_expr_id_t(rhs))));
}

/**	\brief Multiplication of a tensor (lhs) by a scalar (rhs)

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool Assignable, typename Label>
expr<N, T, core_scale<N, T, expr<N, T, core_ident<N, T, Assignable, Label> > > >
inline operator*(
	labeled_btensor<N, T, Assignable, Label> lhs,
	T rhs) {

	typedef core_ident<N, T, Assignable, Label> lhs_expr_id_t;
	typedef expr<N, T, lhs_expr_id_t> lhs_expr_t;
	typedef core_scale<N, T, lhs_expr_t> scale_expr_t;
	typedef expr<N, T, scale_expr_t> expr_t;
	return expr_t(scale_expr_t(rhs, lhs_expr_t(lhs_expr_id_t(lhs))));
}

/**	\brief Multiplication of an expression (rhs) by a scalar (lhs)

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename ExprR>
expr<N, T, core_scale<N, T, expr<N, T, ExprR> > >
inline operator*(
	T lhs,
	expr<N, T, ExprR> rhs) {

	typedef expr<N, T, ExprR> rhs_expr_t;
	typedef core_scale<N, T, rhs_expr_t> scale_expr_t;
	typedef expr<N, T, scale_expr_t> expr_t;
	return expr_t(scale_expr_t(lhs, rhs));
}

/**	\brief Multiplication of an expression (lhs) by a scalar (rhs)

	\ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename ExprL>
expr<N, T, core_scale<N, T, expr<N, T, ExprL> > >
inline operator*(
	expr<N, T, ExprL> lhs,
	T rhs) {

	typedef expr<N, T, ExprL> lhs_expr_t;
	typedef core_scale<N, T, lhs_expr_t> scale_expr_t;
	typedef expr<N, T, scale_expr_t> expr_t;
	return expr_t(scale_expr_t(rhs, lhs));
}

} // namespace labeled_btensor_expr

using labeled_btensor_expr::operator+;
using labeled_btensor_expr::operator-;
using labeled_btensor_expr::operator*;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATORS_H
