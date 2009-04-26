#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_H
#define	LIBTENSOR_LABELED_BTENSOR_EXPR_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor.h"

/**	\defgroup libtensor_btensor_expr Labeled block %tensor expressions
	\ingroup libtensor
 **/

namespace libtensor {

/**	\brief Expression using labeled block tensors
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Expr Expression.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Expr>
class labeled_btensor_expr {
public:
	typedef Expr expression_t;

private:
	Expr m_t;

public:

	labeled_btensor_expr(const Expr &t) : m_t(t) {
	}

	labeled_btensor_expr(const labeled_btensor_expr<N, T, Expr> &e) :
		m_t(e.m_t) {
	}
};

/**	\brief Identity expression

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Traits, typename Label>
class labeled_btensor_expr_ident {
public:
	typedef labeled_btensor<N, T, Traits, Label> labeled_btensor_t;

private:
	labeled_btensor_t &m_t;

public:

	labeled_btensor_expr_ident(labeled_btensor_t &t) : m_t(t) {
	}
};

/**	\brief Operation expression
	\tparam NArg Number of arguments
	\tparam Op Operation
	\tparam ExprL LHS expression
	\tparam ExprR RHS expression

	\ingroup libtensor_btensor_expr
 **/
template<size_t NArg, typename Op, typename ExprL, typename ExprR>
class labeled_btensor_expr_op {
private:
	ExprL m_exprl; //!< Left expression
	ExprR m_exprr; //!< Right expression

public:

	labeled_btensor_expr_op(const ExprL &exprl, const ExprR &exprr) :
		m_exprl(exprl), m_exprr(exprr) {
	}
};

/**	\brief Addition operation
	\tparam NArg Total number of arguments
	\tparam ExprL LHS expression
	\tparam ExprR RHS expression

	\ingroup libtensor_btensor_expr
 **/
template<size_t NArg, typename ExprL, typename ExprR>
class labeled_btensor_expr_op_add {
};

/**	\brief Addition of two tensors

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename TraitsL, typename LabelL,
typename TraitsR, typename LabelR>
labeled_btensor_expr<N, T,
labeled_btensor_expr_op < 2, labeled_btensor_expr_op_add < 2,
labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsL, LabelL> >,
labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > >,
labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsL, LabelL> >,
labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > >
>
operator+(labeled_btensor<N, T, TraitsL, LabelL> lhs,
	labeled_btensor<N, T, TraitsR, LabelR> rhs) {
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, TraitsL, LabelL> > exprl_t;
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > exprr_t;
	typedef labeled_btensor_expr_op_add < 2, exprl_t, exprr_t> opadd_t;
	typedef labeled_btensor_expr_op < 2, opadd_t, exprl_t, exprr_t> op_t;
	return labeled_btensor_expr<N, T, op_t > (
		op_t(exprl_t(lhs), exprr_t(rhs)));
}

/**	\brief Addition of a sum (lhs) and a tensor (rhs)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, size_t NArgL, typename Arg1L, typename Arg2L,
typename TraitsR, typename LabelR>
labeled_btensor_expr<N, T, labeled_btensor_expr_op<NArgL + 1,
labeled_btensor_expr_op_add < NArgL + 1,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< NArgL,
labeled_btensor_expr_op_add< NArgL, Arg1L, Arg2L >, Arg1L, Arg2L > >,

labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > >,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< NArgL,
labeled_btensor_expr_op_add< NArgL, Arg1L, Arg2L >, Arg1L, Arg2L > >,

labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > >
>
operator+(labeled_btensor_expr<N, T, labeled_btensor_expr_op< NArgL,
	labeled_btensor_expr_op_add<NArgL, Arg1L, Arg2L >, Arg1L, Arg2L > > lhs,
	labeled_btensor<N, T, TraitsR, LabelR> rhs) {
	typedef labeled_btensor_expr< N, T,
		labeled_btensor_expr_op< NArgL,
		labeled_btensor_expr_op_add< NArgL, Arg1L, Arg2L >,
		Arg1L, Arg2L > > exprl_t;
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > exprr_t;
	typedef labeled_btensor_expr_op_add < NArgL + 1,
		exprl_t, exprr_t> opadd_t;
	typedef labeled_btensor_expr_op < NArgL + 1, opadd_t,
		exprl_t, exprr_t> op_t;
	return labeled_btensor_expr<N, T, op_t > (
		op_t(exprl_t(lhs), exprr_t(rhs)));
}

/**	\brief Addition of a tensor (lhs) and a sum (rhs)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename TraitsL, typename LabelL,
size_t NArgR, typename Arg1R, typename Arg2R>
labeled_btensor_expr<N, T, labeled_btensor_expr_op<NArgR + 1,
labeled_btensor_expr_op_add < NArgR + 1,

labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsL, LabelL> >,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< NArgR,
labeled_btensor_expr_op_add< NArgR, Arg1R, Arg2R >, Arg1R, Arg2R > > >,

labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsL, LabelL> >,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< NArgR,
labeled_btensor_expr_op_add< NArgR, Arg1R, Arg2R >, Arg1R, Arg2R > >

> >
operator+(labeled_btensor<N, T, TraitsL, LabelL> lhs,
	labeled_btensor_expr<N, T, labeled_btensor_expr_op< NArgR,
	labeled_btensor_expr_op_add<NArgR, Arg1R, Arg2R >,
	Arg1R, Arg2R > > rhs) {

	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, TraitsL, LabelL> > exprl_t;
	typedef labeled_btensor_expr< N, T,
		labeled_btensor_expr_op< NArgR,
		labeled_btensor_expr_op_add< NArgR, Arg1R, Arg2R >,
		Arg1R, Arg2R > > exprr_t;
	typedef labeled_btensor_expr_op_add < NArgR + 1,
		exprl_t, exprr_t> opadd_t;
	typedef labeled_btensor_expr_op < NArgR + 1, opadd_t,
		exprl_t, exprr_t> op_t;
	return labeled_btensor_expr<N, T, op_t > (
		op_t(exprl_t(lhs), exprr_t(rhs)));
}

/**	\brief Addition of two sums

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, size_t NArgL, typename Arg1L, typename Arg2L,
size_t NArgR, typename Arg1R, typename Arg2R>
labeled_btensor_expr<N, T, labeled_btensor_expr_op<NArgL + NArgR,
labeled_btensor_expr_op_add < NArgL + NArgR,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< NArgL,
labeled_btensor_expr_op_add< NArgL, Arg1L, Arg2L >, Arg1L, Arg2L > >,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< NArgR,
labeled_btensor_expr_op_add< NArgR, Arg1R, Arg2R >, Arg1R, Arg2R > > >,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< NArgL,
labeled_btensor_expr_op_add< NArgL, Arg1L, Arg2L >, Arg1L, Arg2L > >,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< NArgR,
labeled_btensor_expr_op_add< NArgR, Arg1R, Arg2R >, Arg1R, Arg2R > >

> >
operator+(labeled_btensor_expr<N, T, labeled_btensor_expr_op< NArgL,
	labeled_btensor_expr_op_add<NArgL, Arg1L, Arg2L >, Arg1L, Arg2L > > lhs,
	labeled_btensor_expr<N, T, labeled_btensor_expr_op< NArgR,
	labeled_btensor_expr_op_add<NArgR, Arg1R, Arg2R >,
	Arg1R, Arg2R > > rhs) {

	typedef labeled_btensor_expr< N, T,
		labeled_btensor_expr_op< NArgL,
		labeled_btensor_expr_op_add< NArgL, Arg1L, Arg2L >,
		Arg1L, Arg2L > > exprl_t;
	typedef labeled_btensor_expr< N, T,
		labeled_btensor_expr_op< NArgR,
		labeled_btensor_expr_op_add< NArgR, Arg1R, Arg2R >,
		Arg1R, Arg2R > > exprr_t;
	typedef labeled_btensor_expr_op_add < NArgL + NArgR,
		exprl_t, exprr_t> opadd_t;
	typedef labeled_btensor_expr_op < NArgL + NArgR, opadd_t,
		exprl_t, exprr_t> op_t;
	return labeled_btensor_expr<N, T, op_t > (
		op_t(exprl_t(lhs), exprr_t(rhs)));
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_H

