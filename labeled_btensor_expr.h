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

	void multiply(T coeff) {
		m_t.multiply(coeff);
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
	T m_coeff;

public:

	labeled_btensor_expr_ident(labeled_btensor_t &t) : m_t(t) {
	}
};

/**	\brief Identity expression (specialized for double)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename Traits, typename Label>
class labeled_btensor_expr_ident<N, double, Traits, Label> {
public:
	typedef labeled_btensor<N, double, Traits, Label> labeled_btensor_t;

private:
	labeled_btensor_t &m_t;
	double m_coeff;

public:

	labeled_btensor_expr_ident(labeled_btensor_t &t, double coeff = 1.0) :
		m_t(t), m_coeff(coeff) {
	}

	void multiply(double coeff) {
		m_coeff *= coeff;
	}
};

/**	\brief Operation expression
	\tparam N Tensor order
	\tparam T Tensor element type
	\tparam NArg Number of arguments
	\tparam Op Operation
	\tparam ExprL LHS expression
	\tparam ExprR RHS expression

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, size_t NArg, typename Op,
typename ExprL, typename ExprR>
class labeled_btensor_expr_op {
private:
	ExprL m_exprl; //!< Left expression
	ExprR m_exprr; //!< Right expression

public:

	labeled_btensor_expr_op(const ExprL &exprl, const ExprR &exprr) :
		m_exprl(exprl), m_exprr(exprr) {
	}
};

/**	\brief Operation expression (specialized for double)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t NArg, typename Op, typename ExprL, typename ExprR>
class labeled_btensor_expr_op<N, double, NArg, Op, ExprL, ExprR> {
private:
	ExprL m_exprl; //!< Left expression
	ExprR m_exprr; //!< Right expression
	double m_coeff; //!< Coefficient

public:

	labeled_btensor_expr_op(const ExprL &exprl, const ExprR &exprr,
		double coeff = 1.0) : m_exprl(exprl), m_exprr(exprr),
		m_coeff(coeff) {
	}

	void multiply(double coeff) {
		m_coeff *= coeff;
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

/**	\brief Addition of two tensors (plain + plain)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename TraitsL, typename LabelL,
typename TraitsR, typename LabelR>
labeled_btensor_expr<N, T,
labeled_btensor_expr_op < N, T, 2, labeled_btensor_expr_op_add < 2,
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
	typedef labeled_btensor_expr_op < N, T, 2, opadd_t, exprl_t, exprr_t>
		op_t;
	return labeled_btensor_expr<N, T, op_t > (
		op_t(exprl_t(lhs), exprr_t(rhs)));
}

/**	\brief Addition of two tensors (plain + identity)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename TraitsL, typename LabelL,
typename TraitsR, typename LabelR>
labeled_btensor_expr<N, T,
labeled_btensor_expr_op < N, T, 2, labeled_btensor_expr_op_add < 2,
labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsL, LabelL> >,
labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > >,
labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsL, LabelL> >,
labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > >
>
operator+(labeled_btensor<N, T, TraitsL, LabelL> lhs,
	labeled_btensor_expr<N, T,
	labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > rhs) {
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, TraitsL, LabelL> > exprl_t;
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > exprr_t;
	typedef labeled_btensor_expr_op_add < 2, exprl_t, exprr_t> opadd_t;
	typedef labeled_btensor_expr_op < N, T, 2, opadd_t, exprl_t, exprr_t>
		op_t;
	return labeled_btensor_expr<N, T, op_t > (
		op_t(exprl_t(lhs), rhs));
}

/**	\brief Addition of two tensors (identity + plain)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename TraitsL, typename LabelL,
typename TraitsR, typename LabelR>
labeled_btensor_expr<N, T,
labeled_btensor_expr_op < N, T, 2, labeled_btensor_expr_op_add < 2,
labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsL, LabelL> >,
labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > >,
labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsL, LabelL> >,
labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > >
>
operator+(labeled_btensor_expr<N, T,
	labeled_btensor_expr_ident<N, T, TraitsL, LabelL> > lhs,
	labeled_btensor<N, T, TraitsR, LabelR> rhs) {
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, TraitsL, LabelL> > exprl_t;
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > exprr_t;
	typedef labeled_btensor_expr_op_add < 2, exprl_t, exprr_t> opadd_t;
	typedef labeled_btensor_expr_op < N, T, 2, opadd_t, exprl_t, exprr_t>
		op_t;
	return labeled_btensor_expr<N, T, op_t > (
		op_t(lhs, exprr_t(rhs)));
}

/**	\brief Addition of two tensors (identity + identity)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename TraitsL, typename LabelL,
typename TraitsR, typename LabelR>
labeled_btensor_expr<N, T,
labeled_btensor_expr_op < N, T, 2, labeled_btensor_expr_op_add < 2,
labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsL, LabelL> >,
labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > >,
labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsL, LabelL> >,
labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > >
>
operator+(labeled_btensor_expr<N, T,
	labeled_btensor_expr_ident<N, T, TraitsL, LabelL> > lhs,
	labeled_btensor_expr<N, T,
	labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > rhs) {
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, TraitsL, LabelL> > exprl_t;
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > exprr_t;
	typedef labeled_btensor_expr_op_add < 2, exprl_t, exprr_t> opadd_t;
	typedef labeled_btensor_expr_op < N, T, 2, opadd_t, exprl_t, exprr_t>
		op_t;
	return labeled_btensor_expr<N, T, op_t > (
		op_t(lhs, rhs));
}

/**	\brief Addition of a sum (lhs) and a tensor (rhs, plain)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, size_t NArgL, typename Arg1L, typename Arg2L,
typename TraitsR, typename LabelR>
labeled_btensor_expr<N, T, labeled_btensor_expr_op<N, T, NArgL + 1,
labeled_btensor_expr_op_add < NArgL + 1,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< N, T, NArgL,
labeled_btensor_expr_op_add< NArgL, Arg1L, Arg2L >, Arg1L, Arg2L > >,

labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > >,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< N, T, NArgL,
labeled_btensor_expr_op_add< NArgL, Arg1L, Arg2L >, Arg1L, Arg2L > >,

labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > >
>
operator+(labeled_btensor_expr<N, T, labeled_btensor_expr_op< N, T, NArgL,
	labeled_btensor_expr_op_add<NArgL, Arg1L, Arg2L >, Arg1L, Arg2L > > lhs,
	labeled_btensor<N, T, TraitsR, LabelR> rhs) {
	typedef labeled_btensor_expr< N, T,
		labeled_btensor_expr_op< N, T, NArgL,
		labeled_btensor_expr_op_add< NArgL, Arg1L, Arg2L >,
		Arg1L, Arg2L > > exprl_t;
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > exprr_t;
	typedef labeled_btensor_expr_op_add < NArgL + 1,
		exprl_t, exprr_t> opadd_t;
	typedef labeled_btensor_expr_op < N, T, NArgL + 1, opadd_t,
		exprl_t, exprr_t> op_t;
	return labeled_btensor_expr<N, T, op_t > (
		op_t(exprl_t(lhs), exprr_t(rhs)));
}

/**	\brief Addition of a sum (lhs) and a tensor (rhs, identity)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, size_t NArgL, typename Arg1L, typename Arg2L,
typename TraitsR, typename LabelR>
labeled_btensor_expr<N, T, labeled_btensor_expr_op<N, T, NArgL + 1,
labeled_btensor_expr_op_add < NArgL + 1,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< N, T, NArgL,
labeled_btensor_expr_op_add< NArgL, Arg1L, Arg2L >, Arg1L, Arg2L > >,

labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > >,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< N, T, NArgL,
labeled_btensor_expr_op_add< NArgL, Arg1L, Arg2L >, Arg1L, Arg2L > >,

labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > >
>
operator+(labeled_btensor_expr<N, T, labeled_btensor_expr_op< N, T, NArgL,
	labeled_btensor_expr_op_add<NArgL, Arg1L, Arg2L >, Arg1L, Arg2L > > lhs,
	labeled_btensor_expr<N, T,
	labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > rhs) {
	typedef labeled_btensor_expr< N, T,
		labeled_btensor_expr_op< N, T, NArgL,
		labeled_btensor_expr_op_add< NArgL, Arg1L, Arg2L >,
		Arg1L, Arg2L > > exprl_t;
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, TraitsR, LabelR> > exprr_t;
	typedef labeled_btensor_expr_op_add < NArgL + 1,
		exprl_t, exprr_t> opadd_t;
	typedef labeled_btensor_expr_op < N, T, NArgL + 1, opadd_t,
		exprl_t, exprr_t> op_t;
	return labeled_btensor_expr<N, T, op_t > (
		op_t(exprl_t(lhs), rhs));
}

/**	\brief Addition of a tensor (lhs, plain) and a sum (rhs)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename TraitsL, typename LabelL,
size_t NArgR, typename Arg1R, typename Arg2R>
labeled_btensor_expr<N, T, labeled_btensor_expr_op<N, T, NArgR + 1,
labeled_btensor_expr_op_add < NArgR + 1,

labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsL, LabelL> >,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< N, T, NArgR,
labeled_btensor_expr_op_add< NArgR, Arg1R, Arg2R >, Arg1R, Arg2R > > >,

labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsL, LabelL> >,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< N, T, NArgR,
labeled_btensor_expr_op_add< NArgR, Arg1R, Arg2R >, Arg1R, Arg2R > >

> >
operator+(labeled_btensor<N, T, TraitsL, LabelL> lhs,
	labeled_btensor_expr<N, T, labeled_btensor_expr_op< N, T, NArgR,
	labeled_btensor_expr_op_add<NArgR, Arg1R, Arg2R >,
	Arg1R, Arg2R > > rhs) {

	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, TraitsL, LabelL> > exprl_t;
	typedef labeled_btensor_expr< N, T,
		labeled_btensor_expr_op< N, T, NArgR,
		labeled_btensor_expr_op_add< NArgR, Arg1R, Arg2R >,
		Arg1R, Arg2R > > exprr_t;
	typedef labeled_btensor_expr_op_add < NArgR + 1,
		exprl_t, exprr_t> opadd_t;
	typedef labeled_btensor_expr_op < N, T, NArgR + 1, opadd_t,
		exprl_t, exprr_t> op_t;
	return labeled_btensor_expr<N, T, op_t > (
		op_t(exprl_t(lhs), exprr_t(rhs)));
}

/**	\brief Addition of a tensor (lhs, identity) and a sum (rhs)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename TraitsL, typename LabelL,
size_t NArgR, typename Arg1R, typename Arg2R>
labeled_btensor_expr<N, T, labeled_btensor_expr_op<N, T, NArgR + 1,
labeled_btensor_expr_op_add < NArgR + 1,

labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsL, LabelL> >,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< N, T, NArgR,
labeled_btensor_expr_op_add< NArgR, Arg1R, Arg2R >, Arg1R, Arg2R > > >,

labeled_btensor_expr<N, T, labeled_btensor_expr_ident<N, T, TraitsL, LabelL> >,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< N, T, NArgR,
labeled_btensor_expr_op_add< NArgR, Arg1R, Arg2R >, Arg1R, Arg2R > >

> >
operator+(labeled_btensor_expr<N, T,
	labeled_btensor_expr_ident<N, T, TraitsL, LabelL> > lhs,
	labeled_btensor_expr<N, T, labeled_btensor_expr_op< N, T, NArgR,
	labeled_btensor_expr_op_add<NArgR, Arg1R, Arg2R >,
	Arg1R, Arg2R > > rhs) {

	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, TraitsL, LabelL> > exprl_t;
	typedef labeled_btensor_expr< N, T,
		labeled_btensor_expr_op< N, T, NArgR,
		labeled_btensor_expr_op_add< NArgR, Arg1R, Arg2R >,
		Arg1R, Arg2R > > exprr_t;
	typedef labeled_btensor_expr_op_add < NArgR + 1,
		exprl_t, exprr_t> opadd_t;
	typedef labeled_btensor_expr_op < N, T, NArgR + 1, opadd_t,
		exprl_t, exprr_t> op_t;
	return labeled_btensor_expr<N, T, op_t > (
		op_t(lhs, exprr_t(rhs)));
}

/**	\brief Addition of two sums

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, size_t NArgL, typename Arg1L, typename Arg2L,
size_t NArgR, typename Arg1R, typename Arg2R>
labeled_btensor_expr<N, T, labeled_btensor_expr_op< N, T, NArgL + NArgR,
labeled_btensor_expr_op_add < NArgL + NArgR,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< N, T, NArgL,
labeled_btensor_expr_op_add< NArgL, Arg1L, Arg2L >, Arg1L, Arg2L > >,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< N, T, NArgR,
labeled_btensor_expr_op_add< NArgR, Arg1R, Arg2R >, Arg1R, Arg2R > > >,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< N, T, NArgL,
labeled_btensor_expr_op_add< NArgL, Arg1L, Arg2L >, Arg1L, Arg2L > >,

labeled_btensor_expr< N, T, labeled_btensor_expr_op< N, T, NArgR,
labeled_btensor_expr_op_add< NArgR, Arg1R, Arg2R >, Arg1R, Arg2R > >

> >
operator+(labeled_btensor_expr<N, T, labeled_btensor_expr_op< N, T, NArgL,
	labeled_btensor_expr_op_add<NArgL, Arg1L, Arg2L >, Arg1L, Arg2L > > lhs,
	labeled_btensor_expr<N, T, labeled_btensor_expr_op< N, T, NArgR,
	labeled_btensor_expr_op_add<NArgR, Arg1R, Arg2R >,
	Arg1R, Arg2R > > rhs) {

	typedef labeled_btensor_expr< N, T,
		labeled_btensor_expr_op< N, T, NArgL,
		labeled_btensor_expr_op_add< NArgL, Arg1L, Arg2L >,
		Arg1L, Arg2L > > exprl_t;
	typedef labeled_btensor_expr< N, T,
		labeled_btensor_expr_op< N, T, NArgR,
		labeled_btensor_expr_op_add< NArgR, Arg1R, Arg2R >,
		Arg1R, Arg2R > > exprr_t;
	typedef labeled_btensor_expr_op_add < NArgL + NArgR,
		exprl_t, exprr_t> opadd_t;
	typedef labeled_btensor_expr_op < N, T, NArgL + NArgR, opadd_t,
		exprl_t, exprr_t> op_t;
	return labeled_btensor_expr<N, T, op_t > (
		op_t(exprl_t(lhs), exprr_t(rhs)));
}

/**	\brief Multiplication of a tensor (rhs) by a scalar (lhs)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Traits, typename Label>
labeled_btensor_expr<N, T,
labeled_btensor_expr_ident<N, T, Traits, Label> >
operator*(T lhs, labeled_btensor<N, T, Traits, Label> rhs) {
	typedef labeled_btensor_expr_ident<N, T, Traits, Label> expr_id_t;
	typedef labeled_btensor_expr<N, T, expr_id_t > expr_t;
	return expr_t(expr_id_t(rhs, lhs));
}

/**	\brief Multiplication of a tensor (lhs) by a scalar (rhs)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Traits, typename Label>
labeled_btensor_expr<N, T,
labeled_btensor_expr_ident<N, T, Traits, Label> >
operator*(labeled_btensor<N, T, Traits, Label> lhs, T rhs) {
	typedef labeled_btensor_expr_ident<N, T, Traits, Label> expr_id_t;
	typedef labeled_btensor_expr<N, T, expr_id_t > expr_t;
	return expr_t(expr_id_t(lhs, rhs));
}

/**	\brief Multiplication of an expression (rhs) by a scalar (lhs)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename ExprR>
labeled_btensor_expr<N, T, ExprR>
operator*(T lhs, labeled_btensor_expr<N, T, ExprR> rhs) {
	rhs.multiply(lhs); return rhs;
}

/**	\brief Multiplication of an expression (lhs) by a scalar (rhs)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename ExprR>
labeled_btensor_expr<N, T, ExprR>
operator*(labeled_btensor_expr<N, T, ExprR> lhs, T rhs) {
	lhs.multiply(rhs); return lhs;
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_H

