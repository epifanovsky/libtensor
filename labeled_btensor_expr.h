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

public:
	static const size_t k_len = expression_t::k_len;

private:
	Expr m_t;

public:

	labeled_btensor_expr(const Expr &t) : m_t(t) {
	}

	labeled_btensor_expr(const labeled_btensor_expr<N, T, Expr> &e) :
		m_t(e.m_t) {
	}

	bool contains(const letter &let) const {
		return m_t.contains(let);
	}

	void multiply(T coeff) {
		m_t.multiply(coeff);
	}
};

/**	\brief Identity expression

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Assignable, typename Label>
class labeled_btensor_expr_ident {
public:
	typedef labeled_btensor<N, T, Assignable, Label> labeled_btensor_t;

public:
	static const size_t k_len = 1;

private:
	labeled_btensor_t &m_t;
	T m_coeff;

public:

	labeled_btensor_expr_ident(labeled_btensor_t &t) : m_t(t) {
	}

	bool contains(const letter &let) const {
		return m_t.contains(let);
	}
};

/**	\brief Identity expression (specialized for double)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, bool Assignable, typename Label>
class labeled_btensor_expr_ident<N, double, Assignable, Label> {
public:
	typedef labeled_btensor<N, double, Assignable, Label> labeled_btensor_t;

public:
	static const size_t k_len = 1;

private:
	labeled_btensor_t &m_t;
	double m_coeff;

public:

	labeled_btensor_expr_ident(labeled_btensor_t &t, double coeff = 1.0) :
		m_t(t), m_coeff(coeff) {
	}

	bool contains(const letter &let) const {
		return m_t.contains(let);
	}

	void multiply(double coeff) {
		m_coeff *= coeff;
	}
};

/**	\brief Operation expression
	\tparam N Tensor order
	\tparam T Tensor element type
	\tparam Op Operation
	\tparam ExprL LHS expression
	\tparam ExprR RHS expression

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Op, typename ExprL, typename ExprR>
class labeled_btensor_expr_op {
public:
	static const size_t k_len = ExprL::k_len + ExprR::k_len;

private:
	ExprL m_exprl; //!< Left expression
	ExprR m_exprr; //!< Right expression

public:

	labeled_btensor_expr_op(const ExprL &exprl, const ExprR &exprr) :
		m_exprl(exprl), m_exprr(exprr) {
	}

	bool contains(const letter &let) const {
		return m_exprl.contains(let) && m_exprr.contains(let);
	}
};

/**	\brief Operation expression (specialized for double)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename Op, typename ExprL, typename ExprR>
class labeled_btensor_expr_op<N, double, Op, ExprL, ExprR> {
public:
	static const size_t k_len = ExprL::k_len + ExprR::k_len;

private:
	ExprL m_exprl; //!< Left expression
	ExprR m_exprr; //!< Right expression
	double m_coeff; //!< Coefficient

public:

	labeled_btensor_expr_op(const ExprL &exprl, const ExprR &exprr,
		double coeff = 1.0) : m_exprl(exprl), m_exprr(exprr),
		m_coeff(coeff) {
	}

	bool contains(const letter &let) const {
		return m_exprl.contains(let) && m_exprr.contains(let);
	}

	void multiply(double coeff) {
		m_coeff *= coeff;
	}
};

/**	\brief Addition operation
	\tparam ExprL LHS expression
	\tparam ExprR RHS expression

	\ingroup libtensor_btensor_expr
 **/
template<typename ExprL, typename ExprR>
class labeled_btensor_expr_op_add {
public:
	static const size_t k_len = ExprL::k_len + ExprR::k_len;
};

/**	\brief Addition of two tensors (plain + plain)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool AssignableL, typename LabelL,
	bool AssignableR, typename LabelR>
labeled_btensor_expr<N, T,
	labeled_btensor_expr_op<N, T,
	labeled_btensor_expr_op_add<
		labeled_btensor_expr<N, T,
			labeled_btensor_expr_ident<N, T, AssignableL, LabelL> >,
		labeled_btensor_expr<N, T,
			labeled_btensor_expr_ident<N, T, AssignableR, LabelR> > >,
	labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, AssignableL, LabelL> >,
	labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, AssignableR, LabelR>
	> >
>
operator+(labeled_btensor<N, T, AssignableL, LabelL> lhs,
	labeled_btensor<N, T, AssignableR, LabelR> rhs) {
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, AssignableL, LabelL> > exprl_t;
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, AssignableR, LabelR> > exprr_t;
	typedef labeled_btensor_expr_op_add<exprl_t, exprr_t> opadd_t;
	typedef labeled_btensor_expr_op<N, T, opadd_t, exprl_t, exprr_t> op_t;
	return labeled_btensor_expr<N, T, op_t>(
		op_t(exprl_t(lhs), exprr_t(rhs)));
}

/**	\brief Addition of two tensors (plain + expression)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool AssignableL, typename LabelL,
	typename ExprR>
labeled_btensor_expr<N, T,
	labeled_btensor_expr_op<N, T,
	labeled_btensor_expr_op_add<
		labeled_btensor_expr<N, T,
			labeled_btensor_expr_ident<N, T, AssignableL, LabelL> >,
		labeled_btensor_expr<N, T, ExprR> >,
	labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, AssignableL, LabelL> >,
	labeled_btensor_expr<N, T, ExprR>
	>
>
operator+(labeled_btensor<N, T, AssignableL, LabelL> lhs,
	labeled_btensor_expr<N, T, ExprR> rhs) {
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, AssignableL, LabelL> > exprl_t;
	typedef labeled_btensor_expr<N, T, ExprR> exprr_t;
	typedef labeled_btensor_expr_op_add<exprl_t, exprr_t> opadd_t;
	typedef labeled_btensor_expr_op<N, T, opadd_t, exprl_t, exprr_t> op_t;
	return labeled_btensor_expr<N, T, op_t>(op_t(exprl_t(lhs), rhs));
}

/**	\brief Addition of two tensors (expression + plain)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename ExprL, bool AssignableR,
	typename LabelR>
labeled_btensor_expr<N, T,
	labeled_btensor_expr_op<N, T,
	labeled_btensor_expr_op_add<
		labeled_btensor_expr<N, T, ExprL>,
		labeled_btensor_expr<N, T,
			labeled_btensor_expr_ident<N, T, AssignableR, LabelR> > >,
	labeled_btensor_expr<N, T, ExprL>,
	labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, AssignableR, LabelR>
	> >
>
operator+(labeled_btensor_expr<N, T, ExprL> lhs,
	labeled_btensor<N, T, AssignableR, LabelR> rhs) {
	typedef labeled_btensor_expr<N, T, ExprL> exprl_t;
	typedef labeled_btensor_expr<N, T,
		labeled_btensor_expr_ident<N, T, AssignableR, LabelR> > exprr_t;
	typedef labeled_btensor_expr_op_add<exprl_t, exprr_t> opadd_t;
	typedef labeled_btensor_expr_op<N, T, opadd_t, exprl_t, exprr_t> op_t;
	return labeled_btensor_expr<N, T, op_t>(op_t(lhs, exprr_t(rhs)));
}

/**	\brief Addition of two tensors (expression + expression)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename ExprL, typename ExprR>
labeled_btensor_expr<N, T,
	labeled_btensor_expr_op<N, T,
	labeled_btensor_expr_op_add<
		labeled_btensor_expr<N, T, ExprL>,
		labeled_btensor_expr<N, T, ExprR> >,
	labeled_btensor_expr<N, T, ExprL>,
	labeled_btensor_expr<N, T, ExprR>
	>
>
operator+(labeled_btensor_expr<N, T, ExprL> lhs,
	labeled_btensor_expr<N, T, ExprR> rhs) {
	typedef labeled_btensor_expr<N, T, ExprL> exprl_t;
	typedef labeled_btensor_expr<N, T, ExprR> exprr_t;
	typedef labeled_btensor_expr_op_add<exprl_t, exprr_t> opadd_t;
	typedef labeled_btensor_expr_op<N, T, opadd_t, exprl_t, exprr_t> op_t;
	return labeled_btensor_expr<N, T, op_t>(op_t(lhs, rhs));
}

/**	\brief Multiplication of a tensor (rhs) by a scalar (lhs)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Assignable, typename Label>
labeled_btensor_expr<N, T,
labeled_btensor_expr_ident<N, T, Assignable, Label> >
operator*(T lhs, labeled_btensor<N, T, Assignable, Label> rhs) {
	typedef labeled_btensor_expr_ident<N, T, Assignable, Label> expr_id_t;
	typedef labeled_btensor_expr<N, T, expr_id_t > expr_t;
	return expr_t(expr_id_t(rhs, lhs));
}

/**	\brief Multiplication of a tensor (lhs) by a scalar (rhs)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Assignable, typename Label>
labeled_btensor_expr<N, T,
labeled_btensor_expr_ident<N, T, Assignable, Label> >
operator*(labeled_btensor<N, T, Assignable, Label> lhs, T rhs) {
	typedef labeled_btensor_expr_ident<N, T, Assignable, Label> expr_id_t;
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

