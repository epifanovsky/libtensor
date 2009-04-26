#ifndef LIBTENSOR_BISPACE_EXPR_H
#define	LIBTENSOR_BISPACE_EXPR_H

#include "defs.h"
#include "exception.h"
#include "bispace.h"

/**	\defgroup libtensor_bispace_expr Block %index space expressions
	\ingroup libtensor

	The members of this group provide the facility to create block %index
	spaces with arbitrary symmetry.
 **/

namespace libtensor {

template<size_t N, typename SymExprT> class bispace;

/**	\brief Base class for block %index space expressions
	\tparam N Expression order

	\ingroup libtensor_bispace_expr
 **/
template<size_t N>
class bispace_expr_base {
};

/**	\brief Block %index space expression
	\tparam N Expression order
	\tparam T Underlying expression type

	\ingroup libtensor_bispace_expr
 **/
template<size_t N, typename T>
class bispace_expr {
private:
	T m_t;

public:

	bispace_expr(const T &t) : m_t(t) {
	}

	bispace_expr(const bispace_expr<N, T> &e) : m_t(e.m_t) {
	}
};

/**	\brief Identity expression

	\ingroup libtensor_bispace_expr
 **/
template<size_t N>
class bispace_expr_ident {
private:
	bispace<N, void> &m_bis;

public:

	bispace_expr_ident(bispace<N, void> &bis) : m_bis(bis) {
	}
};

/**	\brief Binary operation expression

	\ingroup libtensor_bispace_expr
 **/
template<typename T1, typename T2, typename Op>
class bispace_expr_binop {
private:
	T1 m_t1; //!< Left expression
	T2 m_t2; //!< Right expression

public:

	bispace_expr_binop(const T1 &t1, const T2 &t2) : m_t1(t1), m_t2(t2) {
	}
};

/**	\brief Bitwise OR (|) binary operation

	\ingroup libtensor_bispace_expr
 **/
template<typename T1, typename T2>
class bispace_expr_binop_or {
};

/**	\brief Bitwise AND (&) binary operation

	\ingroup libtensor_bispace_expr
 **/
template<typename T1, typename T2>
class bispace_expr_binop_and {
};

/**	\brief Bitwise XOR (^) binary operation

	\ingroup libtensor_bispace_expr
 **/
template<typename T1, typename T2>
class bispace_expr_binop_xor {
};

/**	\brief Multiplication (*) binary operation

	\ingroup libtensor_bispace_expr
 **/
template<typename T1, typename T2>
class bispace_expr_binop_mul {
};

/**	\brief Bitwise AND (&) operator for two spaces

	The bitwise AND operation is translated into permutational symmetry

	\ingroup libtensor_bispace_expr
 **/
template<size_t N, size_t M>
inline bispace_expr< N + M, bispace_expr_binop<
bispace_expr< N, bispace_expr_ident<N> >,
bispace_expr<M, bispace_expr_ident<M> >,
bispace_expr_binop_and<
bispace_expr<N, bispace_expr_ident<N> >,
bispace_expr<M, bispace_expr_ident<M> >
> > >
operator&(bispace<N, void> &lhs, bispace<M, void> &rhs) {
	typedef bispace_expr<N, bispace_expr_ident<N> > expr1_t;
	typedef bispace_expr<M, bispace_expr_ident<M> > expr2_t;
	typedef bispace_expr_binop_and<expr1_t, expr2_t> binop_and_t;
	typedef bispace_expr_binop<expr1_t, expr2_t, binop_and_t> binop_t;
	return bispace_expr < N + M, binop_t > (
		binop_t(expr1_t(lhs), expr2_t(rhs)));
}

/**	\brief Bitwise AND (&) operator for an expression (lhs) and
		a space (rhs)

	\ingroup libtensor_bispace_expr
 **/
template<size_t N, size_t M, typename Expr>
inline bispace_expr< N + M, bispace_expr_binop<
bispace_expr<M, Expr>,
bispace_expr < N, bispace_expr_ident<N> >,
bispace_expr_binop_and<
bispace_expr<M, Expr>, bispace_expr < N, bispace_expr_ident<N> >
> > >
operator&(bispace_expr<M, Expr> lhs, bispace<N, void> &rhs) {
	typedef bispace_expr<M, Expr> expr1_t;
	typedef bispace_expr< N, bispace_expr_ident<N> > expr2_t;
	typedef bispace_expr_binop_and<expr1_t, expr2_t> binop_and_t;
	typedef bispace_expr_binop<expr1_t, expr2_t, binop_and_t> binop_t;
	return bispace_expr < N + M, binop_t > (binop_t(lhs, expr2_t(rhs)));
}

/**	\brief Bitwise AND (&) operator for a space (lhs) and
		an expression (rhs)

	\ingroup libtensor_bispace_expr
 **/
template<size_t N, size_t M, typename Expr>
inline bispace_expr< N + M, bispace_expr_binop<
bispace_expr < N, bispace_expr_ident<N> >,
bispace_expr<M, Expr>,
bispace_expr_binop_and<
bispace_expr < N, bispace_expr_ident<N> >, bispace_expr<M, Expr>
> > >
operator&(bispace<N, void> &lhs, bispace_expr<M, Expr> rhs) {
	typedef bispace_expr< N, bispace_expr_ident<N> > expr1_t;
	typedef bispace_expr<M, Expr> expr2_t;
	typedef bispace_expr_binop_and<expr1_t, expr2_t> binop_and_t;
	typedef bispace_expr_binop<expr1_t, expr2_t, binop_and_t> binop_t;
	return bispace_expr < N + M, binop_t > (binop_t(expr1_t(lhs), rhs));
}

/**	\brief Bitwise AND (&) operator for two expressions

	\ingroup libtensor_bispace_expr
 **/
template<size_t N, typename ExprL, size_t M, typename ExprR>
inline bispace_expr<N + M, bispace_expr_binop<
bispace_expr<N, ExprL>, bispace_expr<M, ExprR>,
bispace_expr_binop_and< bispace_expr<N, ExprL>, bispace_expr<M, ExprR>
> > >
operator&(bispace_expr<N, ExprL> lhs, bispace_expr<M, ExprR> rhs) {
	typedef bispace_expr<N, ExprL> expr1_t;
	typedef bispace_expr<M, ExprR> expr2_t;
	typedef bispace_expr_binop_and<expr1_t, expr2_t> binop_and_t;
	typedef bispace_expr_binop<expr1_t, expr2_t, binop_and_t> binop_t;
	return bispace_expr < N + M, binop_t > (binop_t(lhs, rhs));
}

/**	\brief Multiplication (*) operator for two spaces

	The multiplication operation is translated into direct product

	\ingroup libtensor_bispace_expr
 **/
template<size_t N, size_t M>
inline bispace_expr< N + M, bispace_expr_binop<
bispace_expr< N, bispace_expr_ident<N> >,
bispace_expr<M, bispace_expr_ident<M> >,
bispace_expr_binop_mul<
bispace_expr<N, bispace_expr_ident<N> >,
bispace_expr<M, bispace_expr_ident<M> >
> > >
operator*(bispace<N, void> &lhs, bispace<M, void> &rhs) {
	typedef bispace_expr<N, bispace_expr_ident<N> > expr1_t;
	typedef bispace_expr<M, bispace_expr_ident<M> > expr2_t;
	typedef bispace_expr_binop_mul<expr1_t, expr2_t> binop_mul_t;
	typedef bispace_expr_binop<expr1_t, expr2_t, binop_mul_t> binop_t;
	return bispace_expr < N + M, binop_t > (
		binop_t(expr1_t(lhs), expr2_t(rhs)));
}

/**	\brief Multiplication (*) operator for an expression (lhs) and
		a space (rhs)

	\ingroup libtensor_bispace_expr
 **/
template<size_t N, size_t M, typename Expr>
inline bispace_expr< N + M, bispace_expr_binop<
bispace_expr<M, Expr>,
bispace_expr < N, bispace_expr_ident<N> >,
bispace_expr_binop_mul<
bispace_expr<M, Expr>, bispace_expr < N, bispace_expr_ident<N> >
> > >
operator*(bispace_expr<M, Expr> lhs, bispace<N, void> &rhs) {
	typedef bispace_expr<M, Expr> expr1_t;
	typedef bispace_expr< N, bispace_expr_ident<N> > expr2_t;
	typedef bispace_expr_binop_mul<expr1_t, expr2_t> binop_mul_t;
	typedef bispace_expr_binop<expr1_t, expr2_t, binop_mul_t> binop_t;
	return bispace_expr < N + M, binop_t > (binop_t(lhs, expr2_t(rhs)));
}

/**	\brief Multiplication (*) operator for a space (lhs) and
		an expression (rhs)

	\ingroup libtensor_bispace_expr
 **/
template<size_t N, size_t M, typename Expr>
inline bispace_expr< N + M, bispace_expr_binop<
bispace_expr < N, bispace_expr_ident<N> >,
bispace_expr<M, Expr>,
bispace_expr_binop_mul<
bispace_expr < N, bispace_expr_ident<N> >, bispace_expr<M, Expr>
> > >
operator*(bispace<N, void> &lhs, bispace_expr<M, Expr> rhs) {
	typedef bispace_expr< N, bispace_expr_ident<N> > expr1_t;
	typedef bispace_expr<M, Expr> expr2_t;
	typedef bispace_expr_binop_mul<expr1_t, expr2_t> binop_mul_t;
	typedef bispace_expr_binop<expr1_t, expr2_t, binop_mul_t> binop_t;
	return bispace_expr < N + M, binop_t > (binop_t(expr1_t(lhs), rhs));
}

/**	\brief Multiplication (*) operator for two expressions

	\ingroup libtensor_bispace_expr
 **/
template<size_t N, typename ExprL, size_t M, typename ExprR>
inline bispace_expr<N + M, bispace_expr_binop<
bispace_expr<N, ExprL>, bispace_expr<M, ExprR>,
bispace_expr_binop_mul< bispace_expr<N, ExprL>, bispace_expr<M, ExprR>
> > >
operator*(bispace_expr<N, ExprL> lhs, bispace_expr<M, ExprR> rhs) {
	typedef bispace_expr<N, ExprL> expr1_t;
	typedef bispace_expr<M, ExprR> expr2_t;
	typedef bispace_expr_binop_mul<expr1_t, expr2_t> binop_mul_t;
	typedef bispace_expr_binop<expr1_t, expr2_t, binop_mul_t> binop_t;
	return bispace_expr < N + M, binop_t > (binop_t(lhs, rhs));
}

} // namespace libtensor

#endif // LIBTENSOR_BISPACE_EXPR_H

