#ifndef LIBTENSOR_LETTER_EXPR_H
#define LIBTENSOR_LETTER_EXPR_H

#include "defs.h"
#include "exception.h"

/**	\defgroup libtensor_letter_expr Letter index expressions
	\ingroup libtensor

	The members of this group provide the facility to operate %letter
	indexes.

	<b>See also:</b>

	 * libtensor::letter
**/

namespace libtensor {

class letter;

/**	\brief Base class for %letter %index expressions

	\ingroup libtensor_letter_expr
**/
template<size_t N>
class letter_expr_base {
};

/**	\brief Expression using %letter %tensor indexes

	\ingroup libtensor_letter_expr
**/
template<size_t N, typename T>
class letter_expr : public letter_expr_base<N> {
private:
	T m_t;

public:
	letter_expr(const T &t) : m_t(t) { }
	letter_expr(const letter_expr<N,T> &e) : m_t(e.m_t) { }
};

/**	\brief Identity expression

	\ingroup libtensor_letter_expr
**/
class letter_expr_ident {
private:
	letter &m_let;

public:
	letter_expr_ident(letter &l) : m_let(l) { }
};

/**	\brief Binary operation expression

	\ingroup libtensor_letter_expr
**/
template<typename T1, typename T2, typename Op>
class letter_expr_binop {
private:
	T1 m_t1; //!< Left expression
	T2 m_t2; //!< Right expression

public:
	letter_expr_binop(const T1 &t1, const T2 &t2) : m_t1(t1), m_t2(t2) { }
};

/**	\brief Bitwise OR (|) binary operation

	\ingroup libtensor_letter_expr
**/
template<typename T1, typename T2>
class letter_expr_binop_or {
};

/**	\brief Bitwise OR (|) operator for two letters

	\ingroup libtensor_letter_expr
**/
inline letter_expr< 2, letter_expr_binop<
	letter_expr< 1,letter_expr_ident>, letter_expr<1,letter_expr_ident>,
	letter_expr_binop_or<
		letter_expr<1,letter_expr_ident>,
		letter_expr<1,letter_expr_ident>
	> > >
operator|(letter &l1, letter &l2) {
	typedef letter_expr<1,letter_expr_ident> expr_t;
	typedef letter_expr_binop_or<expr_t,expr_t> binop_or_t;
	typedef letter_expr_binop<expr_t,expr_t,binop_or_t> binop_t;
	return letter_expr<2,binop_t>(binop_t(expr_t(l1), expr_t(l2)));
}

/**	\brief Bitwise OR (|) operator for an expression and a %letter

	\ingroup libtensor_letter_expr
**/
template<size_t N, typename Expr>
inline letter_expr< N+1, letter_expr_binop<
	letter_expr<N,Expr>, letter_expr<1,letter_expr_ident>,
	letter_expr_binop_or<
		letter_expr<N,Expr>, letter_expr<1,letter_expr_ident>
	> > >
operator|(letter_expr<N,Expr> expr1, letter &l2) {
	typedef letter_expr<N,Expr> expr1_t;
	typedef letter_expr<1,letter_expr_ident> expr2_t;
	typedef letter_expr_binop_or<expr1_t,expr2_t> binop_or_t;
	typedef letter_expr_binop<expr1_t,expr2_t,binop_or_t> binop_t;
	return letter_expr<N+1,binop_t>(binop_t(expr1, expr2_t(l2)));
}

} // namespace libtensor

#endif // LIBTENSOR_LETTER_EXPR_H

