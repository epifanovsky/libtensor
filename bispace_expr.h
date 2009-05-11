#ifndef LIBTENSOR_BISPACE_EXPR_H
#define	LIBTENSOR_BISPACE_EXPR_H

#include "defs.h"
#include "exception.h"
#include "rc_ptr.h"
#include "bispace.h"
#include "dimensions.h"

/**	\defgroup libtensor_bispace_expr Block %index space expressions
	\ingroup libtensor

	The members of this group provide the facility to create block %index
	spaces with arbitrary symmetry.
 **/

namespace libtensor {

template<size_t N> class bispace;

/**	\brief Interface for block %index space expressions
	\tparam N Expression order

	\ingroup libtensor_bispace_expr
 **/
template<size_t N>
class bispace_expr_i {
public:
	virtual rc_ptr<bispace_expr_i<N> > clone() const = 0;
	virtual const dimensions<N> &get_dims() const = 0;
	virtual const bispace < 1 > & operator[](size_t i) const = 0;
};

/**	\brief Block %index space expression
	\tparam N Expression order
	\tparam T Underlying expression type

	\ingroup libtensor_bispace_expr
 **/
template<size_t N, typename T>
class bispace_expr : public bispace_expr_i<N> {
public:
	static const size_t k_order = N;

private:
	T m_t;
	dimensions<N> m_dims;

public:

	bispace_expr(const T &t) : m_t(t), m_dims(t.get_dims()) {
	}

	bispace_expr(const bispace_expr<N, T> &e) :
	m_t(e.m_t), m_dims(e.m_dims) {
	}

	virtual rc_ptr<bispace_expr_i<N> > clone() const {
		return rc_ptr<bispace_expr_i<N> >(
			new bispace_expr<N, T > (*this));
	}

	virtual const dimensions<N> &get_dims() const {
		return m_dims;
	}

	virtual const bispace < 1 > & operator[](size_t i) const {
		return m_t[i];
	}
};

/**	\brief Identity expression

	\ingroup libtensor_bispace_expr
 **/
template<size_t N>
class bispace_expr_ident {
public:
	static const size_t k_order = N;

private:
	bispace<N> &m_bis;

public:

	bispace_expr_ident(bispace<N> &bis) : m_bis(bis) {
	}

	const dimensions<N> &get_dims() const {
		return m_bis.get_dims();
	}

	const bispace < 1 > & operator[](size_t i) const {
		return m_bis[i];
	}
};

/**	\brief Binary operation expression

	\ingroup libtensor_bispace_expr
 **/
template<typename T1, typename T2, typename Op>
class bispace_expr_binop {
public:
	static const size_t k_order = T1::k_order + T2::k_order;

private:
	T1 m_t1; //!< Left expression
	T2 m_t2; //!< Right expression
	dimensions<k_order> m_dims;

public:

	bispace_expr_binop(const T1 &t1, const T2 &t2) : m_t1(t1), m_t2(t2),
	m_dims(Op::get_dims(t1, t2)) {
	}

	const dimensions<k_order> &get_dims() const {
		return m_dims;
	}

	const bispace < 1 > & operator[](size_t i) const {
		return i < T1::k_order ? m_t1[i] : m_t2[i - T1::k_order];
	}
};

/**	\brief Bitwise OR (|) binary operation

	\ingroup libtensor_bispace_expr
 **/
template<typename T1, typename T2>
class bispace_expr_binop_or {
public:
	static const size_t k_order = T1::k_order + T2::k_order;

public:

	static const dimensions<k_order> get_dims(const T1 &t1, const T2 &t2) {
		dimensions<T1::k_order> dims_t1(t1.get_dims());
		dimensions<T2::k_order> dims_t2(t2.get_dims());
		index<k_order> i1, i2;
		size_t i = 0;
		for(size_t j = 0; j < T1::k_order; j++, i++) i2[i] = dims_t1[j] - 1;
		for(size_t j = 0; j < T2::k_order; j++, i++) i2[i] = dims_t2[j] - 1;
		return dimensions<k_order > (index_range<k_order > (i1, i2));
	}
};

/**	\brief Bitwise AND (&) binary operation

	\ingroup libtensor_bispace_expr
 **/
template<typename T1, typename T2>
class bispace_expr_binop_and {
public:
	static const size_t k_order = T1::k_order + T2::k_order;

public:

	static const dimensions<k_order> get_dims(const T1 &t1, const T2 &t2) {
		dimensions<T1::k_order> dims_t1(t1.get_dims());
		dimensions<T2::k_order> dims_t2(t2.get_dims());
		index<k_order> i1, i2;
		size_t i = 0;
		for(size_t j = 0; j < T1::k_order; j++, i++) i2[i] = dims_t1[j] - 1;
		for(size_t j = 0; j < T2::k_order; j++, i++) i2[i] = dims_t2[j] - 1;
		return dimensions<k_order > (index_range<k_order > (i1, i2));
	}
};

/**	\brief Bitwise XOR (^) binary operation

	\ingroup libtensor_bispace_expr
 **/
template<typename T1, typename T2>
class bispace_expr_binop_xor {
public:
	static const size_t k_order = T1::k_order + T2::k_order;

public:

	static const dimensions<k_order> get_dims(const T1 &t1, const T2 &t2) {
		dimensions<T1::k_order> dims_t1(t1.get_dims());
		dimensions<T2::k_order> dims_t2(t2.get_dims());
		index<k_order> i1, i2;
		size_t i = 0;
		for(size_t j = 0; j < T1::k_order; j++, i++) i2[i] = dims_t1[j] - 1;
		for(size_t j = 0; j < T2::k_order; j++, i++) i2[i] = dims_t2[j] - 1;
		return dimensions<k_order > (index_range<k_order > (i1, i2));
	}
};

/**	\brief Multiplication (*) binary operation

	\ingroup libtensor_bispace_expr
 **/
template<typename T1, typename T2>
class bispace_expr_binop_mul {
public:
	static const size_t k_order = T1::k_order + T2::k_order;

public:

	static const dimensions<k_order> get_dims(const T1 &t1, const T2 &t2) {
		dimensions<T1::k_order> dims_t1(t1.get_dims());
		dimensions<T2::k_order> dims_t2(t2.get_dims());
		index<k_order> i1, i2;
		size_t i = 0;
		for(size_t j = 0; j < T1::k_order; j++, i++) i2[i] = dims_t1[j] - 1;
		for(size_t j = 0; j < T2::k_order; j++, i++) i2[i] = dims_t2[j] - 1;
		return dimensions<k_order > (index_range<k_order > (i1, i2));
	}
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
operator&(bispace<N> &lhs, bispace<M> &rhs) {
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
operator&(bispace_expr<M, Expr> lhs, bispace<N> &rhs) {
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
operator&(bispace<N> &lhs, bispace_expr<M, Expr> rhs) {
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
operator*(bispace<N> &lhs, bispace<M> &rhs) {
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
operator*(bispace_expr<M, Expr> lhs, bispace<N> &rhs) {
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
operator*(bispace<N> &lhs, bispace_expr<M, Expr> rhs) {
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

