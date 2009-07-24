#ifndef LIBTENSOR_LETTER_EXPR_H
#define LIBTENSOR_LETTER_EXPR_H

#include "defs.h"
#include "exception.h"
#include "core/permutation.h"

/**	\defgroup libtensor_letter_expr Letter index expressions
	\ingroup libtensor_iface

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
public:
	static const size_t k_size = T::k_size;

private:
	T m_t;

public:
	letter_expr(const T &t) : m_t(t) { }
	letter_expr(const letter_expr<N, T> &e) : m_t(e.m_t) { }

	/**	\brief Returns whether the expression contains a %letter
	 **/
	bool contains(const letter &let) const { return m_t.contains(let); }

	/**	\brief Returns the %index of a %letter in the expression
		\throw exception If the expression doesn't contain the %letter.
	 **/
	size_t index_of(const letter &let) const throw(exception);

	/**	\brief Returns the %letter at a given position
		\throw exception If the %index is out of bounds.
	 **/
	const letter &letter_at(size_t i) const throw(exception);

	/**	\brief Returns how letters in the second expression need to be
			permuted to obtain the order of letters in this
			expression
		\param e2 Second expression.
	 **/
	template<typename T2>
	permutation<N> permutation_of(const letter_expr<N, T2> &e2) const
		throw(exception);
};

/**	\brief Identity expression

	\ingroup libtensor_letter_expr
**/
class letter_expr_ident {
public:
	static const size_t k_size = 1;

private:
	const letter &m_let;

public:
	letter_expr_ident(const letter &l) : m_let(l) {}

	bool contains(const letter &let) const { return &m_let == &let; }
	size_t index_of(const letter &let) const throw(exception);
	const letter &letter_at(size_t i) const throw(exception);
};


/**	\brief Binary operation expression

	\ingroup libtensor_letter_expr
**/
template<typename T1, typename T2, typename Op>
class letter_expr_binop {
public:
	static const size_t k_size = T1::k_size + T2::k_size;

private:
	T1 m_t1; //!< Left expression
	T2 m_t2; //!< Right expression

public:
	letter_expr_binop(const T1 &t1, const T2 &t2) : m_t1(t1), m_t2(t2) { }

	bool contains(const letter &let) const;
	size_t index_of(const letter &let) const throw(exception);
	const letter &letter_at(size_t i) const throw(exception);
};

/**	\brief Bitwise OR (|) binary operation

	\ingroup libtensor_letter_expr
**/
template<typename T1, typename T2>
class letter_expr_binop_or {
};

template<size_t N, typename T>
inline size_t letter_expr<N, T>::index_of(const letter &let) const
	throw(exception) {
	return m_t.index_of(let);
}

template<size_t N, typename T>
inline const letter &letter_expr<N, T>::letter_at(size_t i) const
	throw(exception) {
	if(i >= k_size) {
		throw_exc("letter_expr<N, T>", "letter_at(size_t)",
			"Index out of bounds");
	}
	return m_t.letter_at(i);
}

template<size_t N, typename T> template<typename T2>
permutation<N> letter_expr<N, T>::permutation_of(const letter_expr<N, T2> &e2)
	const throw(exception) {

	permutation<N> perm;
	size_t idx[N];
	register size_t i;

	for(i = 0; i < N; i++) {
		idx[i] = m_t.index_of(e2.letter_at(i));
	}

	i = 0;
	while(i < N) {
		if(i > idx[i]) {
			perm.permute(i, idx[i]);
			register size_t j = idx[i];
			idx[i] = idx[j];
			idx[j] = j;
			i = 0;
		} else {
			i++;
		}
	}

	return perm;
}


inline size_t letter_expr_ident::index_of(const letter &let) const
	throw(exception) {

	if(!contains(let)) {
		throw_exc("letter_expr_ident", "index_of(const letter&)",
			"Expression doesn't contain the requested letter");
	}
	return 0;
}

inline const letter &letter_expr_ident::letter_at(size_t i) const
	throw(exception) {
	if(i != 0) {
		throw_exc("letter_expr_ident", "letter_at(size_t)",
			"Index out of bounds");
	}
	return m_let;
}

template<typename T1, typename T2, typename Op>
inline bool letter_expr_binop<T1, T2, Op>::contains(const letter &let) const {
	return m_t1.contains(let)|m_t2.contains(let);
}

template<typename T1, typename T2, typename Op>
inline size_t letter_expr_binop<T1, T2, Op>::index_of(const letter &let) const
	throw(exception) {
	if(m_t1.contains(let)) return m_t1.index_of(let);
	else return T1::k_size + m_t2.index_of(let);
}

template<typename T1, typename T2, typename Op>
inline const letter &letter_expr_binop<T1, T2, Op>::letter_at(size_t i) const
	throw(exception) {
	if(i < T1::k_size) return m_t1.letter_at(i);
	else return m_t2.letter_at(i - T1::k_size);
}

/**	\brief Unary + for one letter (returning a 1d letter_expr)

	\ingroup libtensor_letter_expr
**/
inline letter_expr< 1, letter_expr_ident>
operator+(const letter &l1 ) {
	typedef letter_expr<1,letter_expr_ident> expr_t;

	return expr_t(l1);
}


/**	\brief Bitwise OR (|) operator for two letters

	\ingroup libtensor_letter_expr
**/
inline letter_expr< 2, letter_expr_binop<
	letter_expr< 1,letter_expr_ident>, letter_expr<1,letter_expr_ident>,
	letter_expr_binop_or<
		letter_expr<1,letter_expr_ident>,
		letter_expr<1,letter_expr_ident>
	> > >
operator|(const letter &l1, const letter &l2) {
	typedef letter_expr<1,letter_expr_ident> expr_t;
	typedef letter_expr_binop_or<expr_t,expr_t> binop_or_t;
	typedef letter_expr_binop<expr_t,expr_t,binop_or_t> binop_t;
	if(&l1 == &l2) {
		throw_exc("", "operator|(const letter&, const letter&)",
			"Only unique letters are allowed");
	}
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
operator|(letter_expr<N,Expr> expr1, const letter &l2) {
	typedef letter_expr<N,Expr> expr1_t;
	typedef letter_expr<1,letter_expr_ident> expr2_t;
	typedef letter_expr_binop_or<expr1_t,expr2_t> binop_or_t;
	typedef letter_expr_binop<expr1_t,expr2_t,binop_or_t> binop_t;
	if(expr1.contains(l2)) {
		throw_exc("", "operator|(letter_expr, const letter&)",
			"Only unique letters are allowed");
	}
	return letter_expr<N+1,binop_t>(binop_t(expr1, expr2_t(l2)));
}

} // namespace libtensor

#endif // LIBTENSOR_LETTER_EXPR_H

