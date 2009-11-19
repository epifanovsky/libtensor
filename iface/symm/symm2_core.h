#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM2_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM2_CORE_H

#include "defs.h"
#include "exception.h"
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, bool Sym, typename T, typename SubCore>
class symm2_eval;


/**	\brief Expression core for the symmetrization of a pair of indexes
	\tparam N Tensor order.
	\tparam Sym Symmetrization/antisymmetrization.
	\tparam SubCore Sub-expression core type.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, bool Sym, typename T, typename SubCore>
class symm2_core {
public:
	static const char *k_clazz; //!< Class name

public:
	 //!	Evaluating container type
	typedef symm2_eval<N, Sym, T, SubCore> eval_container_t;

	//!	Sub-expression type
	typedef expr<N, T, SubCore> sub_expression_t;

private:
	letter_expr<2> m_sym; //!< Symmetrized indexes
	sub_expression_t m_expr; //!< Sub-expression

public:
	/**	\brief Creates the expression core
		\param sym Two-letter expression indicating symmetrized indexes
		\param expr Sub-expression.
	 **/
	symm2_core(const letter_expr<2> &sym, const sub_expression_t &expr);

	/**	\brief Copy constructor
	 **/
	symm2_core(const symm2_core<N, Sym, T, SubCore> &core);

	/**	\brief Returns the symmetrized indexes
	 **/
	const letter_expr<2> &get_sym() const {
		return m_sym;
	}

	/**	\brief Returns the sub-expression
	 **/
	expr<N, T, SubCore> &get_sub_expr() {
		return m_expr;
	}

	/**	\brief Returns the sub-expression, const version
	 **/
	const expr<N, T, SubCore> &get_sub_expr() const {
		return m_expr;
	}

	/**	\brief Returns whether the result's label contains a %letter
		\param let Letter.
	 **/
	bool contains(const letter &let) const;

	/**	\brief Returns the %index of a %letter in the result's label
		\param let Letter.
		\throw expr_exception If the label does not contain the
			requested letter.
	 **/
	size_t index_of(const letter &let) const throw(expr_exception);

	/**	\brief Returns the %letter at a given position in
			the result's label
		\param i Letter index.
		\throw out_of_bounds If the index is out of bounds.
	 **/
	const letter &letter_at(size_t i) const throw(out_of_bounds);

};


template<size_t N, bool Sym, typename T, typename SubCore>
const char *symm2_core<N, Sym, T, SubCore>::k_clazz =
	"symm2_core<N, Sym, T, SubCore>";


template<size_t N, bool Sym, typename T, typename SubCore>
symm2_core<N, Sym, T, SubCore>::symm2_core(
	const letter_expr<2> &sym, const sub_expression_t &expr) :

	m_sym(sym), m_expr(expr) {

	static const char *method =
		"symm2_core(const letter_expr<2>&, const Expr&)";

	if(m_sym.letter_at(0) == m_sym.letter_at(1)) {
		throw expr_exception(g_ns, k_clazz, method,
			__FILE__, __LINE__,
			"Symmetrized indexes must be different.");
	}
	for(size_t i = 0; i < 2; i++) {
		const letter &l = m_sym.letter_at(i);
		if(!m_expr.contains(l)) {
			throw expr_exception(g_ns, k_clazz, method,
				__FILE__, __LINE__,
				"Symmetrized index is absent from the result.");
		}
	}
}


template<size_t N, bool Sym, typename T, typename SubCore>
symm2_core<N, Sym, T, SubCore>::symm2_core(
	const symm2_core<N, Sym, T, SubCore> &core) :

	m_sym(core.m_sym), m_expr(core.m_expr) {

}


template<size_t N, bool Sym, typename T, typename SubCore>
bool symm2_core<N, Sym, T, SubCore>::contains(const letter &let) const {

	return m_expr.contains(let);
}


template<size_t N, bool Sym, typename T, typename SubCore>
size_t symm2_core<N, Sym, T, SubCore>::index_of(const letter &let) const
	throw(expr_exception) {

	return m_expr.index_of(let);
}


template<size_t N, bool Sym, typename T, typename SubCore>
const letter &symm2_core<N, Sym, T, SubCore>::letter_at(size_t i) const
	throw(out_of_bounds) {

	return m_expr.letter_at(i);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM2_CORE_H
