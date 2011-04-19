#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM1_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM1_CORE_H

#include "../../defs.h"
#include "../../exception.h"
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, bool Sym, typename T, typename SubCore>
class symm1_eval;


/**	\brief Expression core for the symmetrization of one %index against
		a set of indexes
	\tparam N Tensor order.
	\tparam M Number of indexes in the set.
	\tparam Sym Symmetrization/antisymmetrization.
	\tparam SubCore Sub-expression core type.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, bool Sym, typename T, typename SubCore>
class symm1_core {
public:
	static const char *k_clazz; //!< Class name

public:
	 //!	Evaluating container type
	typedef symm1_eval<N, M, Sym, T, SubCore> eval_container_t;

	//!	Sub-expression type
	typedef expr<N, T, SubCore> sub_expression_t;

private:
	const letter &m_l1; //!< Symmetrized %index
	letter_expr<M> m_sym2; //!< Set of target indexes
	sub_expression_t m_expr; //!< Sub-expression

public:
	/**	\brief Creates the expression core
		\param l1 Symmetrized %index
		\param sym2 Set of target indexes
		\param expr Sub-expression.
	 **/
	symm1_core(const letter &l1, const letter_expr<M> &sym2,
		const sub_expression_t &expr);

	/**	\brief Copy constructor
	 **/
	symm1_core(const symm1_core<N, M, Sym, T, SubCore> &core);

	/**	\brief Returns the symmetrized %index letter
	 **/
	const letter &get_l1() const {
		return m_l1;
	}

	/**	\brief Returns the set of target indexes
	 **/
	const letter_expr<M> &get_sym2() const {
		return m_sym2;
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


template<size_t N, size_t M, bool Sym, typename T, typename SubCore>
const char *symm1_core<N, M, Sym, T, SubCore>::k_clazz =
	"symm1_core<N, M, Sym, T, SubCore>";


template<size_t N, size_t M, bool Sym, typename T, typename SubCore>
symm1_core<N, M, Sym, T, SubCore>::symm1_core(const letter &l1,
	const letter_expr<M> &sym2, const sub_expression_t &expr) :

	m_l1(l1), m_sym2(sym2), m_expr(expr) {

	static const char *method = "symm1_core(const letter&, "
		"const letter_expr<M>&, const Expr&)";

	if(sym2.contains(l1)) {
		throw expr_exception(g_ns, k_clazz, method,
			__FILE__, __LINE__,
			"Symmetrized indexes must be different.");
	}
	if(!m_expr.contains(l1)) {
		throw expr_exception(g_ns, k_clazz, method,
			__FILE__, __LINE__, "Symmetrized index is "
			"absent from the sub-expression.");
	}
	for(size_t i = 0; i < M; i++) {
		const letter &l2 = m_sym2.letter_at(i);
		if(!m_expr.contains(l2)) {
			throw expr_exception(g_ns, k_clazz, method,
				__FILE__, __LINE__, "Symmetrized index is "
				"absent from the sub-expression.");
		}
	}
}


template<size_t N, size_t M, bool Sym, typename T, typename SubCore>
symm1_core<N, M, Sym, T, SubCore>::symm1_core(
	const symm1_core<N, M, Sym, T, SubCore> &core) :

	m_l1(core.m_l1), m_sym2(core.m_sym2), m_expr(core.m_expr) {

}


template<size_t N, size_t M, bool Sym, typename T, typename SubCore>
bool symm1_core<N, M, Sym, T, SubCore>::contains(const letter &let) const {

	return m_expr.contains(let);
}


template<size_t N, size_t M, bool Sym, typename T, typename SubCore>
size_t symm1_core<N, M, Sym, T, SubCore>::index_of(const letter &let) const
	throw(expr_exception) {

	return m_expr.index_of(let);
}


template<size_t N, size_t M, bool Sym, typename T, typename SubCore>
const letter &symm1_core<N, M, Sym, T, SubCore>::letter_at(size_t i) const
	throw(out_of_bounds) {

	return m_expr.letter_at(i);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM1_CORE_H
