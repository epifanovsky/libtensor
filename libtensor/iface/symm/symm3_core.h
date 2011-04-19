#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM3_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM3_CORE_H

#include "../../defs.h"
#include "../../exception.h"
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, bool Sym, typename T, typename SubCore>
class symm3_eval;


/**	\brief Expression core for the symmetrization over three indexes
	\tparam N Tensor order.
	\tparam Sym Symmetrization/antisymmetrization.
	\tparam SubCore Sub-expression core type.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, bool Sym, typename T, typename SubCore>
class symm3_core {
public:
	static const char *k_clazz; //!< Class name

public:
	 //!	Evaluating container type
	typedef symm3_eval<N, Sym, T, SubCore> eval_container_t;

	//!	Sub-expression type
	typedef expr<N, T, SubCore> sub_expression_t;

private:
	const letter &m_l1; //!< First %index
	const letter &m_l2; //!< Second %index
	const letter &m_l3; //!< Third %index
	sub_expression_t m_expr; //!< Sub-expression

public:
	/**	\brief Creates the expression core
		\param l1 First symmetrized %index.
		\param l2 Second symmetrized %index.
		\param l3 Third symmetrized %index.
		\param expr Sub-expression.
	 **/
	symm3_core(const letter &l1, const letter &l2, const letter &l3,
		const sub_expression_t &expr);

	/**	\brief Copy constructor
	 **/
	symm3_core(const symm3_core<N, Sym, T, SubCore> &core);

	/**	\brief Returns the first symmetrized %index
	 **/
	const letter &get_l1() const {
		return m_l1;
	}

	/**	\brief Returns the second symmetrized %index
	 **/
	const letter &get_l2() const {
		return m_l2;
	}

	/**	\brief Returns the third symmetrized %index
	 **/
	const letter &get_l3() const {
		return m_l3;
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
const char *symm3_core<N, Sym, T, SubCore>::k_clazz =
	"symm3_core<N, Sym, T, SubCore>";


template<size_t N, bool Sym, typename T, typename SubCore>
symm3_core<N, Sym, T, SubCore>::symm3_core(const letter &l1,
	const letter &l2, const letter &l3, const sub_expression_t &expr) :

	m_l1(l1), m_l2(l2), m_l3(l3), m_expr(expr) {

	static const char *method = "symm3_core(const letter&, "
		"const letter&, const letter&, const Expr&)";

	if(m_l1 == m_l2 || m_l1 == m_l3 || m_l2 == m_l3) {
		throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Symmetrized indexes must be different.");
	}
}


template<size_t N, bool Sym, typename T, typename SubCore>
symm3_core<N, Sym, T, SubCore>::symm3_core(
	const symm3_core<N, Sym, T, SubCore> &core) :

	m_l1(core.m_l1), m_l2(core.m_l2), m_l3(core.m_l3), m_expr(core.m_expr) {

}


template<size_t N, bool Sym, typename T, typename SubCore>
bool symm3_core<N, Sym, T, SubCore>::contains(const letter &let) const {

	return m_expr.contains(let);
}


template<size_t N, bool Sym, typename T, typename SubCore>
size_t symm3_core<N, Sym, T, SubCore>::index_of(const letter &let) const
	throw(expr_exception) {

	return m_expr.index_of(let);
}


template<size_t N, bool Sym, typename T, typename SubCore>
const letter &symm3_core<N, Sym, T, SubCore>::letter_at(size_t i) const
	throw(out_of_bounds) {

	return m_expr.letter_at(i);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM3_CORE_H
