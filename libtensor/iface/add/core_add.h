#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_ADD_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_ADD_H

#include "../../defs.h"
#include "../../exception.h"
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, typename T, typename ExprL, typename ExprR> class eval_add;


/**	\brief Addition operation expression core
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam ExprL LHS expression type.
	\tparam ExprR RHS expression type.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename ExprL, typename ExprR>
class core_add {
public:
	static const char *k_clazz; //!< Class name

public:
	//!	Evaluating container type
	typedef eval_add<N, T, ExprL, ExprR> eval_container_t;

private:
	ExprL m_expr_l; //!< Left expression
	ExprR m_expr_r; //!< Right expression

public:
	//!	\name Construction
	//@{

	/**	\brief Initializes the core with left and right expressions
	 **/
	core_add(const ExprL &expr_l, const ExprR &expr_r)
		: m_expr_l(expr_l), m_expr_r(expr_r) { }

	//@}

	/**	\brief Returns the left expression
	 **/
	ExprL &get_expr_l() { return m_expr_l; }

	/**	\brief Returns the right expression
	 **/
	ExprR &get_expr_r() { return m_expr_r; }

	/**	\brief Returns whether the %tensor's label contains a %letter
	 **/
	bool contains(const letter &let) const;

	/**	\brief Returns the %index of a %letter in the %tensor's label
	 **/
	size_t index_of(const letter &let) const throw(exception);

	/**	\brief Returns the %letter at a given position in
			the %tensor's label
	 **/
	const letter &letter_at(size_t i) const throw(exception);

};


template<size_t N, typename T, typename ExprL, typename ExprR>
const char *core_add<N, T, ExprL, ExprR>::k_clazz =
	"core_add<N, T, ExprL, ExprR>";


template<size_t N, typename T, typename ExprL, typename ExprR>
inline bool core_add<N, T, ExprL, ExprR>::contains(const letter &let) const {

	return m_expr_l.contains(let);
}


template<size_t N, typename T, typename ExprL, typename ExprR>
inline size_t core_add<N, T, ExprL, ExprR>::index_of(const letter &let) const
	throw(exception) {

	return m_expr_l.index_of(let);
}


template<size_t N, typename T, typename ExprL, typename ExprR>
inline const letter &core_add<N, T, ExprL, ExprR>::letter_at(size_t i) const
	throw(exception) {

	return m_expr_l.letter_at(i);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_ADD_H
