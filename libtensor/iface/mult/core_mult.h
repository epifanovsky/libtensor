#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_MULT_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_MULT_H

#include "../../defs.h"
#include "../../exception.h"
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, typename T, typename E1, typename E2, bool Recip>
class eval_mult;


/**	\brief Element-wise multiplication operation expression core
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam E1 LHS expression type.
	\tparam E2 RHS expression type.
	\tparam Recip If true do element-wise division instead of multiplication.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename E1, typename E2, bool Recip>
class core_mult {
public:
	static const char *k_clazz; //!< Class name

public:
	//!	Evaluating container type
	typedef eval_mult<N, T, E1, E2, Recip> eval_container_t;

private:
	E1 m_expr1; //!< Left expression
	E2 m_expr2; //!< Right expression

public:
	//!	\name Construction
	//@{

	/**	\brief Initializes the core with left and right expressions
	 **/
	core_mult(const E1 &expr1, const E2 &expr2)
		: m_expr1(expr1), m_expr2(expr2) { }

	//@}

	/**	\brief Returns the first expression
	 **/
	E1 &get_expr_1() { return m_expr1; }

	/**	\brief Returns the second expression
	 **/
	E2 &get_expr_2() { return m_expr2; }

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


template<size_t N, typename T, typename E1, typename E2, bool Recip>
const char *core_mult<N, T, E1, E2, Recip>::k_clazz =
		"core_mult<N, T, E1, E2, Recip>";


template<size_t N, typename T, typename E1, typename E2, bool Recip>
inline bool core_mult<N, T, E1, E2, Recip>::contains(const letter &let) const {

	return m_expr1.contains(let);
}


template<size_t N, typename T, typename E1, typename E2, bool Recip>
inline size_t core_mult<N, T, E1, E2, Recip>::index_of(const letter &let) const
	throw(exception) {

	return m_expr1.index_of(let);
}


template<size_t N, typename T, typename E1, typename E2, bool Recip>
inline const letter &core_mult<N, T, E1, E2, Recip>::letter_at(size_t i) const
	throw(exception) {

	return m_expr1.letter_at(i);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_MULT_H
