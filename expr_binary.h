#ifndef LIBTENSOR_EXPR_BINARY_H
#define LIBTENSOR_EXPR_BINARY_H

namespace libtensor {

/**	\brief Expression representing a binary operation

	\ingroup libtensor_expressions
**/
template<typename ExprL, typename ExprR, typename Op>
class expr_binary {
public:
	typedef ExprL expr_left_t; //!< Left expression type
	typedef ExprR expr_right_t; //!< Right expression type
	typedef Op operation_t; //!< Operation type

private:
	expr_left_t m_left; //<! Expression on the left
	expr_right_t m_right; //!< Expression on the right

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initialization with the right and left expressions
	**/
	expr_binary(const ExprL &expr_l, const ExprR &expr_r);

	/**	\brief Copy constructor
	**/
	expr_binary(const expr_binary<ExprL,ExprR,Op> &expr);

	//@}
};

template<typename ExprL, typename ExprR, typename Op>
inline expr_binary<ExprL,ExprR,Op>::expr_binary(const ExprL &expr_l,
	const ExprR &expr_r) : m_left(expr_l), m_right(expr_r) {
}

template<typename ExprL, typename ExprR, typename Op>
inline expr_binary<ExprL,ExprR,Op>::expr_binary(
	const expr_binary<ExprL,ExprR,Op> &expr) : m_left(expr.m_left),
	m_right(expr.m_right) {
}

} // namespace libtensor

#endif // LIBTENSOR_EXPR_BINARY_H

