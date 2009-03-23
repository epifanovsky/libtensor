#ifndef LIBTENSOR_EXPR_LITERAL_H
#define LIBTENSOR_EXPR_LITERAL_H

namespace libtensor {

/**	\brief Expression representing a literal (a constant in an expression)

	\ingroup libtensor_expressions
**/
template<typename T>
class expr_literal {
private:
	T m_expr; //!< Expression

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the expression with a value
	**/
	expr_literal(const T &expr);

	/**	\brief Copy constructor
	**/
	expr_literal(const expr_literal<T> &expr);

	//@}
};

template<typename T>
inline expr_literal<T>::expr_literal(const T &expr) : m_expr(expr) {
}

template<typename T>
inline expr_literal<T>::expr_literal(const expr_literal<T> &expr) :
	m_expr(expr.m_expr) {
}

} // namespace libtensor

#endif // LIBTENSOR_EXPR_LITERAL_H

