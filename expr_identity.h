#ifndef LIBTENSOR_EXPR_IDENTITY_H
#define LIBTENSOR_EXPR_IDENTITY_H

#include <cstdio>
#include <typeinfo>
#include "defs.h"
#include "exception.h"

namespace libtensor {

/**	\brief Expression representing a complex object such as tensor

	\ingroup libtensor_expressions
**/
template<typename T>
class expr_identity {
private:
	const T &m_expr; //!< Expression

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the expression with an object
	**/
	expr_identity(const T &expr);

	/**	\brief Copy constructor
	**/
	expr_identity(const expr_identity<T> &expr);

	//@}

	/**	\brief Evaluates the expression
	**/
	void eval(T &t) throw(exception);
};

template<typename T>
expr_identity<T>::expr_identity(const T &expr) : m_expr(expr) {
}

template<typename T>
expr_identity<T>::expr_identity(const expr_identity<T> &expr) :
	m_expr(expr.m_expr) {
}

template<typename T>
void expr_identity<T>::eval(T &t) throw(exception) {
	char s[1024];
	snprintf(s, 1024, "Evaluation not specified for T=%s",
		typeid(t).name());
	throw_exc("expr_identity<T>", "eval(T &t)", s);
}

} // namespace libtensor

#endif // LIBTENSOR_EXPR_IDENTITY_H

