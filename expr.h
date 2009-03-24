#ifndef LIBTENSOR_EXPR_H
#define LIBTENSOR_EXPR_H

#include "defs.h"
#include "exception.h"

namespace libtensor {

template<typename ExprT>
class expr {
private:
	ExprT m_expr; //!< Expression

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Constructs an expression from, well, an expression
	**/
	expr(const ExprT &expr);

	/**	\brief Copy constructor
	**/
	expr(const expr<ExprT> &expr);

	//@}

	/**	\brief Evaluates the expression
	**/
	template<typename T>
	void eval(T &t) throw(exception);

};

template<typename ExprT>
inline expr<ExprT>::expr(const ExprT &expr) : m_expr(expr) {
}

template<typename ExprT>
inline expr<ExprT>::expr(const expr<ExprT> &expr) : m_expr(expr.m_expr) {
}

template<typename ExprT> template<typename T>
inline void expr<ExprT>::eval(T &t) throw(exception) {
	m_expr.eval(t);
}

} // namespace libtensor

#endif // LIBTENSOR_EXPR_H

