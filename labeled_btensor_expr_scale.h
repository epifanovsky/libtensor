#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SCALE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SCALE_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor_expr_arg.h"

namespace libtensor {

/**	\brief Scales the underlying expression
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Expr Underlying expression.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Expr>
class labeled_btensor_expr_scale {
public:
	//!	\brief Number of %tensor arguments in the expression
	static const size_t k_narg_tensor = Expr::k_narg_tensor;

	//!	\brief Number of %tensor operation arguments in the expression
	static const size_t k_narg_oper = Expr::k_narg_oper;

private:
	Expr m_expr;
	T m_coeff;

public:
	//!	\name Construction
	//@{

	labeled_btensor_expr_scale(T coeff, const Expr &expr);

	//@}

	//!	\name Evaluation
	//@{

	template<typename Label>
	labeled_btensor_expr_arg_tensor<N, T> get_arg_tensor(size_t i) const
		throw(exception);

	//@}
};

template<size_t N, typename T, typename Expr>
inline labeled_btensor_expr_scale<N, T, Expr>::labeled_btensor_expr_scale(
	T coeff, const Expr &expr) : m_coeff(coeff), m_expr(expr) {
}

template<size_t N, typename T, typename Expr>
template<typename Label>
inline labeled_btensor_expr_arg_tensor<N, T>
labeled_btensor_expr_scale<N, T, Expr>::get_arg_tensor(size_t i) const
	throw(exception) {
	labeled_btensor_expr_arg_tensor<N, T> arg =
		m_expr.get_arg_tensor<Label>(i);
	arg.scale(m_coeff);
	return arg;
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SCALE_H
