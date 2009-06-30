#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor_expr.h"
#include "labeled_btensor_expr_arg.h"
#include "letter_expr.h"

namespace libtensor {

/**	\brief Addition operation core expression
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam ExprL LHS expression.
	\tparam ExprR RHS expression.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename ExprL, typename ExprR>
class labeled_btensor_expr_add {
public:
	typedef labeled_btensor_expr<N, T, ExprL> exprl_t;
	typedef labeled_btensor_expr<N, T, ExprR> exprr_t;

public:
	static const size_t k_narg_tensor =
		ExprL::k_narg_tensor + ExprR::k_narg_tensor;
	static const size_t k_narg_oper =
		ExprL::k_narg_oper + ExprR::k_narg_oper;

private:
	exprl_t m_exprl; //!< Left expression
	exprr_t m_exprr; //!< Right expression

public:
	//!	\name Construction
	//@{

	/**	\brief Initializes the core with left and right expressions
	 **/
	labeled_btensor_expr_add(const exprl_t &exprl,
		const exprr_t &exprr);

	//@}

	//!	\name Evaluation
	//@{

	template<typename LabelLhs>
	labeled_btensor_expr_arg_tensor<N, T> get_arg_tensor(
		size_t i, const letter_expr<N, LabelLhs> &label_lhs) const
		throw(exception);

	//@}
};

template<size_t N, typename T, typename ExprL, typename ExprR>
inline labeled_btensor_expr_add<N, T, ExprL, ExprR>::labeled_btensor_expr_add(
	const exprl_t &exprl, const exprr_t &exprr)
	: m_exprl(exprl), m_exprr(exprr) {
}

template<size_t N, typename T, typename ExprL, typename ExprR>
template<typename LabelLhs>
inline labeled_btensor_expr_arg_tensor<N, T>
labeled_btensor_expr_add<N, T, ExprL, ExprR>::get_arg_tensor(
	size_t i, const letter_expr<N, LabelLhs> &label_lhs) const
	throw(exception) {
	if(ExprL::k_narg_tensor > 0 && ExprL::k_narg_tensor > i)
		return m_exprl.get_arg_tensor(i, label_lhs);
	size_t j = i - ExprL::k_narg_tensor;
	if(ExprR::k_narg_tensor > 0 && ExprR::k_narg_tensor > j)
		return m_exprr.get_arg_tensor(j, label_lhs);
	throw_exc("labeled_btensor_expr_add<N, T, ExprL, ExprR>",
		"get_arg_tensor(size_t)", "Inconsistent expression");
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_H
