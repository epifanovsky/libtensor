#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_H
#define	LIBTENSOR_LABELED_BTENSOR_EXPR_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor.h"
#include "labeled_btensor_expr_arg.h"

/**	\defgroup libtensor_btensor_expr Labeled block %tensor expressions
	\ingroup libtensor
 **/

namespace libtensor {

template<size_t N, typename T>
class labeled_btensor_expr_base {

};

/**	\brief Expression meta-wrapper
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Core Expression core type.

	Tensor expressions make extensive use of a meta-programming technique
	call "expression templates". It allows us to store the expression
	tree as the C++ type thus transferring a number of sanity checks to
	the compilation level.

	This template wraps around the real expression type to facilitate
	the matching of overloaded operator templates.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Core>
class labeled_btensor_expr : public labeled_btensor_expr_base<N, T> {
public:
	//!	\brief Number of %tensor arguments in the expression
	static const size_t k_narg_tensor = Core::k_narg_tensor;

	//!	\brief Number of %tensor operation arguments in the expression
	static const size_t k_narg_oper = Core::k_narg_oper;

private:
	Core m_core; //!< Expression core

public:
	//!	\name Construction
	//@{

	/**	\brief Constructs the expression using a core
	 **/
	labeled_btensor_expr(const Core &core);

	/**	\brief Copy constructor
	 **/
	labeled_btensor_expr(const labeled_btensor_expr<N, T, Core> &expr);

	//@}

	//!	\name Evaluation
	//@{

	/**	\brief Evaluates the expression into an assignable labeled
			%tensor
		\tparam Label Label expression.
	 **/
	template<typename Label>
	void eval(labeled_btensor<N, T, true, Label> &t) const
		throw(exception);

	/**	\brief Returns a single %tensor argument
		\tparam Label Label expression (to figure out the %permutation)
	 **/
	template<typename Label>
	labeled_btensor_expr_arg_tensor<N, T> get_arg_tensor(size_t i) const
		throw(exception);

	//@}
};

template<size_t N, typename T, typename Core>
inline labeled_btensor_expr<N, T, Core>::labeled_btensor_expr(
	const Core &core)
	: m_core(core) {
}

template<size_t N, typename T, typename Core>
inline labeled_btensor_expr<N, T, Core>::labeled_btensor_expr(
	const labeled_btensor_expr<N, T, Core> &expr)
	: m_core(expr.m_core) {
}

template<size_t N, typename T, typename Expr>
template<typename Label>
void labeled_btensor_expr<N, T, Expr>::eval(
	labeled_btensor<N, T, true, Label> &to) const throw(exception) {

	for(size_t i = 0; i < k_narg_tensor; i++) {
		labeled_btensor_expr_arg_tensor<N, T> arg =
			get_arg_tensor<Label>(i);
	}
}

template<size_t N, typename T, typename Expr>
template<typename Label>
inline labeled_btensor_expr_arg_tensor<N, T>
labeled_btensor_expr<N, T, Expr>::get_arg_tensor(size_t i)
	const throw(exception) {
	return m_core.get_arg_tensor<Label>(i);
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_H

