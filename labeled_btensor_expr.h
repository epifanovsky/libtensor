#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_H
#define	LIBTENSOR_LABELED_BTENSOR_EXPR_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor.h"
#include "labeled_btensor_expr_arg.h"
#include "btod_copy.h"

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
	template<size_t NTensor, size_t NOper>
	struct eval_tag { };

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
		\tparam LabelLhs Label expression of the left-hand side.
	 **/
	template<typename LabelLhs>
	void eval(labeled_btensor<N, T, true, LabelLhs> &t) const
		throw(exception);

	/**	\brief Returns a single %tensor argument
		\tparam LabelLhs Label expression of the left-hand side
			(to figure out the %permutation)
	 **/
	template<typename LabelLhs>
	labeled_btensor_expr_arg_tensor<N, T> get_arg_tensor(size_t i,
		const LabelLhs &label) const throw(exception);

	//@}

private:
	/**	\brief Specialization T=double and tod_sum + tod_add
	 **/
	template<typename Label, size_t NTensor, size_t NOper>
	void eval_case(labeled_btensor<N, T, true, Label> &t,
		const eval_tag<NTensor, NOper> &tag) const throw(exception);

	/**	\brief Specialization T=double and tod_add
	 **/
	template<typename Label, size_t NTensor>
	void eval_case(labeled_btensor<N, T, true, Label> &t,
		const eval_tag<NTensor, 0> &tag) const throw(exception);

	/**	\brief Specialization T=double and tod_sum
	 **/
	template<typename Label, size_t NOper>
	void eval_case(labeled_btensor<N, T, true, Label> &t,
		const eval_tag<0, NOper> &tag) const throw(exception);

	/**	\brief Specialization T=double and tod_copy
	 **/
	template<typename Label>
	void eval_case(labeled_btensor<N, T, true, Label> &t,
		const eval_tag<1, 0> &tag) const throw(exception);

	/**	\brief Specialization T=double and direct evaluation
	 **/
	template<typename Label>
	void eval_case(labeled_btensor<N, T, true, Label> &t,
		const eval_tag<0, 1> &tag) const throw(exception);
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
	labeled_btensor<N, T, true, Label> &t) const throw(exception) {
	eval_tag<k_narg_tensor, k_narg_oper> tag;
	eval_case(t, tag);
}

template<size_t N, typename T, typename Expr>
template<typename Label, size_t NTensor, size_t NOper>
void labeled_btensor_expr<N, T, Expr>::eval_case(
	labeled_btensor<N, T, true, Label> &t,
	const eval_tag<NTensor, NOper> &tag) const throw(exception) {

	// use tod_sum + tod_add

	for(size_t i = 0; i < k_narg_tensor; i++) {
		labeled_btensor_expr_arg_tensor<N, T> arg =
			get_arg_tensor(i, t.get_label());
	}
}

template<size_t N, typename T, typename Expr>
template<typename Label, size_t NTensor>
void labeled_btensor_expr<N, T, Expr>::eval_case(
	labeled_btensor<N, T, true, Label> &t,
	const eval_tag<NTensor, 0> &tag) const throw(exception) {
	// use tod_add
}

template<size_t N, typename T, typename Expr>
template<typename Label, size_t NOper>
void labeled_btensor_expr<N, T, Expr>::eval_case(
	labeled_btensor<N, T, true, Label> &t,
	const eval_tag<0, NOper> &tag) const throw(exception) {
	// use tod_sum
}

template<size_t N, typename T, typename Expr>
template<typename Label>
void labeled_btensor_expr<N, T, Expr>::eval_case(
	labeled_btensor<N, T, true, Label> &t,
	const eval_tag<1, 0> &tag) const throw(exception) {

	// a(i|j) = c * b(i|j)

	labeled_btensor_expr_arg_tensor<N, T> src =
		get_arg_tensor(0, t.get_label());

	btod_copy<N> op(
		src.get_btensor(), src.get_permutation(), src.get_coeff());
	op.perform(t.get_btensor());
}

template<size_t N, typename T, typename Expr>
template<typename Label>
void labeled_btensor_expr<N, T, Expr>::eval_case(
	labeled_btensor<N, T, true, Label> &t,
	const eval_tag<0, 1> &tag) const throw(exception) {
	// use direct evaluation
}

template<size_t N, typename T, typename Expr>
template<typename LabelLhs>
inline labeled_btensor_expr_arg_tensor<N, T>
labeled_btensor_expr<N, T, Expr>::get_arg_tensor(
	size_t i, const LabelLhs &label) const throw(exception) {
	return m_core.get_arg_tensor(i, label);
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_H

