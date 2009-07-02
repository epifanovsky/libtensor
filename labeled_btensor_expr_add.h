#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor_expr.h"
#include "labeled_btensor_expr_arg.h"
#include "letter_expr.h"

namespace libtensor {

template<size_t N, typename T, typename CoreL, typename CoreR>
class labeled_btensor_eval_add;

/**	\brief Addition operation core expression
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam ExprL LHS expression type (labeled_btensor_expr).
	\tparam ExprR RHS expression type (labeled_btensor_expr).

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename ExprL, typename ExprR>
class labeled_btensor_expr_add {
public:
	//!	Evaluating container type
	typedef labeled_btensor_eval_add<N, T, ExprL, ExprR> eval_container_t;

private:
	ExprL m_expr_l; //!< Left expression
	ExprR m_expr_r; //!< Right expression

public:
	//!	\name Construction
	//@{

	/**	\brief Initializes the core with left and right expressions
	 **/
	labeled_btensor_expr_add(const ExprL &expr_l, const ExprR &expr_r)
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

/**	\brief Evaluates the addition expression

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename ExprL, typename ExprR>
class labeled_btensor_eval_add {
public:
	//!	Addition expression core type
	typedef labeled_btensor_expr_add<N, T, ExprL, ExprR> core_t;

	//!	Addition expression type
	typedef labeled_btensor_expr<N, T, core_t> expression_t;

	//!	Evaluating container type for the left expression
	typedef typename ExprL::eval_container_t eval_container_l_t;

	//!	Evaluating container type for the right expression
	typedef typename ExprR::eval_container_t eval_container_r_t;

	//!	Number of %tensor arguments in the expression
	static const size_t k_narg_tensor =
		eval_container_l_t::k_narg_tensor +
		eval_container_r_t::k_narg_tensor;

	//!	Number of %tensor operation arguments in the expression
	static const size_t k_narg_oper =
		eval_container_l_t::k_narg_oper +
		eval_container_r_t::k_narg_oper;

private:
	expression_t &m_expr; //!< Addition expression
	eval_container_l_t m_cont_l; //!< Left evaluating container
	eval_container_r_t m_cont_r; //!< Right evaluating container

public:
	template<typename LabelLhs>
	labeled_btensor_eval_add(expression_t &expr,
		labeled_btensor<N, T, true, LabelLhs> &result) throw(exception);

	//!	\name Evaluation
	//@{

	labeled_btensor_expr_arg_tensor<N, T> get_arg_tensor(size_t i) const
		throw(exception);

	labeled_btensor_expr_arg_oper<N, T> get_arg_oper(size_t i) const
		throw(exception);

	//@}
};

template<size_t N, typename T, typename ExprL, typename ExprR>
inline bool labeled_btensor_expr_add<N, T, ExprL, ExprR>::contains(
	const letter &let) const {

	return m_expr_l.contains(let);
}

template<size_t N, typename T, typename ExprL, typename ExprR>
inline size_t labeled_btensor_expr_add<N, T, ExprL, ExprR>::index_of(
	const letter &let) const throw(exception) {

	return m_expr_l.index_of(let);
}

template<size_t N, typename T, typename ExprL, typename ExprR>
inline const letter &labeled_btensor_expr_add<N, T, ExprL, ExprR>::letter_at(
	size_t i) const throw(exception) {

	return m_expr_l.letter_at(i);
}

template<size_t N, typename T, typename ExprL, typename ExprR>
template<typename LabelLhs>
labeled_btensor_eval_add<N, T, ExprL, ExprR>::labeled_btensor_eval_add(
	expression_t &expr, labeled_btensor<N, T, true, LabelLhs> &result)
	throw(exception)
	: m_expr(expr), m_cont_l(expr.get_core().get_expr_l(), result),
	m_cont_r(expr.get_core().get_expr_r(), result) {

}

template<size_t N, typename T, typename ExprL, typename ExprR>
labeled_btensor_expr_arg_tensor<N, T>
labeled_btensor_eval_add<N, T, ExprL, ExprR>::get_arg_tensor(size_t i) const
	throw(exception) {

	if(eval_container_l_t::k_narg_tensor > 0 &&
		eval_container_l_t::k_narg_tensor > i)
		return m_cont_l.get_arg_tensor(i);

	size_t j = i - eval_container_l_t::k_narg_tensor;
	if(eval_container_r_t::k_narg_tensor > 0 &&
		eval_container_r_t::k_narg_tensor > j)
		return m_cont_r.get_arg_tensor(j);

	throw_exc("labeled_btensor_eval_add<N, T, ExprL, ExprR>",
		"get_arg_tensor(size_t)", "Inconsistent expression");
}

template<size_t N, typename T, typename ExprL, typename ExprR>
labeled_btensor_expr_arg_oper<N, T>
labeled_btensor_eval_add<N, T, ExprL, ExprR>::get_arg_oper(size_t i) const
	throw(exception) {

	if(eval_container_l_t::k_narg_oper > 0 &&
		eval_container_l_t::k_narg_oper > i)
		return m_cont_l.get_arg_oper(i);

	size_t j = i - eval_container_l_t::k_narg_oper;
	if(eval_container_l_t::k_narg_oper > 0 &&
		eval_container_l_t::k_narg_oper > j)
		return m_cont_r.get_arg_oper(j);

	throw_exc("labeled_btensor_eval_add<N, T, ExprL, ExprR>",
		"get_arg_oper(size_t)", "Inconsistent expression");
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_H
