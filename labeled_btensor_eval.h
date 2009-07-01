#ifndef LIBTENSOR_LABELED_BTENSOR_EVAL_H
#define LIBTENSOR_LABELED_BTENSOR_EVAL_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor_expr.h"
#include "btod_add.h"
#include "btod_copy.h"
#include "btod_sum.h"

namespace libtensor {

/**	\brief Evaluates an expression
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Core Expression core type.
	\tparam Label Recipient label type.

	Provides the facility to evaluate an expression. This class is
	instantiated when both the expression and the recipient are known,
	and therefore all necessary %tensor operations can be constructed.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Core, typename Label>
class labeled_btensor_eval {
public:
	//!	Expression type
	typedef labeled_btensor_expr<N, T, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, T, true, Label> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

	//!	Number of %tensor arguments in the expression
	static const size_t k_narg_tensor = eval_container_t::k_narg_tensor;

	//!	Number of %tensor operation arguments in the expression
	static const size_t k_narg_oper = eval_container_t::k_narg_oper;

private:
	template<size_t NTensor, size_t NOper>
	struct eval_tag { };

private:
	expression_t m_expr; //!< Expression
	result_t &m_result; //!< Result
	eval_container_t m_eval_container; //!< Container

public:
	//!	\name Construction and destruction
	//@{
	labeled_btensor_eval(const expression_t &expr, result_t &result);
	//@}

	//!	\name Evaluation
	//@{

	/**	\brief Evaluates the expression
	 **/
	void evaluate() throw(exception);

	//@}

private:
	/**	\brief Specialization T=double and tod_sum + tod_add
	 **/
	template<size_t NTensor, size_t NOper>
	void evaluate_case(const eval_tag<NTensor, NOper> &tag)
		throw(exception);

	/**	\brief Specialization T=double and tod_add
	 **/
	template<size_t NTensor>
	void evaluate_case(const eval_tag<NTensor, 0> &tag) throw(exception);

	/**	\brief Specialization T=double and tod_sum
	 **/
	template<size_t NOper>
	void evaluate_case(const eval_tag<0, NOper> &tag) throw(exception);

	/**	\brief Specialization T=double and tod_copy
	 **/
	//template<>
	void evaluate_case(const eval_tag<1, 0> &tag) throw(exception);

	/**	\brief Specialization T=double and direct evaluation
	 **/
	//template<>
	void evaluate_case(const eval_tag<0, 1> &tag) throw(exception);
};

template<size_t N, typename T, typename Core, typename Label>
labeled_btensor_eval<N, T, Core, Label>::labeled_btensor_eval(
	const expression_t &expr, result_t &result)
	: m_expr(expr), m_result(result), m_eval_container(m_expr, result) {

}

template<size_t N, typename T, typename Core, typename Label>
inline void labeled_btensor_eval<N, T, Core, Label>::evaluate()
	throw(exception) {

	eval_tag<k_narg_tensor, k_narg_oper> tag;
	evaluate_case(tag);
}

template<size_t N, typename T, typename Core, typename Label>
template<size_t NTensor, size_t NOper>
void labeled_btensor_eval<N, T, Core, Label>::evaluate_case(
	const eval_tag<NTensor, NOper> &tag) throw(exception) {

	// a(i|j) = c1*b1(i|j) + c2*b2(i|j) + ... + d1*fn1() + d2*fn2() + ...

	btod_add<N> coreop;

	for(size_t i = 0; i < NTensor; i++) {
		labeled_btensor_expr_arg_tensor<N, T> operand =
			m_eval_container.get_arg_tensor(i);
		coreop.add_op(operand.get_btensor(), operand.get_permutation(),
			operand.get_coeff());
	}

	btod_sum<N> op(coreop);

	for(size_t i = 0; i < NOper; i++) {
		labeled_btensor_expr_arg_oper<N, T> operand =
			m_eval_container.get_arg_oper(i);
		op.add_op(operand.get_operation(), operand.get_coeff());
	}
	op.perform(m_result.get_btensor());
}

template<size_t N, typename T, typename Core, typename Label>
template<size_t NTensor>
void labeled_btensor_eval<N, T, Core, Label>::evaluate_case(
	const eval_tag<NTensor, 0> &tag) throw(exception) {

	// a(i|j) = c1*b1(i|j) + c2*b2(i|j) + ...

	btod_add<N> op;

	for(size_t i = 0; i < NTensor; i++) {
		labeled_btensor_expr_arg_tensor<N, T> operand =
			m_eval_container.get_arg_tensor(i);
		op.add_op(operand.get_btensor(), operand.get_permutation(),
			operand.get_coeff());
	}
	op.perform(m_result.get_btensor());
}

template<size_t N, typename T, typename Core, typename Label>
template<size_t NOper>
void labeled_btensor_eval<N, T, Core, Label>::evaluate_case(
	const eval_tag<0, NOper> &tag) throw(exception) {

	// a(i|j) = c1*fn1() + c2*fn2() + ...

	btod_add<N> coreop;
	btod_sum<N> op(coreop);

	for(size_t i = 0; i < NOper; i++) {
		labeled_btensor_expr_arg_oper<N, T> operand =
			m_eval_container.get_arg_oper(i);
		op.add_op(operand.get_operation(), operand.get_coeff());
	}
	op.perform(m_result.get_btensor());
}

template<size_t N, typename T, typename Core, typename Label>
void labeled_btensor_eval<N, T, Core, Label>::evaluate_case(
	const eval_tag<1, 0> &tag) throw(exception) {

	// a(i|j) = c * b(i|j)

	labeled_btensor_expr_arg_tensor<N, T> operand =
		m_eval_container.get_arg_tensor(0);

	btod_copy<N> op(operand.get_btensor(), operand.get_permutation(),
		operand.get_coeff());
	op.perform(m_result.get_btensor());
}

template<size_t N, typename T, typename Core, typename Label>
void labeled_btensor_eval<N, T, Core, Label>::evaluate_case(
	const eval_tag<0, 1> &tag) throw(exception) {

	// a(i|j) = c * fn()

	// zero output tensor here!

	labeled_btensor_expr_arg_oper<N, T> operand =
		m_eval_container.get_arg_oper(0);
	operand.get_operation().perform(m_result.get_btensor(),
		operand.get_coeff());
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EVAL_H
