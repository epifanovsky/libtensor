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

};

template<size_t N, typename T, typename Core, typename Label,
	size_t NTensor, size_t NOper>
class labeled_btensor_eval_functor {
public:
	//!	Expression type
	typedef labeled_btensor_expr<N, T, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, T, true, Label> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

private:
	expression_t &m_expr;
	result_t &m_result;
	eval_container_t &m_eval_container;

public:
	labeled_btensor_eval_functor(expression_t &expr, result_t &res,
		eval_container_t &cont)
		: m_expr(expr), m_result(res), m_eval_container(cont) { }

	void evaluate() throw(exception);
};

template<size_t N, typename T, typename Core, typename Label,
	size_t NTensor>
class labeled_btensor_eval_functor<N, T, Core, Label, NTensor, 0> {
public:
	//!	Expression type
	typedef labeled_btensor_expr<N, T, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, T, true, Label> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

private:
	expression_t &m_expr;
	result_t &m_result;
	eval_container_t &m_eval_container;

public:
	labeled_btensor_eval_functor(expression_t &expr, result_t &res,
		eval_container_t &cont)
		: m_expr(expr), m_result(res), m_eval_container(cont) { }

	void evaluate() throw(exception);
};

template<size_t N, typename T, typename Core, typename Label, size_t NOper>
class labeled_btensor_eval_functor<N, T, Core, Label, 0, NOper> {
public:
	//!	Expression type
	typedef labeled_btensor_expr<N, T, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, T, true, Label> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

private:
	expression_t &m_expr;
	result_t &m_result;
	eval_container_t &m_eval_container;

public:
	labeled_btensor_eval_functor(expression_t &expr, result_t &res,
		eval_container_t &cont)
		: m_expr(expr), m_result(res), m_eval_container(cont) { }

	void evaluate() throw(exception);
};

template<size_t N, typename T, typename Core, typename Label>
class labeled_btensor_eval_functor<N, T, Core, Label, 1, 0> {
public:
	//!	Expression type
	typedef labeled_btensor_expr<N, T, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, T, true, Label> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

private:
	expression_t &m_expr;
	result_t &m_result;
	eval_container_t &m_eval_container;

public:
	labeled_btensor_eval_functor(expression_t &expr, result_t &res,
		eval_container_t &cont)
		: m_expr(expr), m_result(res), m_eval_container(cont) { }

	void evaluate() throw(exception);
};

template<size_t N, typename T, typename Core, typename Label>
class labeled_btensor_eval_functor<N, T, Core, Label, 0, 1> {
public:
	//!	Expression type
	typedef labeled_btensor_expr<N, T, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, T, true, Label> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

private:
	expression_t &m_expr;
	result_t &m_result;
	eval_container_t &m_eval_container;

public:
	labeled_btensor_eval_functor(expression_t &expr, result_t &res,
		eval_container_t &cont)
		: m_expr(expr), m_result(res), m_eval_container(cont) { }

	void evaluate() throw(exception);
};

template<size_t N, typename T, typename Core, typename Label>
labeled_btensor_eval<N, T, Core, Label>::labeled_btensor_eval(
	const expression_t &expr, result_t &result)
	: m_expr(expr), m_result(result), m_eval_container(m_expr, result) {

}

template<size_t N, typename T, typename Core, typename Label>
inline void labeled_btensor_eval<N, T, Core, Label>::evaluate()
	throw(exception) {

	labeled_btensor_eval_functor<N, T, Core, Label,
		k_narg_tensor, k_narg_oper>(
		m_expr, m_result, m_eval_container).evaluate();
}

template<size_t N, typename T, typename Core, typename Label, size_t NTensor,
	size_t NOper>
void labeled_btensor_eval_functor<N, T, Core, Label, NTensor, NOper>::
	evaluate() throw(exception) {

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

template<size_t N, typename T, typename Core, typename Label, size_t NTensor>
void labeled_btensor_eval_functor<N, T, Core, Label, NTensor, 0>::evaluate()
	throw(exception) {

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

template<size_t N, typename T, typename Core, typename Label, size_t NOper>
void labeled_btensor_eval_functor<N, T, Core, Label, 0, NOper>::evaluate()
	throw(exception) {

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
void labeled_btensor_eval_functor<N, T, Core, Label, 1, 0>::evaluate()
	throw(exception) {

	// a(i|j) = c * b(i|j)

	labeled_btensor_expr_arg_tensor<N, T> operand =
		m_eval_container.get_arg_tensor(0);

	btod_copy<N> op(operand.get_btensor(), operand.get_permutation(),
		operand.get_coeff());
	op.perform(m_result.get_btensor());
}

template<size_t N, typename T, typename Core, typename Label>
void labeled_btensor_eval_functor<N, T, Core, Label, 0, 1>::evaluate()
	throw(exception) {

	// a(i|j) = c * fn()

	// zero output tensor here!

	labeled_btensor_expr_arg_oper<N, T> operand =
		m_eval_container.get_arg_oper(0);
	operand.get_operation().perform(m_result.get_btensor(),
		operand.get_coeff());
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EVAL_H
