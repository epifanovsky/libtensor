#ifndef LIBTENSOR_LABELED_BTENSOR_EVAL_H
#define LIBTENSOR_LABELED_BTENSOR_EVAL_H

#include "defs.h"
#include "exception.h"
#include "btod/btod_add.h"
#include "btod/btod_copy.h"
#include "btod/btod_sum.h"
#include "labeled_btensor_expr.h"

namespace libtensor {

namespace labeled_btensor_expr {

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
template<size_t N, typename T, typename Core>
class eval {
public:
	//!	Expression type
	typedef expr<N, T, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, T, true> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

private:
	expression_t m_expr; //!< Expression
	result_t &m_result; //!< Result
	eval_container_t m_eval_container; //!< Container

public:
	//!	\name Construction and destruction
	//@{
	eval(const expression_t &expr, result_t &result);
	//@}

	//!	\name Evaluation
	//@{

	/**	\brief Evaluates the expression
	 **/
	void evaluate() throw(exception);

	//@}

};

template<size_t N, typename T, typename Core, size_t NTensor, size_t NOper>
class eval_functor {
public:
	//!	Expression type
	typedef expr<N, T, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, T, true> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

private:
	expression_t &m_expr;
	result_t &m_result;
	eval_container_t &m_eval_container;

public:
	eval_functor(expression_t &expr, result_t &res, eval_container_t &cont)
		: m_expr(expr), m_result(res), m_eval_container(cont) { }

	void evaluate() throw(exception);
};

template<size_t N, typename T, typename Core, size_t NTensor>
class eval_functor<N, T, Core, NTensor, 0> {
public:
	//!	Expression type
	typedef expr<N, T, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, T, true> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

private:
	expression_t &m_expr;
	result_t &m_result;
	eval_container_t &m_eval_container;

public:
	eval_functor(expression_t &expr, result_t &res, eval_container_t &cont)
		: m_expr(expr), m_result(res), m_eval_container(cont) { }

	void evaluate() throw(exception);
};

template<size_t N, typename T, typename Core, size_t NOper>
class eval_functor<N, T, Core, 0, NOper> {
public:
	//!	Expression type
	typedef expr<N, T, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, T, true> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

private:
	expression_t &m_expr;
	result_t &m_result;
	eval_container_t &m_eval_container;

public:
	eval_functor(expression_t &expr, result_t &res, eval_container_t &cont)
		: m_expr(expr), m_result(res), m_eval_container(cont) { }

	void evaluate() throw(exception);
};

template<size_t N, typename T, typename Core>
class eval_functor<N, T, Core, 1, 0> {
public:
	//!	Expression type
	typedef expr<N, T, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, T, true> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

private:
	expression_t &m_expr;
	result_t &m_result;
	eval_container_t &m_eval_container;

public:
	eval_functor(expression_t &expr, result_t &res, eval_container_t &cont)
		: m_expr(expr), m_result(res), m_eval_container(cont) { }

	void evaluate() throw(exception);
};

template<size_t N, typename T, typename Core>
class eval_functor<N, T, Core, 0, 1> {
public:
	//!	Expression type
	typedef expr<N, T, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, T, true> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

private:
	expression_t &m_expr;
	result_t &m_result;
	eval_container_t &m_eval_container;

public:
	eval_functor(expression_t &expr, result_t &res, eval_container_t &cont)
		: m_expr(expr), m_result(res), m_eval_container(cont) { }

	void evaluate() throw(exception);
};

template<size_t N, typename T, typename Core>
eval<N, T, Core>::eval(const expression_t &expr, result_t &result) :
	m_expr(expr),
	m_result(result),
	m_eval_container(m_expr, result) {

}

template<size_t N, typename T, typename Core>
inline void eval<N, T, Core>::evaluate() throw(exception) {

	const size_t narg_tensor =
		eval_container_t::template narg<tensor_tag>::k_narg;
	const size_t narg_oper =
		eval_container_t::template narg<oper_tag>::k_narg;

	eval_functor<N, T, Core, narg_tensor, narg_oper>(
		m_expr, m_result, m_eval_container).evaluate();
}

template<size_t N, typename T, typename Core, size_t NTensor, size_t NOper>
void eval_functor<N, T, Core, NTensor, NOper>::evaluate() throw(exception) {

	// a(i|j) = c1*b1(i|j) + c2*b2(i|j) + ... + d1*fn1() + d2*fn2() + ...

	typedef arg<N, T, tensor_tag> tensor_arg_t;
	typedef arg<N, T, oper_tag> oper_arg_t;

	tensor_tag ttag;
	oper_tag otag;

	tensor_arg_t operand0 = m_eval_container.get_arg(ttag, 0);
	btod_add<N> coreop(operand0.get_btensor(), operand0.get_perm(),
			operand0.get_coeff());

	for(size_t i = 1; i < NTensor; i++) {
		tensor_arg_t operand = m_eval_container.get_arg(ttag, i);
		coreop.add_op(operand.get_btensor(), operand.get_perm(),
			operand.get_coeff());
	}

	btod_sum<N> op(coreop);

	for(size_t i = 0; i < NOper; i++) {
		oper_arg_t operand = m_eval_container.get_arg(otag, i);
		op.add_op(operand.get_operation(), operand.get_coeff());
	}
	op.perform(m_result.get_btensor());
}

template<size_t N, typename T, typename Core, size_t NTensor>
void eval_functor<N, T, Core, NTensor, 0>::evaluate() throw(exception) {

	// a(i|j) = c1*b1(i|j) + c2*b2(i|j) + ...

	typedef arg<N, T, tensor_tag> tensor_arg_t;

	tensor_tag ttag;

	tensor_arg_t operand0 = m_eval_container.get_arg(ttag, 0);
	btod_add<N> op(operand0.get_btensor(), operand0.get_perm(),
		operand0.get_coeff());

	for(size_t i = 1; i < NTensor; i++) {
		tensor_arg_t operand = m_eval_container.get_arg(ttag, i);
		op.add_op(operand.get_btensor(), operand.get_perm(),
			operand.get_coeff());
	}
	op.perform(m_result.get_btensor());
}

template<size_t N, typename T, typename Core, size_t NOper>
void eval_functor<N, T, Core, 0, NOper>::evaluate() throw(exception) {

	// a(i|j) = c1*fn1() + c2*fn2() + ...
/*
	btod_add<N> coreop;
	btod_sum<N> op(coreop);

	for(size_t i = 0; i < NOper; i++) {
		labeled_btensor_expr_arg_oper<N, T> operand =
			m_eval_container.get_arg_oper(i);
		op.add_op(operand.get_operation(), operand.get_coeff());
	}
	op.perform(m_result.get_btensor());*/
}

template<size_t N, typename T, typename Core>
void eval_functor<N, T, Core, 1, 0>::evaluate() throw(exception) {

	// a(i|j) = c * b(i|j)

	typedef arg<N, T, tensor_tag> tensor_arg_t;

	tensor_tag ttag;

	tensor_arg_t operand = m_eval_container.get_arg(ttag, 0);

	btod_copy<N> op(operand.get_btensor(), operand.get_perm(),
		operand.get_coeff());
	op.perform(m_result.get_btensor());
}

template<size_t N, typename T, typename Core>
void eval_functor<N, T, Core, 0, 1>::evaluate() throw(exception) {

	// a(i|j) = c * fn()

	// zero output tensor here!

	typedef arg<N, T, oper_tag> oper_arg_t;

	oper_tag otag;

	oper_arg_t operand = m_eval_container.get_arg(otag, 0);
	operand.get_operation().perform(m_result.get_btensor(),
		operand.get_coeff());
}

} // namespace labeled_btensor_expr

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EVAL_H
