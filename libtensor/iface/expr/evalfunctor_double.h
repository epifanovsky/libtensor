#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_DOUBLE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_DOUBLE_H

#include "../../defs.h"
#include "../../exception.h"
#include "../../btod/btod_add.h"
#include "../../btod/btod_additive.h"
#include "../../btod/btod_copy.h"
#include "../../btod/btod_sum.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Evaluates an expression that contains both tensors and
		operations (T = double)
	\tparam N Tensor order.
	\tparam Core Expression core type.
	\tparam NTensor Number of tensors in the expression.
	\tparam NOper Number of operations in the expression.

	An expression that consists of both individual tensors and
	sub-operations is evaluated as a sum of tensors (btod_add<N>) and
	a sum of operations (btod_sum<N>).

	\ingroup labeled_btensor_expr
 **/
template<size_t N, typename Core, size_t NTensor, size_t NOper>
class evalfunctor<N, double, Core, NTensor, NOper> {
public:
	//!	Expression type
	typedef expr<N, double, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, double, true> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

	//!	Tensor argument type
	typedef arg<N, double, tensor_tag> tensor_arg_t;

	//!	Operation argument type
	typedef arg<N, double, oper_tag> oper_arg_t;

private:
	expression_t &m_expr;
	eval_container_t &m_eval_container;
	tensor_arg_t m_arg0;
	btod_add<N> m_op_add;
	btod_sum<N> m_op_sum;

public:
	/**	\brief Initializes the functor
	 **/
	evalfunctor(expression_t &expr, eval_container_t &cont);

	/**	\brief Returns the block %tensor operation
	 **/
	btod_additive<N> &get_bto() {
		return m_op_sum;
	}

};


/**	\brief Converts an expression that consists of only tensors into a
		block %tensor operation (T = double)
	\tparam N Tensor order.
	\tparam Core Expression core type.
	\tparam NTensor Number of tensors in the expression.

	The expression is converted into a btod_add<N>, each operand being a
	%tensor from the expression with a permutation and a coefficient.

	\ingroup labeled_btensor_expr
 **/
template<size_t N, typename Core, size_t NTensor>
class evalfunctor<N, double, Core, NTensor, 0> {
public:
	//!	Expression type
	typedef expr<N, double, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, double, true> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

	//!	Tensor argument type
	typedef arg<N, double, tensor_tag> tensor_arg_t;

private:
	expression_t &m_expr;
	eval_container_t &m_eval_container;
	tensor_arg_t m_arg0;
	btod_add<N> m_op;

public:
	/**	\brief Initializes the functor
	 **/
	evalfunctor(expression_t &expr, eval_container_t &cont);

	/**	\brief Returns the block %tensor operation
	 **/
	btod_additive<N> &get_bto() {
		return m_op;
	}

};


/**	\brief Converts an expression that consists of only operations into a
		block %tensor operation (T = double)
	\tparam N Tensor order.
	\tparam Core Expression core type.
	\tparam NOper Number of operations in the expression.

	The expression is converted into a btod_sum<N>, each operand being a
	sub-operation from the expression with a coefficient.

	\ingroup labeled_btensor_expr
 **/
template<size_t N, typename Core, size_t NOper>
class evalfunctor<N, double, Core, 0, NOper> {
public:
	//!	Expression type
	typedef expr<N, double, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, double, true> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

	//!	Operation argument type
	typedef arg<N, double, oper_tag> oper_arg_t;

private:
	expression_t &m_expr;
	eval_container_t &m_eval_container;
	oper_arg_t m_arg0;
	btod_sum<N> m_op;

public:
	/**	\brief Initializes the functor
	 **/
	evalfunctor(expression_t &expr, eval_container_t &cont);

	/**	\brief Returns the block %tensor operation
	 **/
	btod_additive<N> &get_bto() {
		return m_op;
	}

};


/**	\brief Converts an expression that consists of only one %tensor
		into a copy operation (T = double)
	\tparam N Tensor order.
	\tparam Core Expression core type.

	The expression is converted into a btod_sum<N>, each operand being a
	sub-operation from the expression with a coefficient.

	\ingroup labeled_btensor_expr
 **/
template<size_t N, typename Core>
class evalfunctor<N, double, Core, 1, 0> {
public:
	//!	Expression type
	typedef expr<N, double, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, double, true> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

	//!	Tensor argument type
	typedef arg<N, double, tensor_tag> tensor_arg_t;

private:
	expression_t &m_expr;
	eval_container_t &m_eval_container;
	tensor_arg_t m_arg;
	btod_copy<N> m_op;

public:
	/**	\brief Initializes the functor
	 **/
	evalfunctor(expression_t &expr, eval_container_t &cont);

	/**	\brief Returns the block %tensor operation
	 **/
	btod_additive<N> &get_bto() {
		return m_op;
	}

};


template<size_t N, typename Core, size_t NTensor, size_t NOper>
evalfunctor<N, double, Core, NTensor, NOper>::evalfunctor(
	expression_t &expr, eval_container_t &cont) :
		m_expr(expr),
		m_eval_container(cont),
		m_arg0(m_eval_container.get_arg(tensor_tag(), 0)),
		m_op_add(m_arg0.get_btensor(), m_arg0.get_perm(),
			m_arg0.get_coeff()),
		m_op_sum(m_op_add, 1.0) {

	tensor_tag ttag;
	oper_tag otag;
	for(size_t i = 1; i < NTensor; i++) {
		tensor_arg_t arg = m_eval_container.get_arg(ttag, i);
		m_op_add.add_op(arg.get_btensor(), arg.get_perm(),
			arg.get_coeff());
	}
	for(size_t i = 0; i < NOper; i++) {
		oper_arg_t arg = m_eval_container.get_arg(otag, i);
		m_op_sum.add_op(arg.get_operation(), arg.get_coeff());
	}
}


template<size_t N, typename Core, size_t NTensor>
evalfunctor<N, double, Core, NTensor, 0>::evalfunctor(
	expression_t &expr, eval_container_t &cont) :
		m_expr(expr),
		m_eval_container(cont),
		m_arg0(m_eval_container.get_arg(tensor_tag(), 0)),
		m_op(m_arg0.get_btensor(), m_arg0.get_perm(),
			m_arg0.get_coeff()) {

	tensor_tag ttag;
	for(size_t i = 1; i < NTensor; i++) {
		tensor_arg_t arg = m_eval_container.get_arg(ttag, i);
		m_op.add_op(arg.get_btensor(), arg.get_perm(),
			arg.get_coeff());
	}
}


template<size_t N, typename Core, size_t NOper>
evalfunctor<N, double, Core, 0, NOper>::evalfunctor(
	expression_t &expr, eval_container_t &cont) :
		m_expr(expr),
		m_eval_container(cont),
		m_arg0(m_eval_container.get_arg(oper_tag(), 0)),
		m_op(m_arg0.get_operation(), m_arg0.get_coeff()) {

	oper_tag otag;
	for(size_t i = 1; i < NOper; i++) {
		oper_arg_t arg = m_eval_container.get_arg(otag, i);
		m_op.add_op(arg.get_operation(), arg.get_coeff());
	}
}


template<size_t N, typename Core>
evalfunctor<N, double, Core, 1, 0>::evalfunctor(
	expression_t &expr, eval_container_t &cont) :
		m_expr(expr),
		m_eval_container(cont),
		m_arg(m_eval_container.get_arg(tensor_tag(), 0)),
		m_op(m_arg.get_btensor(), m_arg.get_perm(), m_arg.get_coeff()) {

}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_DOUBLE_H
