#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_H

#include "defs.h"
#include "exception.h"
#include "core/direct_block_tensor_operation.h"
#include "expr.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Evaluates an expression that contains both tensors and
		operations
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Core Expression core type.
	\tparam NTensor Number of tensors in the expression.
	\tparam NOper Number of operations in the expression.

	\ingroup labeled_btensor_expr
 **/
template<size_t N, typename T, typename Core, size_t NTensor, size_t NOper>
class evalfunctor {
public:
	//!	Expression type
	typedef expr<N, T, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, T, true> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

private:
	expression_t &m_expr;
	eval_container_t &m_eval_container;

public:
	evalfunctor(expression_t &expr, eval_container_t &cont);
	direct_block_tensor_operation<N, T> &get_bto();
};


} // namespace labeled_btensor_expr
} // namespace libtensor

#include "evalfunctor_double.h" // Specialization for T = double

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_H
