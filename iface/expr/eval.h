#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_H

#include "defs.h"
#include "exception.h"
#include "expr.h"
#include "evalfunctor.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Evaluates an expression into a %tensor
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Core Expression core type.

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


template<size_t N, typename T, typename Core>
eval<N, T, Core>::eval(const expression_t &expr, result_t &result) try :
	m_expr(expr),
	m_result(result),
	m_eval_container(m_expr, m_result.get_label()) {
} catch(...) {

	std::cout << " eval[";
	for(size_t i = 0; i < N; i++) {
		if(i != 0) std::cout << " ";
		std::cout << &m_expr.letter_at(i);
	}
	std::cout << "] ";

}


template<size_t N, typename T, typename Core>
inline void eval<N, T, Core>::evaluate() throw(exception) {

	const size_t narg_tensor =
		eval_container_t::template narg<tensor_tag>::k_narg;
	const size_t narg_oper =
		eval_container_t::template narg<oper_tag>::k_narg;

	m_eval_container.prepare();
	evalfunctor<N, T, Core, narg_tensor, narg_oper>(
		m_expr, m_eval_container).get_bto().perform(
			m_result.get_btensor());
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_H
