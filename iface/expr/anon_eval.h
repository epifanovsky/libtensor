#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_ANON_EVAL_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_ANON_EVAL_H

#include "defs.h"
#include "exception.h"
#include "expr.h"
#include "evalfunctor.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Evaluates an expression into an anonymous %tensor
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Core Expression core type.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Core>
class anon_eval {
public:
	//!	Expression type
	typedef expr<N, T, Core> expression_t;

	//!	Output labeled block %tensor type
	typedef labeled_btensor<N, T, true> result_t;

	//!	Evaluating container type
	typedef typename expression_t::eval_container_t eval_container_t;

	//!	Number of tensor arguments
	static const size_t k_narg_tensor =
		eval_container_t::template narg<tensor_tag>::k_narg;

	//!	Number of operation arguments
	static const size_t k_narg_oper =
		eval_container_t::template narg<oper_tag>::k_narg;

	//!	Evaluation functor type
	typedef evalfunctor<N, T, Core, k_narg_tensor, k_narg_oper>
		evalfunctor_t;

private:
	expression_t m_expr; //!< Expression
	eval_container_t m_eval_container; //!< Container
	evalfunctor_t m_functor; //!< Evaluation functor
	btensor<N, T> m_bt; //!< Block %tensor
	result_t m_result; //!< Result

public:
	//!	\name Construction and destruction
	//@{
	anon_eval(const expression_t &expr, const letter_expr<N> &label);
	//@}

	//!	\name Evaluation
	//@{

	/**	\brief Evaluates the expression
	 **/
	void evaluate() throw(exception);

	/**	\brief Returns the block %tensor
	 **/
	btensor_i<N, T> &get_btensor() {
		return m_bt;
	}

	//@}

};


template<size_t N, typename T, typename Core>
anon_eval<N, T, Core>::anon_eval(
	const expression_t &expr, const letter_expr<N> &label) :
		m_expr(expr),
		m_eval_container(m_expr, label),
		m_functor(m_expr, m_eval_container),
		m_bt(m_functor.get_bto().get_bis()),
		m_result(m_bt(label)) {

}


template<size_t N, typename T, typename Core>
void anon_eval<N, T, Core>::evaluate() throw(exception) {

	m_functor.get_bto().perform(m_bt);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_ANON_EVAL_H
