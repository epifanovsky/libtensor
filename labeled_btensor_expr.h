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
	//!	Expression evaluating container type
	typedef typename Core::eval_container_t eval_container_t;

private:
	Core m_core; //!< Expression core

public:
	/**	\brief Constructs the expression using a core
	 **/
	labeled_btensor_expr(const Core &core) : m_core(core) { }

	/**	\brief Copy constructor
	 **/
	labeled_btensor_expr(const labeled_btensor_expr<N, T, Core> &expr)
	: m_core(expr.m_core) { }

	/**	\brief Returns the core of the expression
	 **/
	Core &get_core() { return m_core; }

};

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_H

