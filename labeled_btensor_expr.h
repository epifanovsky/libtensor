#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_H
#define	LIBTENSOR_LABELED_BTENSOR_EXPR_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor.h"

/**	\defgroup libtensor_btensor_expr Labeled block %tensor expressions
	\ingroup libtensor
 **/

namespace libtensor {

/**	\brief Expression using labeled block tensors
	\tparam N Tensor order
	\tparam ExprT Expression type

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename ExprT>
class labeled_btensor_expr {
public:
	typedef ExprT expression_t;

private:
	ExprT m_t;

public:

	labeled_btensor_expr(const ExprT &t) : m_t(t) {
	}

	labeled_btensor_expr(const labeled_btensor_expr<N, ExprT> &e) :
		m_t(e.m_t) {
	}
};

/**	\brief Identity expression

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Traits, typename LabelT>
class labeled_btensor_expr_ident {
public:
	typedef labeled_btensor<N, T, Traits, LabelT> labeled_btensor_t;

private:
	labeled_btensor_t &m_t;

public:

	labeled_btensor_expr_ident(labeled_btensor_t &t) : m_t(t) {
	}
};

/**	\brief Operation expression
	\tparam NArg Number of arguments

	\ingroup libtensor_btensor_expr
 **/
template<size_t NArg>
class labeled_btensor_expr_op {
};

/**	\brief Addition operation

	\ingroup libtensor_btensor_expr
 **/
template<typename T1, typename T2>
class labeled_btensor_expr_op_add {
};

/**	\brief Addition of two labeled block tensors

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename TraitsL, typename LabelL,
	typename TraitsR, typename LabelR>
void
operator+(labeled_btensor<N, T, TraitsL, LabelL> lhs,
	labeled_btensor<N, T, TraitsR, LabelR> rhs) {

}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_H

