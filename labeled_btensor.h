#ifndef LIBTENSOR_LABELED_BTENSOR_H
#define	LIBTENSOR_LABELED_BTENSOR_H

#include "defs.h"
#include "exception.h"
#include "letter.h"
#include "letter_expr.h"

namespace libtensor {

template<size_t N, typename T, typename Traits> class btensor;

/**	\brief Block %tensor with an attached label
	\tparam N Tensor order.
	\tparam Traits Tensor traits.
	\tparam ExprT Label expression.

	\ingroup libtensor
 **/
template<size_t N, typename T, typename Traits, typename ExprT>
class labeled_btensor {
private:
	typedef T element_t;
	typedef typename Traits::allocator_t allocator_t;

private:
	btensor<N,T,Traits> &m_t;
	letter_expr<N,ExprT> m_expr;

public:
	labeled_btensor(btensor<N,T,Traits> &t, const letter_expr<N,ExprT> expr);
};

template<size_t N, typename T, typename Traits, typename ExprT>
inline labeled_btensor<N,T,Traits,ExprT>::labeled_btensor(btensor<N,T,Traits> &t,
	const letter_expr<N,ExprT> expr) : m_t(t), m_expr(expr) {
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_H

