#ifndef LIBTENSOR_BTENSOR_I_H
#define LIBTENSOR_BTENSOR_I_H

#include "defs.h"
#include "exception.h"
#include "block_tensor_i.h"
#include "labeled_btensor.h"

namespace libtensor {

/**	\brief Block tensor interface
	\tparam N Block %tensor order.
	\tparam T Block %tensor element type.

	\ingroup libtensor
**/
template<size_t N, typename T>
class btensor_i : public block_tensor_i<N, T> {
public:
	/**	\brief Attaches a label to this %tensor and returns it as a
			labeled %tensor
	 **/
	template<typename ExprT>
	labeled_btensor<N, T, false, letter_expr<N, ExprT> > operator()(
		letter_expr<N, ExprT> expr);
};

template<size_t N, typename T> template<typename ExprT>
labeled_btensor<N, T, false, letter_expr<N, ExprT> >
btensor_i<N, T>::operator()(letter_expr<N, ExprT> expr) {
	return labeled_btensor<N, T, false, letter_expr<N, ExprT> >(
		*this, expr);
}

} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_I_H

