#ifndef LIBTENSOR_BTENSOR_I_H
#define LIBTENSOR_BTENSOR_I_H

#include "../defs.h"
#include "../exception.h"
#include "../core/block_tensor_i.h"
#include "labeled_btensor.h"

namespace libtensor {


/**	\brief Block tensor interface
	\tparam N Block %tensor order.
	\tparam T Block %tensor element type.

	\ingroup libtensor_iface
**/
template<size_t N, typename T>
class btensor_i : public block_tensor_i<N, T> {
public:
	/**	\brief Attaches a label to this %tensor and returns it as a
			labeled %tensor
	 **/
	labeled_btensor<N, T, false> operator()(letter_expr<N> expr);
};


template<typename T>
class btensor_i<1, T> : public block_tensor_i<1, T> {
public:
	labeled_btensor<1, T, false> operator()(const letter &let);
};


template<size_t N, typename T>
inline labeled_btensor<N, T, false> btensor_i<N, T>::operator()(
	letter_expr<N> expr) {

	return labeled_btensor<N, T, false>(*this, expr);
}


template<typename T>
inline labeled_btensor<1, T, false> btensor_i<1, T>::operator()(
	const letter &let) {

	return labeled_btensor<1, T, false>(*this, letter_expr<1>(let));
}


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_I_H

