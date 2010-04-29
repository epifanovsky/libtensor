#ifndef LIBTENSOR_TOD_ADDITIVE_H
#define LIBTENSOR_TOD_ADDITIVE_H

#include "../defs.h"
#include "../exception.h"
#include "../core/tensor_i.h"

namespace libtensor {

/**	\brief Additive %tensor operation (double)

	A %tensor operation that operates on the double precision element data
	type should implement this interface if its sole result is a %tensor
	and it can add it an existing %tensor without allocating a buffer.

	The two perform() methods must render the same operation, with the
	only difference that one of them adds the result to a %tensor.

	\ingroup libtensor
**/
template<size_t N>
class tod_additive {
public:
	/**	\brief Prefetches the operands
	 **/
	virtual void prefetch() = 0;

	/**	\brief Performs the operation and writes the result to a %tensor
		\param t Tensor.
		\param c Coefficient.
	 **/
	virtual void perform(tensor_i<N, double> &t) = 0;

	/**	\brief Performs the operation and adds its result to a %tensor
			with a coefficient
		\param t Tensor.
		\param c Coefficient.
	 **/
	virtual void perform(tensor_i<N, double> &t, double c) = 0;

};

}

#endif // LIBTENSOR_TOD_ADDITIVE_H

