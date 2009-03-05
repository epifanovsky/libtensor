#ifndef LIBTENSOR_DIRECT_TENSOR_ADDITIVE_OPERATION_H
#define LIBTENSOR_DIRECT_TENSOR_ADDITIVE_OPERATION_H

#include "defs.h"
#include "exception.h"
#include "direct_tensor_operation.h"

namespace libtensor {

/**	\brief Direct %tensor operation that adds its result to a %tensor

	\ingroup libtensor
**/
template<typename T>
class direct_tensor_additive_operation : public direct_tensor_operation<T> {
public:
	virtual void perform(tensor_i<T> &t, const double c)
		throw(exception) = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_DIRECT_TENSOR_ADDITIVE_OPERATION_H

