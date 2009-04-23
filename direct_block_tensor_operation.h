#ifndef LIBTENSOR_DIRECT_BLOCK_TENSOR_OPERATION_H
#define LIBTENSOR_DIRECT_BLOCK_TENSOR_OPERATION_H

#include "defs.h"
#include "exception.h"
#include "btensor_i.h"

namespace libtensor {

/**	\brief Underlying operation for a direct block %tensor

	\ingroup libtensor
**/
template<size_t N, typename T>
class direct_block_tensor_operation {
public:
	/**	\brief Invoked to execute the operation
	**/
	virtual void perform(btensor_i<N,T> &bt) throw(exception) = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BLOCK_TENSOR_OPERATION_H

