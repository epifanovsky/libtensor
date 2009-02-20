#ifndef __LIBTENSOR_BLOCK_TENSOR_I_H
#define __LIBTENSOR_BLOCK_TENSOR_I_H

#include "defs.h"
#include "exception.h"
#include "tensor_i.h"
#include "block_tensor_operation_handler.h"

namespace libtensor {

/**	\brief Block tensor interface


	\ingroup libtensor
**/
template<typename T>
class block_tensor_i : public tensor_i<T> {
protected:
	virtual block_tensor_operation_handler<T>
		&get_block_tensor_operation_handler() = 0;
};

} // namespace libtensor

#endif // __LIBTENSOR_BLOCK_TENSOR_I_H

