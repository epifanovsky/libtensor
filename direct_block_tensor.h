#ifndef LIBTENSOR_DIRECT_BLOCK_TENSOR_H
#define LIBTENSOR_DIRECT_BLOCK_TENSOR_H

#include "defs.h"
#include "exception.h"

namespace libtensor {

/**	\brief Direct block %tensor

	\ingroup libtensor
**/
template<typename T, typename Alloc>
class direct_block_tensor : public block_tensor_i<T> {
};

} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BLOCK_TENSOR_H

