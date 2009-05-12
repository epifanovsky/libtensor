#ifndef LIBTENSOR_DIRECT_BLOCK_TENSOR_H
#define LIBTENSOR_DIRECT_BLOCK_TENSOR_H

#include "defs.h"
#include "exception.h"
#include "block_tensor_i.h"

namespace libtensor {

/**	\brief Direct block %tensor

	\ingroup libtensor
**/
template<size_t N, typename T, typename Alloc>
class direct_block_tensor : public block_tensor_i<N, T> {
};

} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BLOCK_TENSOR_H

