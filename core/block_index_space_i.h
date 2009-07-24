#ifndef LIBTENSOR_BLOCK_INDEX_SPACE_I_H
#define LIBTENSOR_BLOCK_INDEX_SPACE_I_H

#include "defs.h"
#include "exception.h"
#include "dimensions.h"

namespace libtensor {

/**	\brief Block index space interface

	\ingroup libtensor_core
 **/
template<size_t N>
class block_index_space_i {
public:
	virtual const dimensions<N> &get_dims() const = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_INDEX_SPACE_I_H
