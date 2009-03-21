#ifndef LIBTENSOR_BLOCK_INFO_I_H
#define LIBTENSOR_BLOCK_INFO_I_H

#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "index.h"
#include "symmetry_i.h"

namespace libtensor {

/**	\brief Provides information on how to do block splitting

	\param N Block %tensor order

	\ingroup libtensor
**/
template<size_t N>
class block_info_i {
public:
	/**	\brief Returns the total %dimensions
	**/
	virtual const dimensions<N> &get_dims() const = 0;

	virtual const dimensions<N> &get_block_dims(const index<N> &i)
		const = 0;

	virtual const symmetry_i<N> &get_symmetry() const = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_INFO_I_H

