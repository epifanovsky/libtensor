#ifndef LIBTENSOR_BLOCK_TENSOR_H
#define LIBTENSOR_BLOCK_TENSOR_H

#include "defs.h"
#include "exception.h"
#include "block_info_i.h"
#include "block_tensor_i.h"

namespace libtensor {

/**	\brief Block %tensor

	<b>Request to lower symmetry (req_lower_symmetry)</b>

	Lowers the permutational symmetry of the block tensor to the requested
	or lower, if necessary.

	\ingroup libtensor
**/
template<typename T>
class block_tensor : public block_tensor_i<T> {
public:
	//!	\name Construction and destruction
	//@{
	/**	\brief Constructs a block %tensor using provided information
			about blocks
		\param bi Information about blocks
	**/
	block_tensor(const block_info_i &bi);

	/**	\brief Constructs a block %tensor using information about
			blocks from another block %tensor
		\param bt Another block %tensor
	**/
	block_tensor(const block_tensor_i<T> &bt);

	/**	\brief Virtual destructor
	**/
	virtual ~block_tensor();
	//@}
};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_H

