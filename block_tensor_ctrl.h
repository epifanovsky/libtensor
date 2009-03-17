#ifndef LIBTENSOR_BLOCK_TENSOR_CTRL_H
#define LIBTENSOR_BLOCK_TENSOR_CTRL_H

#include "defs.h"
#include "exception.h"
#include "block_tensor_i.h"
#include "index.h"
#include "symmetry_i.h"

namespace libtensor {

/**	\brief Block %tensor control

	\ingroup libtensor
**/
template<typename T>
class block_tensor_ctrl {
public:
	//!	\name Construction and destruction
	//@{
	block_tensor_ctrl(block_tensor_i<T> &bt);
	~block_tensor_ctrl();
	//@}

	//!	\name Event forwarding
	//@{
	void req_symmetry(const symmetry_i &sym) throw(exception);
	tensor_i<T> &req_block(const index &idx) throw(exception);
	//@}
};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_CTRL_H

