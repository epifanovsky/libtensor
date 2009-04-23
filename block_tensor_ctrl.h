#ifndef LIBTENSOR_BLOCK_TENSOR_CTRL_H
#define LIBTENSOR_BLOCK_TENSOR_CTRL_H

#include "defs.h"
#include "exception.h"
#include "btensor_i.h"
#include "index.h"
#include "symmetry_i.h"

namespace libtensor {

/**	\brief Block %tensor control

	\ingroup libtensor
**/
template<size_t N, typename T>
class block_tensor_ctrl {
public:
	//!	\name Construction and destruction
	//@{
	block_tensor_ctrl(btensor_i<N,T> &bt);
	~block_tensor_ctrl();
	//@}

	//!	\name Event forwarding
	//@{
	void req_symmetry(const symmetry_i<N> &sym) throw(exception);
	tensor_i<N,T> &req_block(const index<N> &idx) throw(exception);
	//@}
};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_CTRL_H

