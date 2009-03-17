#ifndef LIBTENSOR_BLOCK_TENSOR_I_H
#define LIBTENSOR_BLOCK_TENSOR_I_H

#include "defs.h"
#include "exception.h"
#include "index.h"
#include "symmetry_i.h"
#include "tensor_i.h"

namespace libtensor {

template<typename T> class block_tensor_ctrl;

/**	\brief Block tensor interface


	\ingroup libtensor
**/
template<typename T>
class block_tensor_i : public tensor_i<T> {
	friend class block_tensor_ctrl<T>;

protected:
	virtual void on_req_symmetry(const symmetry_i &sym)
		throw(exception) = 0;
	virtual tensor_i<T> &on_req_unique_block(const index &idx)
		throw(exception) = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_I_H

