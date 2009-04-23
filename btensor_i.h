#ifndef LIBTENSOR_BTENSOR_I_H
#define LIBTENSOR_BTENSOR_I_H

#include "defs.h"
#include "exception.h"
#include "index.h"
#include "symmetry_i.h"
#include "tensor_i.h"

namespace libtensor {

template<size_t N, typename T> class block_tensor_ctrl;

/**	\brief Block tensor interface

	\param N Block %tensor order.
	\param T Block %tensor element type.


	\ingroup libtensor
**/
template<size_t N, typename T>
class btensor_i : public tensor_i<N,T> {
	friend class block_tensor_ctrl<N,T>;

protected:
	virtual void on_req_symmetry(const symmetry_i<N> &sym)
		throw(exception) = 0;
	virtual tensor_i<N,T> &on_req_unique_block(const index<N> &idx)
		throw(exception) = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_I_H

