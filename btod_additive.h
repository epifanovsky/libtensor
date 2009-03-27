#ifndef LIBTENSOR_BTOD_ADDITIVE_H
#define LIBTENSOR_BTOD_ADDITIVE_H

#include "defs.h"
#include "exception.h"
#include "direct_block_tensor_operation.h"

namespace libtensor {

/**	\brief Additive direct block %tensor operation

	\ingroup libtensor
**/
template<class N>
class btod_additive : public direct_block_tensor_operation<N,double> {
public:
	virtual void perform(block_tensor_i<N,double> &bt, double c)
		throw(exception) = 0;

	virtual void perform(block_tensor_i<N,double> &bt) throw(exception) = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_ADDITIVE_H

