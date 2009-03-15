#ifndef LIBTENSOR_BTOD_ADDITIVE_H
#define LIBTENSOR_BTOD_ADDITIVE_H

#include "defs.h"
#include "exception.h"
#include "direct_block_tensor_operation.h"

namespace libtensor {

/**	\brief Additive direct block %tensor operation

	\ingroup libtensor
**/
class btod_additive : public direct_block_tensor_operation<double> {
public:
	virtual void perform(block_tensor_i<double> &bt, double c)
		throw(exception) = 0;

	virtual void perform(block_tensor_i<double> &bt) throw(exception) = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_ADDITIVE_H

