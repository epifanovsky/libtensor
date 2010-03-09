#ifndef LIBTENSOR_SYMMETRY_OPERATION_BASE_H
#define LIBTENSOR_SYMMETRY_OPERATION_BASE_H

#include "symmetry_operation_handlers.h"

namespace libtensor {


template<typename SymOp>
class symmetry_operation_base {
public:
	symmetry_operation_base() {

		symmetry_operation_handlers<SymOp>::install_handlers();
	}
};


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_OPERATION_BASE_H