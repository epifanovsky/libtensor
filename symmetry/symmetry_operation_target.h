#ifndef LIBTENSOR_SYMMETRY_OPERATION_TARGET_H
#define LIBTENSOR_SYMMETRY_OPERATION_TARGET_H

#include "defs.h"
#include "exception.h"

namespace libtensor {

template<size_t N, typename T, typename Sym>
class symmetry_operation_target {
public:
	virtual void perform(Sym &sym) throw(exception) = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_OPERATION_TARGET_H
