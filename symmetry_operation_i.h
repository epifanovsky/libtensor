#ifndef LIBTENSOR_SYMMETRY_OPERATION_I_H
#define LIBTENSOR_SYMMETRY_OPERATION_I_H

#include "defs.h"
#include "exception.h"
#include "symmetry_i.h"

namespace libtensor {

template<size_t N, typename T>
class symmetry_operation_i {
public:
	/**	\brief Virtual destructor
	 **/
	virtual ~symmetry_operation_i() { }

	virtual void perform(symmetry_i<N, T> &sym) throw(exception) = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_OPERATION_I_H
