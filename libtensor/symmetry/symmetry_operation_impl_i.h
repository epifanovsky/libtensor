#ifndef LIBTENSOR_SYMMETRY_OPERATION_IMPL_I_H
#define LIBTENSOR_SYMMETRY_OPERATION_IMPL_I_H

#include <string>
#include "symmetry_operation_params.h"

namespace libtensor {


/**	\brief Interface for concrete %symmetry operations
 **/
class symmetry_operation_impl_i {
public:
	/**	\brief Returns the %symmetry element class id
	 **/
	virtual const char *get_id() const = 0;

	/**	\brief Clones the implementation
	 **/
	virtual symmetry_operation_impl_i *clone() const = 0;

	/**	\brief Invokes the operation
	 **/
	virtual void perform(symmetry_operation_params_i &params) const = 0;
};


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_OPERATION_IMPL_I_H

