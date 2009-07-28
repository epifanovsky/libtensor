#ifndef LIBTENSOR_SYMMETRY_TARGET_I_H
#define LIBTENSOR_SYMMETRY_TARGET_I_H

#include "defs.h"
#include "exception.h"

namespace libtensor {

template<size_t N, typename T> class symmetry_i;

/**	\brief Symmetry dispatch target interface
	\tparam N Tensor order.
	\tparam T Tensor element type.

	The dispatch mechanism is used to determine the runtime type of
	%symmetry used in an operation. See libtensor::symmetry_i<N, T> for
	more detail about dispatching.

	This interface establishes the default target and is to be implemented
	by real targets. The default target is only called when a specific
	target is not implemented by a derived class.

	\ingroup libtensor_core
 **/
template<size_t N, typename T>
class symmetry_target_i {
public:
	//!	\name Interface libtensor::symmetry_target_i<N, T>
	//@{

	/**	\brief Accepts the default dispatch
		\param sym Symmetry object.
	 **/
	virtual void accept_default(const symmetry_i<N, T> &sym)
		throw(exception) = 0;

	/**	\brief Accepts the default dispatch
		\param sym Symmetry object.
	 **/
	virtual void accept_default(symmetry_i<N, T> &sym) throw(exception) = 0;

	//@}
};

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_TARGET_I_H
