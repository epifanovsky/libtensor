#ifndef __LIBTENSOR_TOD_SET_H
#define __LIBTENSOR_TOD_SET_H

#include "defs.h"
#include "exception.h"
#include "direct_tensor_operation.h"

namespace libtensor {

/**	\brief Assigns a value to all elements

	\ingroup libtensor_tod
**/
class tod_set : public direct_tensor_operation<double> {
private:
	double m_val; //!< Value

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the operation
		\param v Tensor element value
	**/
	tod_set(const double v = 0.0);

	/**	\brief Destructor
	**/
	~tod_set();

	//@}

	//!	\name Operation
	//@{

	/**	\brief Assigns the elements of a tensor a value
		\param t Tensor.
	**/
	void perform(tensor_i<double> &t) throw(exception);

	virtual void prefetch() throw(exception);

	//@}
};

inline tod_set::tod_set(const double v) {
	m_val = v;
}

inline tod_set::~tod_set() {
}

inline void tod_set::prefetch() throw(exception) {
}

} // namespace libtensor

#endif // __LIBTENSOR_TOD_SET_H

