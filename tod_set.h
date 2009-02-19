#ifndef __LIBTENSOR_TOD_SET_H
#define __LIBTENSOR_TOD_SET_H

#include "defs.h"
#include "exception.h"
#include "tensor_operation_base.h"

namespace libtensor {

/**	\brief Tensor (double) operation: assign all elements a value

	\ingroup libtensor

**/
class tod_set : public tensor_operation_base<double> {
private:
	double m_val; //!< Value

public:
	//! Default constructor
	tod_set(const double v = 0.0);

	//! Virtual destructor
	~tod_set();

	virtual void perform(tensor_i<double> &t) throw(exception);
};

inline tod_set::tod_set(const double v) {
	m_val = v;
}

inline tod_set::~tod_set() {
}

} // namespace libtensor

#endif // __LIBTENSOR_TOD_SET_H

