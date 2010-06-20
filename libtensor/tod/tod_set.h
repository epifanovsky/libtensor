#ifndef LIBTENSOR_TOD_SET_H
#define LIBTENSOR_TOD_SET_H

#include "../defs.h"
#include "../exception.h"
#include "../core/tensor_ctrl.h"

namespace libtensor {

/**	\brief Assigns a value to all elements

	\ingroup libtensor_tod
**/
template<size_t N>
class tod_set {
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
	void perform(tensor_i<N,double> &t) throw(exception);

	void prefetch() throw(exception);

	//@}
};

template<size_t N>
inline tod_set<N>::tod_set(const double v) {
	m_val = v;
}

template<size_t N>
inline tod_set<N>::~tod_set() {
}

template<size_t N>
void tod_set<N>::perform(tensor_i<N,double> &t) throw(exception) {
	tensor_ctrl<N,double> tctrl(t);
	double *d = tctrl.req_dataptr();
	size_t sz = t.get_dims().get_size();
	#pragma unroll(8)
	for(size_t i=0; i<sz; i++) d[i] = m_val;
	tctrl.ret_dataptr(d);
}

template<size_t N>
inline void tod_set<N>::prefetch() throw(exception) {
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_H

