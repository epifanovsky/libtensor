#ifndef LIBTENSOR_TOD_COPY_H
#define LIBTENSOR_TOD_COPY_H

#include "defs.h"
#include "exception.h"
#include "tensor_i.h"
#include "tensor_ctrl.h"
#include "direct_tensor_operation.h"

namespace libtensor {

/**	\brief Copies the content of one %tensor to another

	This operation copies all elements of one %tensor to another
	%tensor. The tensors must have the same dimensions, or an exception
	will be thrown.

	Example:
	\code
	tensor_i<2,double> &t1(...), &t2(...);
	tod_copy<2> cp(t1);
	cp.perform(t2); // Copies t1 to t2
	\endcode

	\ingroup libtensor_tod
**/
template<size_t N>
class tod_copy : public direct_tensor_operation<N,double> {
private:
	tensor_i<N,double> &m_tsrc;
	tensor_ctrl<N,double> m_tctrl_src;

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the operation
		\param tsrc Tensor to create copies from.
	**/
	tod_copy(tensor_i<N,double> &tsrc);

	/**	\brief Virtual destructor
	**/
	virtual ~tod_copy();

	//@}

	//!	\name Implementation of direct_tensor_operation
	//@{

	virtual void prefetch() throw(exception);

	/**	\brief Makes a copy
		\throw Exception if the tensors have different dimensions
			or another error occurs
	**/
	virtual void perform(tensor_i<N,double> &t) throw(exception);

	//@}

};

template<size_t N>
inline tod_copy<N>::tod_copy(tensor_i<N,double> &tsrc) : m_tsrc(tsrc),
	m_tctrl_src(tsrc) {
}

template<size_t N>
inline tod_copy<N>::~tod_copy() {
}

template<size_t N>
inline void tod_copy<N>::prefetch() throw(exception) {
	m_tctrl_src.req_prefetch();
}

template<size_t N>
void tod_copy<N>::perform(tensor_i<N,double> &tdst) throw(exception) {
	const dimensions<N> &dim_src(m_tsrc.get_dims()),
		&dim_dst(tdst.get_dims());
	for(size_t i=0; i<N; i++) if(dim_src[i]!=dim_dst[i]) {
		throw_exc("tod_copy<N>", "perform(tensor_i<N,double>&)",
			"The tensors have different dimensions");
	}

	tensor_ctrl<N,double> tctrl_dst(tdst);
	const double *psrc = m_tctrl_src.req_const_dataptr();
	double *pdst = tctrl_dst.req_dataptr();

	cblas_dcopy(dim_src.get_size(), psrc, 1, pdst, 1);

	m_tctrl_src.ret_dataptr(psrc);
	tctrl_dst.ret_dataptr(pdst);
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_COPY_H

