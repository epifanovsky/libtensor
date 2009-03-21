#ifndef LIBTENSOR_TENSOR_CTRL_H
#define LIBTENSOR_TENSOR_CTRL_H

#include "defs.h"
#include "exception.h"
#include "tensor_i.h"

namespace libtensor {

/**	\brief Tensor control

	\param N Tensor order.
	\param T Tensor element type.

	\ingroup libtensor
**/
template<size_t N, typename T>
class tensor_ctrl {
private:
	tensor_i<N,T> &m_t; //!< Controlled tensor

public:
	//!	\name Construction and destruction
	//@{
	tensor_ctrl(tensor_i<N,T> &t);
	~tensor_ctrl();
	//@}

	//!	\name Event forwarding
	//@{
	void req_prefetch() throw(exception);
	T *req_dataptr() throw(exception);
	const T *req_const_dataptr() throw(exception);
	void ret_dataptr(const T *p) throw(exception);
	//@}
};

template<size_t N, typename T>
inline tensor_ctrl<N,T>::tensor_ctrl(tensor_i<N,T> &t) : m_t(t) {
}

template<size_t N, typename T>
inline tensor_ctrl<N,T>::~tensor_ctrl() {
}

template<size_t N, typename T>
inline void tensor_ctrl<N,T>::req_prefetch() throw(exception) {
	m_t.on_req_prefetch();
}

template<size_t N, typename T>
inline T *tensor_ctrl<N,T>::req_dataptr() throw(exception) {
	return m_t.on_req_dataptr();
}

template<size_t N, typename T>
inline const T *tensor_ctrl<N,T>::req_const_dataptr() throw(exception) {
	return m_t.on_req_const_dataptr();
}

template<size_t N, typename T>
inline void tensor_ctrl<N,T>::ret_dataptr(const T *p) throw(exception) {
	m_t.on_ret_dataptr(p);
}

} // namespace libtensor

#endif // LIBTENSOR_TENSOR_CTRL_H

