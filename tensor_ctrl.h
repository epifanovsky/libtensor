#ifndef LIBTENSOR_TENSOR_CTRL_H
#define LIBTENSOR_TENSOR_CTRL_H

#include "defs.h"
#include "exception.h"
#include "tensor_i.h"

namespace libtensor {

/**	\brief Tensor control

	\ingroup libtensor
**/
template<typename T>
class tensor_ctrl {
private:
	tensor_i<T> &m_t; //!< Controlled tensor

public:
	//!	\name Construction and destruction
	//@{
	tensor_ctrl(tensor_i<T> &t);
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

template<typename T>
inline tensor_ctrl<T>::tensor_ctrl(tensor_i<T> &t) : m_t(t) {
}

template<typename T>
inline tensor_ctrl<T>::~tensor_ctrl() {
}

template<typename T>
inline void tensor_ctrl<T>::req_prefetch() throw(exception) {
	m_t.on_req_prefetch();
}

template<typename T>
inline T *tensor_ctrl<T>::req_dataptr() throw(exception) {
	return m_t.on_req_dataptr();
}

template<typename T>
inline const T *tensor_ctrl<T>::req_const_dataptr() throw(exception) {
	return m_t.on_req_const_dataptr();
}

template<typename T>
inline void tensor_ctrl<T>::ret_dataptr(const T *p) throw(exception) {
	m_t.on_ret_dataptr(p);
}

} // namespace libtensor

#endif // LIBTENSOR_TENSOR_CTRL_H

