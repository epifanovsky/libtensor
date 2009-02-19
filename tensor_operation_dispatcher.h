#ifndef __LIBTENSOR_TENSOR_OPERATION_DISPATCHER_H
#define __LIBTENSOR_TENSOR_OPERATION_DISPATCHER_H

#include <libvmm.h>

#include "defs.h"
#include "exception.h"
#include "permutation.h"
#include "tensor_i.h"

namespace libtensor {

/**	\brief Dispatches %tensor operation events to appropriate handlers
	\param T Tensor element type.

	\ingroup libtensor
**/
template<typename T>
class tensor_operation_dispatcher :
	public libvmm::singleton< tensor_operation_dispatcher<T> > {

	friend libvmm::singleton< tensor_operation_dispatcher<T> >;

public:
	typedef T element_t; //!< Tensor element type

protected:
	tensor_operation_dispatcher();

public:
	element_t *req_dataptr(tensor_i<element_t> &t, const permutation &p)
		throw(exception);

	const T *req_const_dataptr(tensor_i<T> &t, const permutation &p)
		throw(exception);

	void ret_dataptr(tensor_i<T> &t, const T *ptr) throw(exception);
};

template<typename T>
inline tensor_operation_dispatcher<T>::tensor_operation_dispatcher() {
}

template<typename T>
inline T *tensor_operation_dispatcher<T>::req_dataptr(tensor_i<T> &t,
	const permutation &p) throw(exception) {
	return t.get_tensor_operation_handler().req_dataptr(p);
}

template<typename T>
inline const T *tensor_operation_dispatcher<T>::req_const_dataptr(
	tensor_i<T> &t, const permutation &p) throw(exception) {
	return t.get_tensor_operation_handler().req_const_dataptr(p);
}

template<typename T>
inline void tensor_operation_dispatcher<T>::ret_dataptr(
	tensor_i<T> &t, const T *ptr) throw(exception) {
	t.get_tensor_operation_handler().ret_dataptr(ptr);
}

} // namespace libtensor

#endif // __LIBTENSOR_TENSOR_OPERATION_DISPATCHER_H

