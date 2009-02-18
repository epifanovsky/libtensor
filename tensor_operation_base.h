#ifndef __LIBTENSOR_TENSOR_OPERATION_BASE_H
#define __LIBTENSOR_TENSOR_OPERATION_BASE_H

#include "defs.h"
#include "exception.h"
#include "tensor_operation_dispatcher.h"
#include "tensor_operation_i.h"

namespace libtensor {

/**	\brief Base class for tensor operations
**/
template<typename T, typename PermT>
class tensor_operation_base : public tensor_operation_i<T> {
public:
	typedef T element_t; //!< Tensor element type
	typedef PermT permutation_t; //!< Permutation type

protected:
	element_t *req_dataptr(tensor_i<element_t> &t, const permutation_t &p)
		throw(exception);

	void ret_dataptr(tensor_i<element_t> &t, const element_t *p)
		throw(exception);
};

template<typename T, typename PermT>
inline T *tensor_operation_base<T,PermT>::req_dataptr(tensor_i<T> &t,
	const PermT &p) throw(exception) {
	return tensor_operation_dispatcher::get_instance().req_dataptr(t, p);
}

template<typename T, typename PermT>
inline void tensor_operation_base<T,PermT>::ret_dataptr(tensor_i<T> &t,
	const T *ptr) throw(exception) {
	tensor_operation_dispatcher::get_instance().ret_dataptr(t, ptr);
}

} // namespace libtensor

#endif // __LIBTENSOR_TENSOR_OPERATION_BASE_H

