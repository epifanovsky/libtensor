#ifndef __LIBTENSOR_TENSOR_OPERATION_HANDLER_BASE_H
#define __LIBTENSOR_TENSOR_OPERATION_HANDLER_BASE_H

#include "defs.h"
#include "exception.h"
#include "tensor_i.h"
#include "tensor_operation_handler_i.h"

namespace libtensor {

template<typename _T>
class tensor_operation_handler_base : public tensor_operation_handler_i<_T> {
public:
	typedef _T element_t;

public:
	virtual element_t *req_dataptr(const permutation &p) throw(exception);

	virtual const element_t *req_const_dataptr(const permutation &p)
		throw(exception);

	virtual element_t *req_range_dataptr(const permutation &p,
		const index_range &r) throw(exception);

	virtual const element_t *req_range_const_dataptr(const permutation &p,
		const index_range &r) throw(exception);

	virtual void ret_dataptr(const element_t *p) throw(exception);

private:
	/**	\brief Throws an exception
	**/
	void throw_exc(const char *method, const char *msg) const
		throw(exception);
};

template<typename _T>
tensor_operation_handler_base<_T>::element_t*
tensor_operation_handler_base<_T>::req_dataptr(const permutation &p)
	throw(exception) {
	throw_exc("tensor_operation_handler_base<_T>::req_dataptr(const permutation&)",
		"Unhandled event");
	return NULL;
}

template<typename _T>
const tensor_operation_handler_base<_T>::element_t*
tensor_operation_handler_base<_T>::req_const_dataptr(const permutation &p)
	throw(exception) {
	throw_exc("tensor_operation_handler_base<_T>::req_const_dataptr(const permutation&)",
		"Unhandled event");
	return NULL;
}

template<typename _T>
tensor_operation_handler_base<_T>::element_t*
tensor_operation_handler_base<_T>::req_range_dataptr(const permutation &p,
	const index_range &r) throw(exception) {
	throw_exc("tensor_operation_handler_base<_T>::req_range_dataptr"
		"(const permutation&, const index_range&)",
		"Unhandled event");
	return NULL;
}

template<typename _T>
const tensor_operation_handler_base<_T>::element_t*
tensor_operation_handler_base<_T>::req_range_const_dataptr(const permutation &p,
	const index_range &r) throw(exception) {
	throw_exc("tensor_operation_handler_base<_T>::req_range_const_dataptr"
		"(const permutation&, const index_range&)",
		"Unhandled event");
	return NULL;
}

template<typename _T>
void tensor_operation_handler_base<_T>::ret_dataptr(const element_t *p)
	throw(exception) {
	throw_exc("tensor_operation_handler_base<_T>::ret_dataptr(const element_t*)",
		"Unhandled event");
}

template<typename _T>
void tensor_operation_handler_base<_T>::throw_exc(const char *method,
	const char *msg) const throw(exception) {
	char s[1024];
	snprintf(s, 1024, "[tensor::tensor_operation_handler_base<_T>::%s] %s.",
		method, msg);
	throw exception(s);
}

} // namespace libtensor

#endif // __LIBTENSOR_TENSOR_OPERATION_HANDLER_BASE_H

