#ifndef LIBTENSOR_TENSOR_OPERATION_H
#define LIBTENSOR_TENSOR_OPERATION_H

#include "defs.h"
#include "exception.h"
#include "permutation.h"
#include "tensor_i.h"
#include "tensor_operation_handler.h"

namespace libtensor {

/**	\brief Base class for %tensor operations

	Any implementation of a %tensor operation has to derive from this
	base class in order to get direct access to %tensor elements.

	<b>Accessing %tensor elements</b>

	To obtain a memory pointer to %tensor elements, use req_dataptr()
	or req_const_dataptr(), depending on whether the data are needed for
	reading only or reading and writing:
	\code
	tensor_i<double> &t(...);
	permutation &p(...);

	double *pd = req_dataptr(t, p);
	\endcode
	or
	\code
	tensor_i<double> &t(...);
	permutation &p(...);

	const double *pd = req_const_dataptr(t, p);
	\endcode

	Once the operation is done working with the data, it must return
	the pointer:
	\code
	ret_dataptr(t, pd);
	\endcode

	The pointer can only be checked out once. Another request for the
	pointer will cause an exception, unless the pointer was properly
	turned in.
	\code
	tensor_i<double> &t(...);
	permutation &p(...);

	double *pd = req_dataptr(t, p);
	// ...
	ret_dataptr(t, pd); pd = NULL;

	// ...

	pd = req_dataptr(t, p); // OK, the pointer was returned before
	// ...
	ret_dataptr(t, pd); pd = NULL;
	\endcode

	\code
	tensor_i<double> &t(...);
	permutation &p(...);

	double *pd = req_dataptr(t, p);

	// Error! This will cause an exception
	const double *cpd = req_const_dataptr(t, p);
	\endcode

	<b>Permutation of %tensor elements</b>

	When implementing an operation, one should realize that requesting
	permuted %tensor elements can come with a cost: the %tensor will
	permute the data when handling the request.
	To find out the associated cost, use req_permutation_cost(). To obtain
	the least-cost %permutation, use req_simplest_permutation().

	\ingroup libtensor
**/
template<typename T>
class tensor_operation {
public:
	typedef T element_t; //!< Tensor element type

protected:
	//!	\name Events
	//@{

	/**	\brief Request to move the %tensor data to the fast memory
		\param t Tensor.
	**/
	void req_prefetch(tensor_i<element_t> &t) throw(exception);

	/**	\brief Checks out a memory pointer to %tensor elements permuted
			as requested
		\param t Tensor.
	**/
	element_t *req_dataptr(tensor_i<element_t> &t) throw(exception);

	/**	\brief Checks out a const memory pointer to %tensor elements
			permuted as requested
		\param t Tensor.
	**/
	const element_t *req_const_dataptr(tensor_i<element_t> &t)
		throw(exception);

	/**	\brief Turns in a previously checked out memory pointer.
		\param t Tensor.
		\param ptr Memory pointer previously checked out with
			req_dataptr() or req_const_dataptr()
	**/
	void ret_dataptr(tensor_i<element_t> &t, const element_t *p)
		throw(exception);

	//@}
};

template<typename T>
inline void tensor_operation<T>::req_prefetch(tensor_i<T> &t)
	throw(exception) {
	t.get_tensor_operation_handler().req_prefetch();
}

template<typename T>
inline T *tensor_operation<T>::req_dataptr(tensor_i<T> &t) throw(exception) {
	return t.get_tensor_operation_handler().req_dataptr();
}

template<typename T>
inline const T *tensor_operation<T>::req_const_dataptr(tensor_i<T> &t)
	throw(exception) {
	return t.get_tensor_operation_handler().req_const_dataptr();
}

template<typename T>
inline void tensor_operation<T>::ret_dataptr(tensor_i<T> &t,
	const T *ptr) throw(exception) {
	t.get_tensor_operation_handler().ret_dataptr(ptr);
}

} // namespace libtensor

#endif // LIBTENSOR_TENSOR_OPERATION_H

