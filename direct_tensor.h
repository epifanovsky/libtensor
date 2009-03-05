#ifndef LIBTENSOR_DIRECT_TENSOR_H
#define LIBTENSOR_DIRECT_TENSOR_H

#include "defs.h"
#include "exception.h"
#include "direct_tensor_operation.h"
#include "tensor_i.h"
#include "permutator.h"

namespace libtensor {

/**	\brief Tensor calculated directly upon request
	\param T Tensor element type.
	\param Alloc Memory allocator.
	\param Perm Permutator.

	\ingroup libtensor
**/
template<typename T, typename Alloc, typename Perm = permutator<T> >
class direct_tensor : public tensor_i<T> {
public:
	typedef T element_t; //!< Tensor element type
	typedef typename Alloc::ptr_t ptr_t; //!< Memory pointer type

private:
	class toh : public tensor_operation_handler<T> {
	public:
		virtual element_t *req_dataptr() throw(exception);
		virtual const element_t *req_const_dataptr()
			throw(exception);
		virtual void ret_dataptr(const element_t *p) throw(exception);
	};

	dimensions m_dims;
	direct_tensor_operation<T> &m_op;

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the direct %tensor
		\param op Underlying operation.
		\param d Dimensions of the %tensor.
	**/
	direct_tensor(direct_tensor_operation<T> &op, const dimensions &d);

	/**	\brief Virtual destructor
	**/
	virtual ~direct_tensor();

	//@}

	//!	\name Implementation of tensor_i<T>
	//@{
	virtual const dimensions &get_dims() const;
	//@}

protected:
	//!	\name Implementation of tensor_i<T>
	//@{
	virtual tensor_operation_handler<T> &get_tensor_operation_handler();
	//@}

private:
	void throw_exc(const char *method, const char *msg) throw(exception);
};

template<typename T, typename Alloc, typename Perm>
direct_tensor<T,Alloc,Perm>::direct_tensor(
	direct_tensor_operation<T> &op, const dimensions &d) :
	m_op(op), m_dims(d) {
}

template<typename T, typename Alloc, typename Perm>
direct_tensor<T,Alloc,Perm>::~direct_tensor() {
}

template<typename T, typename Alloc, typename Perm>
inline const dimensions &direct_tensor<T,Alloc,Perm>::get_dims() const {
	return m_dims;
}

template<typename T, typename Alloc, typename Perm>
tensor_operation_handler<T>&
direct_tensor<T,Alloc,Perm>::get_tensor_operation_handler() {
	throw_exc("get_tensor_operation_handler()", "NIY");
}

template<typename T, typename Alloc, typename Perm>
inline void direct_tensor<T,Alloc,Perm>::throw_exc(const char *method,
	const char *msg) throw(exception) {
	char s[1024];
	snprintf(s, 1024, "[libtensor::direct_tensor<T,Alloc,Perm>::%s] %s.",
		method, msg);
	throw exception(s);
}

template<typename T, typename Alloc, typename Perm>
T *direct_tensor<T,Alloc,Perm>::toh::req_dataptr() throw(exception) {
	throw_exc("toh::req_dataptr()",
		"Non-const data cannot be requested from a direct tensor");
}

template<typename T, typename Alloc, typename Perm>
const T *direct_tensor<T,Alloc,Perm>::toh::req_const_dataptr()
	throw(exception) {
	throw_exc("toh::req_const_dataptr()", "NIY");
}

template<typename T, typename Alloc, typename Perm>
void direct_tensor<T,Alloc,Perm>::toh::ret_dataptr(const element_t *p)
	throw(exception) {
	throw_exc("toh::ret_dataptr(const T*)", "NIY");
}

} // namespace libtensor

#endif // LIBTENSOR_DIRECT_TENSOR_H

