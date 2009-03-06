#ifndef LIBTENSOR_DIRECT_TENSOR_H
#define LIBTENSOR_DIRECT_TENSOR_H

#include "defs.h"
#include "exception.h"
#include "direct_tensor_operation.h"
#include "tensor_i.h"
#include "tensor.h"

namespace libtensor {

/**	\brief Tensor calculated directly on demand
	\param T Tensor element type.
	\param Alloc Memory allocator.

	\ingroup libtensor
**/
template<typename T, typename Alloc>
class direct_tensor : public tensor_i<T> {
public:
	typedef T element_t; //!< Tensor element type
	typedef typename Alloc::ptr_t ptr_t; //!< Memory pointer type

private:
	class toh : public tensor_operation_handler<T> {
	private:
		direct_tensor<T,Alloc> &m_t; //!< Underlying tensor

	public:
		//!	\name Construction and destruction
		//@{
		//!	\brief Initializes the handler
		toh(direct_tensor<T,Alloc> &t) : m_t(t) { }
		//!	\brief Destroys the handler
		virtual ~toh() { }
		//@}

		virtual element_t *req_dataptr() throw(exception);
		virtual const element_t *req_const_dataptr()
			throw(exception);
		virtual void ret_dataptr(const element_t *p) throw(exception);
	};

	dimensions m_dims; //!< Tensor dimensions
	toh m_toh; //!< Tensor operation handler
	direct_tensor_operation<T> &m_op; //!< Underlying base operation
	tensor_i<T> *m_tensor; //!< Calculated tensor
	size_t m_ptrcount; //!< Count the number of pointers given out
	bool m_buffering; //!< Indicates whether buffering is enabled

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the direct %tensor
		\param d Dimensions of the %tensor.
		\param op Underlying operation.
	**/
	direct_tensor(const dimensions &d, direct_tensor_operation<T> &op);

	/**	\brief Virtual destructor
	**/
	virtual ~direct_tensor();

	//@}

	//!	\name Implementation of tensor_i<T>
	//@{
	virtual const dimensions &get_dims() const;
	//@}

	//!	\name Buffering
	//@{

	/**	\brief Enables buffering
	**/
	void enable_buffering();

	/**	\brief Disables buffering
	**/
	void disable_buffering();

	//@}

protected:
	//!	\name Implementation of tensor_i<T>
	//@{
	virtual tensor_operation_handler<T> &get_tensor_operation_handler();
	//@}

private:
	const element_t *req_const_dataptr() throw(exception);
	void ret_dataptr(const element_t *p) throw(exception);
	void throw_exc(const char *method, const char *msg) throw(exception);
};

template<typename T, typename Alloc>
direct_tensor<T,Alloc>::direct_tensor(const dimensions &d,
	direct_tensor_operation<T> &op) : m_dims(d), m_toh(*this), m_op(op),
	m_tensor(NULL), m_ptrcount(0), m_buffering(false) {
}

template<typename T, typename Alloc>
direct_tensor<T,Alloc>::~direct_tensor() {
	delete m_tensor;
}

template<typename T, typename Alloc>
inline const dimensions &direct_tensor<T,Alloc>::get_dims() const {
	return m_dims;
}

template<typename T, typename Alloc>
tensor_operation_handler<T>&
direct_tensor<T,Alloc>::get_tensor_operation_handler() {
	return m_toh;
}

template<typename T, typename Alloc>
inline void direct_tensor<T,Alloc>::enable_buffering() {
	m_buffering = true;
}

template<typename T, typename Alloc>
inline void direct_tensor<T,Alloc>::disable_buffering() {
	if(m_tensor) {
		delete m_tensor; m_tensor = NULL;
	}
	m_buffering = false;
}

template<typename T, typename Alloc>
inline void direct_tensor<T,Alloc>::throw_exc(const char *method,
	const char *msg) throw(exception) {
	char s[1024];
	snprintf(s, 1024, "[libtensor::direct_tensor<T,Alloc>::%s] %s.",
		method, msg);
	throw exception(s);
}

template<typename T, typename Alloc>
T *direct_tensor<T,Alloc>::toh::req_dataptr() throw(exception) {
	m_t.throw_exc("toh::req_dataptr()",
		"Non-const data cannot be requested from a direct tensor");
}

template<typename T, typename Alloc>
const T *direct_tensor<T,Alloc>::toh::req_const_dataptr()
	throw(exception) {
	return m_t.req_const_dataptr();
}

template<typename T, typename Alloc>
const T *direct_tensor<T,Alloc>::req_const_dataptr() throw(exception) {
	if(m_tensor == 0) {
		m_tensor = new tensor<T,Alloc>(m_dims);
		m_op.perform(*m_tensor);
		m_ptrcount = 0;
	}
	m_ptrcount++;
	return get_tensor_operation_handler1(*m_tensor).req_const_dataptr();
}

template<typename T, typename Alloc>
void direct_tensor<T,Alloc>::toh::ret_dataptr(const element_t *p)
	throw(exception) {
	m_t.ret_dataptr(p);
}

template<typename T, typename Alloc>
void direct_tensor<T,Alloc>::ret_dataptr(const element_t *p) throw(exception) {
	if(m_ptrcount == 0) {
		throw_exc("direct_tensor<T,Alloc>::ret_dataptr(const T*)",
			"Event is out of place");
	}
	m_ptrcount --;
	if(m_ptrcount == 0 && !m_buffering) {
		delete m_tensor; m_tensor = NULL;
	}
}

} // namespace libtensor

#endif // LIBTENSOR_DIRECT_TENSOR_H

