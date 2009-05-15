#ifndef LIBTENSOR_DIRECT_TENSOR_H
#define LIBTENSOR_DIRECT_TENSOR_H

#include "defs.h"
#include "exception.h"
#include "direct_tensor_operation.h"
#include "tensor_i.h"
#include "tensor.h"
#include "tensor_ctrl.h"

namespace libtensor {

/**	\brief Tensor calculated directly on demand
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Op Tensor operation type.
	\tparam Alloc Memory allocator type.

	<b>Direct %tensor</b>

	Given a %tensor operation, a direct %tensor computes its elements when
	requested by another %tensor operation for which the direct %tensor is
	an argument. Unlike regular tensors, direct tensors store the underlying
	operation rather than %tensor elements.

	The underlying operation is set up when the direct %tensor is created.
	A copy of the operation is stored within the %tensor, so once the
	%tensor object is created, its underlying operation cannot be altered.

	There are two modes in which the direct %tensor can exists: the
	buffering mode and the non-buffering (truly direct) mode. When in the
	buffering mode, the direct %tensor is calculated upon the first
	request, but then the elements are saved in memory to be retrieved
	from this cache if they are needed again. In the truly direct mode,
	the elements are discarded as soon as they are not needed anymore.
	The modes can be switched with enable_buffering() and
	disable_buffering(). By default, buffering is disabled.

	<b>Underlying operation</b>

	The underlying %tensor operation is a template parameter for
	libtensor::direct_tensor. There are no limitations on what that
	operation is, however it must implement a copy constructor,
	\c prefetch() and \c perform() methods. The \c prefetch() method
	takes no arguments and may throw an exception. The \c perform() method
	has one argument: the %tensor that accepts the result of the operation.
	The template parameters of the argument %tensor must agree with those
	of the direct %tensor.

	Example:
	\code
	class operation {
	public:
		// This constructor takes some operation arguments
		operation(...);

		// Copy constructor (required)
		operation(const operation &op);

		// Requests that the arguments are prefetched
		void prefetch() throw(exception);

		// Performs the operation
		void perform(tensor_i<N, T> &t) throw(exception);
	};
	\endcode

	\ingroup libtensor
**/
template<size_t N, typename T, typename Op, typename Alloc>
class direct_tensor : public tensor_i<N, T> {
public:
	typedef T element_t; //!< Tensor element type
	typedef Op operation_t; //!< Tensor operation type
	typedef Alloc allocator_t; //!< Memory allocator type
	typedef typename Alloc::ptr_t ptr_t; //!< Memory pointer type

private:
	dimensions<N> m_dims; //!< Tensor dimensions
	operation_t m_op; //!< Underlying operation
	tensor_i<N,T> *m_tensor; //!< Calculated tensor
	tensor_ctrl<N,T> *m_tensor_ctrl; //!< Calculated tensor control
	size_t m_ptrcount; //!< Counts the number of data pointers given out
	bool m_buffering; //!< Indicates whether buffering is enabled

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the direct %tensor
		\param d Dimensions of the %tensor.
		\param op Underlying operation.
	**/
	direct_tensor(const dimensions<N> &d, const operation_t &op);

	/**	\brief Virtual destructor
	**/
	virtual ~direct_tensor();

	//@}

	//!	\name Implementation of tensor_i
	//@{
	virtual const dimensions<N> &get_dims() const;
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
	//!	\name Implementation of tensor_i
	//@{
	virtual void on_req_prefetch() throw(exception);
	virtual T *on_req_dataptr() throw(exception);
	virtual const T *on_req_const_dataptr() throw(exception);
	virtual void on_ret_dataptr(const T *p) throw(exception);
	//@}

};

template<size_t N, typename T, typename Op, typename Alloc>
direct_tensor<N, T, Op, Alloc>::direct_tensor(const dimensions<N> &d,
	const operation_t &op) : m_dims(d), m_op(op), m_tensor(NULL),
	m_tensor_ctrl(NULL), m_ptrcount(0), m_buffering(false) {
}

template<size_t N, typename T, typename Op, typename Alloc>
direct_tensor<N, T, Op, Alloc>::~direct_tensor() {
	delete m_tensor_ctrl;
	delete m_tensor;
}

template<size_t N, typename T, typename Op, typename Alloc>
inline const dimensions<N> &direct_tensor<N, T, Op, Alloc>::get_dims() const {
	return m_dims;
}

template<size_t N, typename T, typename Op, typename Alloc>
inline void direct_tensor<N, T, Op, Alloc>::enable_buffering() {
	m_buffering = true;
}

template<size_t N, typename T, typename Op, typename Alloc>
inline void direct_tensor<N, T, Op, Alloc>::disable_buffering() {
	if(m_tensor) {
		delete m_tensor_ctrl; m_tensor_ctrl = NULL;
		delete m_tensor; m_tensor = NULL;
	}
	m_buffering = false;
}

template<size_t N, typename T, typename Op, typename Alloc>
void direct_tensor<N, T, Op, Alloc>::on_req_prefetch() throw(exception) {
	m_op.prefetch();
}

template<size_t N, typename T, typename Op, typename Alloc>
T *direct_tensor<N, T, Op, Alloc>::on_req_dataptr() throw(exception) {
	throw_exc("direct_tensor<N, T, Op, Alloc>", "on_req_dataptr()",
		"Non-const data cannot be requested from a direct tensor");
	return NULL;
}

template<size_t N, typename T, typename Op, typename Alloc>
const T *direct_tensor<N, T, Op, Alloc>::on_req_const_dataptr()
	throw(exception) {

	if(m_tensor == 0) {
		m_tensor = new tensor<N, T, Alloc>(m_dims);
		m_op.perform(*m_tensor);
		m_ptrcount = 0;
		m_tensor_ctrl = new tensor_ctrl<N, T>(*m_tensor);
	}
	m_ptrcount++;
	return m_tensor_ctrl->req_const_dataptr();
}

template<size_t N, typename T, typename Op, typename Alloc>
void direct_tensor<N, T, Op, Alloc>::on_ret_dataptr(const element_t *p)
	throw(exception) {

	if(m_ptrcount == 0) {
		throw_exc("direct_tensor<N, T, Op, Alloc>",
			"on_ret_dataptr(const T*)", "Event is out of place");
	}
	if(m_tensor_ctrl == NULL) {
		throw_exc("direct_tensor<N, T, Op, Alloc>",
			"on_ret_dataptr(const T*)",
			"NULL tensor control object");
	}
	m_tensor_ctrl->ret_dataptr(p);
	m_ptrcount --;
	if(m_ptrcount == 0 && !m_buffering) {
		delete m_tensor_ctrl; m_tensor_ctrl = NULL;
		delete m_tensor; m_tensor = NULL;
	}
}

} // namespace libtensor

#endif // LIBTENSOR_DIRECT_TENSOR_H

