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

	\param N Tensor order.
	\param T Tensor element type.
	\param Alloc Memory allocator.

	A direct %tensor has an underlying operator, an object of
	direct_tensor_operation. So, instead of keeping the %tensor elements
	in memory like the regular libtensor::tensor does, the direct
	%tensor know how to calculate the elements. The operation is attached
	to the direct %tensor upon creation and cannot be replaced by another
	operation. However, since the owner of the direct %tensor also owns
	the operation, it can be altered even after the %tensor is created.

	There are two modes in which the direct %tensor can exists: the
	buffering mode and the non-buffering (truly direct) mode. If in the
	buffering mode, the direct %tensor is calculated upon the first
	request, but then the elements are saved in memory to be retrieved
	from this cache if they are needed again. In the truly direct mode,
	the elements are discarded as soon as they are not needed anymore.
	The modes can be switched with enable_buffering() and
	disable_buffering(). By default, buffering is disabled.

	\ingroup libtensor
**/
template<size_t N, typename T, typename Alloc>
class direct_tensor : public tensor_i<N,T> {
public:
	typedef T element_t; //!< Tensor element type
	typedef typename Alloc::ptr_t ptr_t; //!< Memory pointer type

private:
	dimensions<N> m_dims; //!< Tensor dimensions
	direct_tensor_operation<N,T> &m_op; //!< Underlying base operation
	tensor_i<N,T> *m_tensor; //!< Calculated tensor
	tensor_ctrl<N,T> *m_tensor_ctrl; //!< Calculated tensor control
	size_t m_ptrcount; //!< Count the number of pointers given out
	bool m_buffering; //!< Indicates whether buffering is enabled

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the direct %tensor
		\param d Dimensions of the %tensor.
		\param op Underlying operation.
	**/
	direct_tensor(const dimensions<N> &d, direct_tensor_operation<N,T> &op);

	/**	\brief Virtual destructor
	**/
	virtual ~direct_tensor();

	//@}

	//!	\name Implementation of tensor_i<T>
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
	//!	\name Implementation of tensor_i<T>
	//@{
	virtual void on_req_prefetch() throw(exception);
	virtual T *on_req_dataptr() throw(exception);
	virtual const T *on_req_const_dataptr() throw(exception);
	virtual void on_ret_dataptr(const T *p) throw(exception);
	//@}

};

template<size_t N, typename T, typename Alloc>
direct_tensor<N,T,Alloc>::direct_tensor(const dimensions<N> &d,
	direct_tensor_operation<N,T> &op) : m_dims(d), m_op(op), m_tensor(NULL),
	m_tensor_ctrl(NULL), m_ptrcount(0), m_buffering(false) {
}

template<size_t N, typename T, typename Alloc>
direct_tensor<N,T,Alloc>::~direct_tensor() {
	delete m_tensor_ctrl;
	delete m_tensor;
}

template<size_t N, typename T, typename Alloc>
inline const dimensions<N> &direct_tensor<N,T,Alloc>::get_dims() const {
	return m_dims;
}

template<size_t N, typename T, typename Alloc>
inline void direct_tensor<N,T,Alloc>::enable_buffering() {
	m_buffering = true;
}

template<size_t N, typename T, typename Alloc>
inline void direct_tensor<N,T,Alloc>::disable_buffering() {
	if(m_tensor) {
		delete m_tensor_ctrl; m_tensor_ctrl = NULL;
		delete m_tensor; m_tensor = NULL;
	}
	m_buffering = false;
}

template<size_t N, typename T, typename Alloc>
void direct_tensor<N,T,Alloc>::on_req_prefetch() throw(exception) {
	m_op.prefetch();
}

template<size_t N, typename T, typename Alloc>
T *direct_tensor<N,T,Alloc>::on_req_dataptr() throw(exception) {
	throw_exc("direct_tensor<N,T,Alloc>", "on_req_dataptr()",
		"Non-const data cannot be requested from a direct tensor");
	return NULL;
}

template<size_t N, typename T, typename Alloc>
const T *direct_tensor<N,T,Alloc>::on_req_const_dataptr() throw(exception) {
	if(m_tensor == 0) {
		m_tensor = new tensor<N,T,Alloc>(m_dims);
		m_op.perform(*m_tensor);
		m_ptrcount = 0;
		m_tensor_ctrl = new tensor_ctrl<N,T>(*m_tensor);
	}
	m_ptrcount++;
	return m_tensor_ctrl->req_const_dataptr();
}

template<size_t N, typename T, typename Alloc>
void direct_tensor<N,T,Alloc>::on_ret_dataptr(const element_t *p)
	throw(exception) {

	if(m_ptrcount == 0) {
		throw_exc("direct_tensor<N,T,Alloc>",
			"on_ret_dataptr(const T*)", "Event is out of place");
	}
	if(m_tensor_ctrl == NULL) {
		throw_exc("direct_tensor<N,T,Alloc>",
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

