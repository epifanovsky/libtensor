#ifndef __LIBTENSOR_TENSOR_H
#define __LIBTENSOR_TENSOR_H

#include <libvmm.h>

#include "defs.h"
#include "exception.h"
#include "permutation.h"
#include "tensor_i.h"

#include "permutator.h"
#include "tensor_operation_handler_base.h"

namespace libtensor {

/**	\brief Simple implementation of a %tensor, which stores all its
		elements in memory

	\param T Tensor element type.
	\param Alloc Memory allocator.
	\param PermT Permutation type.
	\param Perm Permutator.

	<b>Element type</b>

	Tensor elements can be of any POD type or any class or structure
	that implements a default constructor and the assignment operator.

	\code
	class tensor_element {
	public:
		tensor_element();
		tensor_element &operator=(const tensor_element&);
	};
	\endcode

	<b>Allocator</b>

	Tensor uses a memory allocator to obtain storage. For more information,
	see the libvmm package.

	<b>Storage format</b>

	Tensor elements are stored in memory one after another in the order
	of the running %index. The first %index element is the slowest
	running, the last one is the fastest running.

	<b>Permutations and permutator</b>

	(This section needs an update.)

	This %tensor class delegates %permutation of data to an external
	class, which implements a static method permute. The %permutator must
	be compatible with the %tensor in the element type and storage format.

	\code
	template<typename _T>
	class permutator {
	public:
		static void permute(const _T *src, _T *dst, const dimensions &d,
			const permutation &p);
	};
	\endcode

	In the above interface, \e src and \e dst point to two nonoverlapping
	blocks of memory. \e src contains %tensor elements before %permutation.
	Elements in the permuted order are to be written at \e dst. Dimensions
	\e d is the %dimensions of the %tensor, also specifying the length of
	\e src and \e dst. Permutation \e p specifies the change in the order
	of indices.

	Permutator implementations should assume that all necessary checks
	regarding the validity of the input parameters have been done, and
	the input is consistent and correct.

	<b>Immutability</b>

	A %tensor can be set immutable via set_immutable(), after which only
	reading operations are allowed on the %tensor. Permutations or
	operations that attempt to modify the elements are prohibited. Once
	the %tensor status is set to immutable, it cannot be changed back.
	To perform a check whether the %tensor is mutable or immutable,
	is_immutable() can be used.

	\code
	tensor<double> t(...);

	// Any operations or permutations are allowed with the tensor t

	t.set_immutable();

	// Only reading operations are allowed with the tensor t

	bool b = t.is_immutable(); // true
	\endcode

	<b>Exceptions</b>

	Exceptions libtensor::exception are thrown if a requested operation
	fails for any reason. If an %exception is thrown, the state of the
	%tensor object is the same as it was before the operation.

	\ingroup libtensor
**/
template<typename T, typename Alloc, typename PermT = permutation,
	typename Perm = permutator<T> >
class tensor : public tensor_i<T> {
public:
	typedef T element_t; //!< Tensor element type
	typedef typename Alloc::ptr_t ptr_t; //!< Memory pointer type
	typedef PermT permutation_t; //!< Permutation type

private:
	//! Operation handler
	class operation_handler :
		public tensor_operation_handler_base<element_t> {

	private:
		tensor<T,Alloc,PermT,Perm> &m_t;
		element_t *m_dataptr;

	public:
		operation_handler(tensor<T,Alloc,PermT,Perm> &t) :
			m_t(t), m_dataptr(NULL) {}
		virtual ~operation_handler() {}

		virtual element_t *req_dataptr(const PermT &p)
			throw(exception);
		virtual const element_t *req_const_dataptr(const PermT &p)
			throw(exception);
		virtual element_t *req_range_dataptr(const PermT &p,
			const index_range &r) throw(exception);
		virtual const element_t *req_range_const_dataptr(const PermT &p,
			const index_range &r) throw(exception);
		virtual void ret_dataptr(const element_t *p) throw(exception);
	};
	friend operation_handler;

	dimensions m_dims; //!< Tensor %dimensions
	ptr_t m_data; //!< Pointer to data
	PermT m_perm; //!< Specifies how data elemens should be permuted
	bool m_immutable; //!< Indicates whether the %tensor is immutable
	operation_handler m_oph; //!< Operation handler

public:
	/**	\brief Creates a %tensor with specified %dimensions

		Creates a %tensor with specified %dimensions.

		\param d Dimensions of the %tensor.
		\throw exception If an initialization error occurs.
	**/
	tensor(const dimensions &d) throw(exception);

	/**	\brief Creates a %tensor with the %dimensions of another %tensor
			(by tensor_i<T> interface)

		Creates a %tensor with the %dimensions of another %tensor.
		This constructor doesn't copy the data.

		\param t Another %tensor.
		\throw exception If an initialization error occurs.
	**/
	tensor(const tensor_i<T> &t) throw(exception);

	/**	\brief Creates a %tensor with the %dimensions of another %tensor
			(by tensor<T,Alloc,PermT,Perm> reference)

		Creates a %tensor with the %dimensions of another %tensor.
		This constructor doesn't copy the data.

		\param t Another %tensor.
		\throw exception If an initialization error occurs.
	**/
	tensor(const tensor<T,Alloc,PermT,Perm> &t) throw(exception);

	/**	\brief Virtual destructor
	**/
	virtual ~tensor();

	/**	\brief Checks if the %tensor is immutable

		Returns true if the %tensor is immutable, false otherwise.
	**/
	bool is_immutable() const;

	/**	\brief Sets the %tensor status as immutable.

		Sets the %tensor status as immutable. If the %tensor has already
		been set immutable, it stays immutable.
	**/
	void set_immutable();

	//!	\name Implementation of libtensor::tensor_i< T >
	//@{
	/**	\brief Returns the %dimensions of the %tensor

		Returns the %dimensions of the %tensor.
	**/
	virtual const dimensions &get_dims() const;

	/**	\brief Performs an arbitrary %tensor operation

		Performs an arbitrary %tensor operation.
		For more information about tensor operations, see
		tensor::tensor_operation_i.

		\throw exception If the operation unsuccessful.
	**/
	virtual void operation(tensor_operation_i<T> &op) throw(exception);
	//@}

protected:
	//!	\name Implementation of libtensor::tensor_i< T >
	//@{
	virtual tensor_operation_handler_i<T> &get_tensor_operation_handler();
	//@}

private:
	void throw_exc(const char *method, const char *msg) throw(exception);
};

template<typename T, typename Alloc, typename PermT, typename Perm>
tensor<T,Alloc,PermT,Perm>::tensor(const dimensions &d) throw(exception) :
	m_dims(d), m_perm(m_dims.get_order()), m_oph(*this),
	m_data(Alloc::invalid_ptr) {
#ifdef TENSOR_DEBUG
	if(m_dims.get_size() == 0) {
		throw_exc("tensor(const dimensions&)",
			"Zero tensor size is not allowed");
	}
#endif
	m_data = Alloc::allocate(m_dims.get_size());
	m_immutable = false;
}

template<typename T, typename Alloc, typename PermT, typename Perm>
tensor<T,Alloc,PermT,Perm>::tensor(const tensor_i<T> &t) throw(exception) :
	m_dims(t.get_dims()), m_perm(m_dims.get_order()), m_oph(*this),
	m_data(Alloc::invalid_ptr) {
#ifdef TENSOR_DEBUG
	if(m_dims.get_size() == 0) {
		throw_exc("tensor(const tensor_i<T>&)",
			"Zero tensor size is not allowed");
	}
#endif
	m_data = Alloc::allocate(m_dims.get_size());
	m_immutable = false;
}

template<typename T, typename Alloc, typename PermT, typename Perm>
tensor<T,Alloc,PermT,Perm>::tensor(const tensor<T,Alloc,PermT,Perm> &t)
	throw(exception) : m_dims(t.m_dims), m_perm(m_dims.get_order()),
	m_oph(*this), m_data(Alloc::invalid_ptr) {
#ifdef TENSOR_DEBUG
	if(m_dims.get_size() == 0) {
		throw_exc("tensor(const tensor<T,Alloc,PermT,Perm>&)",
			"Zero tensor size is not allowed");
	}
#endif
	m_data = Alloc::allocate(m_dims.get_size());
	m_immutable = false;
}

template<typename T, typename Alloc, typename PermT, typename Perm>
inline tensor<T,Alloc,PermT,Perm>::~tensor() {
	Alloc::deallocate(m_data);
}

template<typename T, typename Alloc, typename PermT, typename Perm>
inline bool tensor<T,Alloc,PermT,Perm>::is_immutable() const {
	return m_immutable;
}

template<typename T, typename Alloc, typename PermT, typename Perm>
inline void tensor<T,Alloc,PermT,Perm>::set_immutable() {
	m_immutable = true;
}

template<typename T, typename Alloc, typename PermT, typename Perm>
inline const dimensions& tensor<T,Alloc,PermT,Perm>::get_dims() const {
	return m_dims;
}

template<typename T, typename Alloc, typename PermT, typename Perm>
inline void tensor<T,Alloc,PermT,Perm>::operation(tensor_operation_i<T> &op)
	throw(exception) {
	op.perform(*this);
}

template<typename T, typename Alloc, typename PermT, typename Perm>
inline tensor_operation_handler_i<T>&
tensor<T,Alloc,PermT,Perm>::get_tensor_operation_handler() {
	return m_oph;
}

template<typename T, typename Alloc, typename PermT, typename Perm>
inline void tensor<T,Alloc,PermT,Perm>::throw_exc(const char *method,
	const char *msg) throw(exception) {
	char s[1024];
	snprintf(s, 1024, "[libtensor::tensor<T,Alloc,PermT,Perm>::%s] %s.",
		method, msg);
	throw exception(s);
}

template<typename T, typename Alloc, typename PermT, typename Perm>
T *tensor<T,Alloc,PermT,Perm>::operation_handler::req_dataptr(
	const PermT &p) throw(exception) {

	if(m_t.m_immutable) {
		m_t.throw_exc(
			"operation_handler::req_dataptr(const permutation&)",
			"Tensor is immutable, writing operations are "
				"prohibited");
	}

	if(m_dataptr) {
		m_t.throw_exc(
			"operation_handler::req_dataptr(const permutation&)",
			"Data pointer has already been checked out");
	}

	m_dataptr = Alloc::lock(m_t.m_data);

	// Permute elements here if necessary

	return m_dataptr;
}

template<typename T, typename Alloc, typename PermT, typename Perm>
const T *tensor<T,Alloc,PermT,Perm>::operation_handler::req_const_dataptr(
	const PermT &p) throw(exception) {

	if(m_dataptr) {
		m_t.throw_exc(
			"operation_handler::req_dataptr(const permutation&)",
			"Data pointer has already been checked out");
	}

	m_dataptr = Alloc::lock(m_t.m_data);

	// Permute elements here if necessary

	return m_dataptr;
}

template<typename T, typename Alloc, typename PermT, typename Perm>
T *tensor<T,Alloc,PermT,Perm>::operation_handler::req_range_dataptr(
	const PermT &p, const index_range &r) throw(exception) {
	m_t.throw_exc("operation_handler::req_range_dataptr"
		"(const permutation&, const index_range&)",
		"Event hander not implemented");
	return NULL;
}

template<typename T, typename Alloc, typename PermT, typename Perm>
const T *tensor<T,Alloc,PermT,Perm>::operation_handler::req_range_const_dataptr(
	const PermT &p, const index_range &r) throw(exception) {
	m_t.throw_exc("operation_handler::req_range_const_dataptr"
		"(const PermT&, const index_range&)",
		"Event hander not implemented");
	return NULL;
}

template<typename T, typename Alloc, typename PermT, typename Perm>
void tensor<T,Alloc,PermT,Perm>::operation_handler::ret_dataptr(
	const element_t *p) throw(exception) {
	if(m_dataptr != p) {
		m_t.throw_exc(
			"operation_handler::ret_dataptr(const element_t*)",
			"Unrecognized data pointer");
	}
	Alloc::unlock(m_t.m_data);
	m_dataptr = NULL;
}

} // namespace libtensor

#endif // __LIBTENSOR_TENSOR_H

