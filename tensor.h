#ifndef __LIBTENSOR_TENSOR_H
#define __LIBTENSOR_TENSOR_H

#include <libvmm.h>

#include "defs.h"
#include "exception.h"
#include "permutation.h"
#include "tensor_i.h"

#include "permutator.h"

namespace libtensor {

/**	\brief Simple %tensor, which stores all its elements in memory

	\param T Tensor element type.
	\param Alloc Memory allocator.
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

	<b>Tensor operations</b>

	Tensor elements cannot be accessed directly. Only an extension of
	tensor_operation has the ability to read and write them.

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
	template<typename T>
	class permutator {
	public:
		static void permute(const T *src, T *dst, const dimensions &d,
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
	reading operations are allowed on the %tensor. Operations that attempt
	to modify the elements are prohibited and will cause an exception.
	Once the %tensor status is set to immutable, it cannot be changed back.
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

	Exceptions exception are thrown if a requested operation fails for any
	reason. If an %exception is thrown, the state of the %tensor object
	is undefined.

	\ingroup libtensor
**/
template<typename T, typename Alloc, typename Perm = permutator<T> >
class tensor : public tensor_i<T> {
public:
	typedef T element_t; //!< Tensor element type
	typedef typename Alloc::ptr_t ptr_t; //!< Memory pointer type

private:
	/**	\brief Tensor operation handler
	**/
	class toh : public tensor_operation_handler<T> {
	private:
		tensor<T,Alloc,Perm> &m_t; //!< Underlying tensor

	public:
		//!	\name Construction and destruction
		//@{
		//!	Initializes the handler
		toh(tensor<T,Alloc,Perm> &t) : m_t(t) {}

		//!	Destroys the handler
		virtual ~toh() {}
		//@}

		//!	\name Overload of libtensor::tensor_operation_handler<T>
		//@{
		virtual element_t *req_dataptr(const permutation &p)
			throw(exception);

		virtual const element_t *req_const_dataptr(const permutation &p)
			throw(exception);

		virtual void ret_dataptr(const element_t *p) throw(exception);

		virtual const permutation &req_simplest_permutation()
			throw(exception);

		virtual size_t req_permutation_cost(const permutation &p)
			throw(exception);
		//@}
	};

private:
	dimensions m_dims; //!< Tensor %dimensions
	toh m_toh; //!< Tensor operation handler
	ptr_t m_data; //!< Pointer to data
	T *m_dataptr; //!< Pointer to checked out data
	permutation m_perm; //!< Current %permutation of the elements
	bool m_immutable; //!< Indicates whether the %tensor is immutable

public:
	//!	\name Construction and destruction
	//@{

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
			(by tensor<T,Alloc,Perm> reference)

		Creates a %tensor with the %dimensions of another %tensor.
		This constructor doesn't copy the data.

		\param t Another %tensor.
		\throw exception If an initialization error occurs.
	**/
	tensor(const tensor<T,Alloc,Perm> &t) throw(exception);

	/**	\brief Virtual destructor
	**/
	virtual ~tensor();

	//@}

	//!	\name Immutability
	//@{

	/**	\brief Checks if the %tensor is immutable

		Returns true if the %tensor is immutable, false otherwise.
	**/
	bool is_immutable() const;

	/**	\brief Sets the %tensor status as immutable.

		Sets the %tensor status as immutable. If the %tensor has already
		been set immutable, it stays immutable.
	**/
	void set_immutable();

	//@}

	//!	\name Implementation of tensor_i<T>
	//@{

	/**	\brief Returns the %dimensions of the %tensor

		Returns the %dimensions of the %tensor.
	**/
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
tensor<T,Alloc,Perm>::tensor(const dimensions &d) throw(exception) :
	m_dims(d), m_toh(*this), m_perm(m_dims.get_order()),
	m_data(Alloc::invalid_ptr), m_dataptr(NULL) {
#ifdef TENSOR_DEBUG
	if(m_dims.get_size() == 0) {
		throw_exc("tensor(const dimensions&)",
			"Zero tensor size is not allowed");
	}
#endif
	m_data = Alloc::allocate(m_dims.get_size());
	m_immutable = false;
}

template<typename T, typename Alloc, typename Perm>
tensor<T,Alloc,Perm>::tensor(const tensor_i<T> &t) throw(exception) :
	m_dims(t.get_dims()), m_toh(*this), m_perm(m_dims.get_order()),
	m_data(Alloc::invalid_ptr), m_dataptr(NULL) {
#ifdef TENSOR_DEBUG
	if(m_dims.get_size() == 0) {
		throw_exc("tensor(const tensor_i<T>&)",
			"Zero tensor size is not allowed");
	}
#endif
	m_data = Alloc::allocate(m_dims.get_size());
	m_immutable = false;
}

template<typename T, typename Alloc, typename Perm>
tensor<T,Alloc,Perm>::tensor(const tensor<T,Alloc,Perm> &t)
	throw(exception) : m_dims(t.m_dims), m_toh(*this),
	m_perm(m_dims.get_order()), m_data(Alloc::invalid_ptr),
	m_dataptr(NULL) {
#ifdef TENSOR_DEBUG
	if(m_dims.get_size() == 0) {
		throw_exc("tensor(const tensor<T,Alloc,Perm>&)",
			"Zero tensor size is not allowed");
	}
#endif
	m_data = Alloc::allocate(m_dims.get_size());
	m_immutable = false;
}

template<typename T, typename Alloc, typename Perm>
inline tensor<T,Alloc,Perm>::~tensor() {
	if(m_dataptr) {
		Alloc::unlock(m_data);
		m_dataptr = NULL;
	}
	Alloc::deallocate(m_data);
}

template<typename T, typename Alloc, typename Perm>
inline bool tensor<T,Alloc,Perm>::is_immutable() const {
	return m_immutable;
}

template<typename T, typename Alloc, typename Perm>
inline void tensor<T,Alloc,Perm>::set_immutable() {
	m_immutable = true;
}

template<typename T, typename Alloc, typename Perm>
inline const dimensions& tensor<T,Alloc,Perm>::get_dims() const {
	return m_dims;
}

template<typename T, typename Alloc, typename Perm>
inline tensor_operation_handler<T>&
tensor<T,Alloc,Perm>::get_tensor_operation_handler() {
	return m_toh;
}

template<typename T, typename Alloc, typename Perm>
inline void tensor<T,Alloc,Perm>::throw_exc(const char *method,
	const char *msg) throw(exception) {
	char s[1024];
	snprintf(s, 1024, "[libtensor::tensor<T,Alloc,Perm>::%s] %s.",
		method, msg);
	throw exception(s);
}

template<typename T, typename Alloc, typename Perm>
T *tensor<T,Alloc,Perm>::toh::req_dataptr(const permutation &p)
	throw(exception) {
	if(m_t.m_immutable) {
		m_t.throw_exc("toh::req_dataptr(const permutation&)",
			"Tensor is immutable, writing operations are "
				"prohibited");
	}

	if(m_t.m_dataptr) {
		m_t.throw_exc("toh::req_dataptr(const permutation&)",
			"Data pointer has already been checked out");
	}

	m_t.m_dataptr = Alloc::lock(m_t.m_data);

	// No permutation necessary
	if(p.equals(m_t.m_perm)) return m_t.m_dataptr;

	// Tensor elements need to be permuted

	typename Alloc::ptr_t data_dst =
		Alloc::allocate(m_t.get_dims().get_size());
	T *dataptr_dst = Alloc::lock(data_dst);

	// How elemens need to be permuted from current order
	permutation perm(m_t.m_perm, true);
	perm.permute(p);

	// Permuted dimensions
	dimensions dims(m_t.get_dims());
	dims.permute(m_t.m_perm);

	Perm::permute(m_t.m_dataptr, dataptr_dst, dims, perm);

	Alloc::unlock(m_t.m_data);
	Alloc::deallocate(m_t.m_data);
	m_t.m_data = data_dst;
	m_t.m_dataptr = dataptr_dst;
	m_t.m_perm.permute(perm);

	return m_t.m_dataptr;
}

template<typename T, typename Alloc, typename Perm>
const T *tensor<T,Alloc,Perm>::toh::req_const_dataptr(const permutation &p)
	throw(exception) {

	if(m_t.m_dataptr) {
		m_t.throw_exc("toh::req_dataptr(const permutation&)",
			"Data pointer has already been checked out");
	}

	m_t.m_dataptr = Alloc::lock(m_t.m_data);

	// Permute elements here if necessary

	return m_t.m_dataptr;
}

template<typename T, typename Alloc, typename Perm>
void tensor<T,Alloc,Perm>::toh::ret_dataptr(const element_t *p)
	throw(exception) {
	if(m_t.m_dataptr != p) {
		m_t.throw_exc("toh::ret_dataptr(const element_t*)",
			"Unrecognized data pointer");
	}
	Alloc::unlock(m_t.m_data);
	m_t.m_dataptr = NULL;
}

template<typename T, typename Alloc, typename Perm>
const permutation &tensor<T,Alloc,Perm>::toh::req_simplest_permutation()
	throw(exception) {
	return m_t.m_perm;
}

template<typename T, typename Alloc, typename Perm>
size_t tensor<T,Alloc,Perm>::toh::req_permutation_cost(const permutation &p)
	throw(exception) {
	m_t.throw_exc("toh::req_permutation_cost(const permutation&)",
		"Unhandled event");
}

} // namespace libtensor

#endif // __LIBTENSOR_TENSOR_H

