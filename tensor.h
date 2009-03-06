#ifndef LIBTENSOR_TENSOR_H
#define LIBTENSOR_TENSOR_H

#include <libvmm.h>

#include "defs.h"
#include "exception.h"
#include "immutable.h"
#include "permutation.h"
#include "tensor_i.h"

namespace libtensor {

/**	\brief Simple %tensor, which stores all its elements in memory

	\param T Tensor element type.
	\param Alloc Memory allocator.

	This class is a container for %tensor elements located in memory.
	It allocates and deallocates memory for the elements, provides facility
	for performing %tensor operations. The %tensor class doesn't perform
	any mathematical operations on its elements, but rather establishes
	a protocol, with which an operation of any complexity can be implemented
	as a separate class. 

	<b>Element type</b>

	Tensor elements can be of any POD type or any class or structure
	that implements a default constructor and the assignment operator.

	Example:
	\code
	struct tensor_element_t {
		// ...
		tensor_element_t();
		tensor_element_t &operator=(const tensor_element_t&);
	};
	\endcode

	The type of the elements is a template parameter:
	\code
	typedef libvmm::std_allocator<tensor_element_t> tensor_element_alloc;
	tensor<tensor_element_t, tensor_element_alloc, permutator> t(...);
	\endcode

	<b>Tensor operations</b>

	The %tensor class does not perform any mathematical operations, nor
	does it allow direct access to its elements. Tensor operations are
	implemented as separate classes by extending
	libtensor::tensor_operation.

	<b>Allocator</b>

	Tensor uses a memory allocator to obtain storage. For more information,
	see the libvmm package.

	<b>Storage format</b>

	Tensor elements are stored in memory one after another in the order
	of the running %index. The first %index element is the slowest
	running, the last one is the fastest running.

	<b>Permutations and %permutator</b>

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

	The default %permutator implementation is libtensor::permutator.

	<b>Immutability</b>

	A %tensor can be set %immutable via set_immutable(), after which only
	reading operations are allowed on the %tensor. Operations that attempt
	to modify the elements are prohibited and will cause an %exception.
	Once the %tensor status is set to %immutable, it cannot be changed back.
	To perform a check whether the %tensor is mutable or %immutable,
	is_immutable() can be used. Immutability is provided by
	libtensor::immutable.

	Example:
	\code
	tensor<double> t(...);

	// Any operations or permutations are allowed with the tensor t

	t.set_immutable();

	// Only reading operations are allowed with the tensor t

	bool b = t.is_immutable(); // true
	\endcode

	<b>Exceptions</b>

	Exceptions libtensor::exception are thrown if a requested operation
	fails for any reason. If an %exception is thrown, the state of
	the %tensor object is undefined.

	\ingroup libtensor
**/
template<typename T, typename Alloc>
class tensor : public tensor_i<T>, public immutable {
public:
	typedef T element_t; //!< Tensor element type
	typedef typename Alloc::ptr_t ptr_t; //!< Memory pointer type

private:
	dimensions m_dims; //!< Tensor %dimensions
	ptr_t m_data; //!< Pointer to data
	T *m_dataptr; //!< Pointer to checked out data
	size_t m_ptrcount; //!< Number of read-only data pointers given out

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
			(by tensor<T,Alloc> reference)

		Creates a %tensor with the %dimensions of another %tensor.
		This constructor doesn't copy the data.

		\param t Another %tensor.
		\throw exception If an initialization error occurs.
	**/
	tensor(const tensor<T,Alloc> &t) throw(exception);

	/**	\brief Virtual destructor
	**/
	virtual ~tensor();

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

	virtual void on_req_prefetch() throw(exception);
	virtual T *on_req_dataptr() throw(exception);
	virtual const T *on_req_const_dataptr() throw(exception);
	virtual void on_ret_dataptr(const T *p) throw(exception);

	//@}

private:
	void throw_exc(const char *method, const char *msg) throw(exception);
};

template<typename T, typename Alloc>
tensor<T,Alloc>::tensor(const dimensions &d) throw(exception) :
	m_dims(d), m_data(Alloc::invalid_ptr), m_dataptr(NULL), m_ptrcount(0) {
#ifdef LIBTENSOR_DEBUG
	if(m_dims.get_size() == 0) {
		throw_exc("tensor(const dimensions&)",
			"Zero tensor size is not allowed");
	}
#endif
	m_data = Alloc::allocate(m_dims.get_size());
}

template<typename T, typename Alloc>
tensor<T,Alloc>::tensor(const tensor_i<T> &t) throw(exception) :
	m_dims(t.get_dims()), m_data(Alloc::invalid_ptr), m_dataptr(NULL),
	m_ptrcount(0) {
#ifdef LIBTENSOR_DEBUG
	if(m_dims.get_size() == 0) {
		throw_exc("tensor(const tensor_i<T>&)",
			"Zero tensor size is not allowed");
	}
#endif
	m_data = Alloc::allocate(m_dims.get_size());
}

template<typename T, typename Alloc>
tensor<T,Alloc>::tensor(const tensor<T,Alloc> &t)
	throw(exception) : m_dims(t.m_dims), m_data(Alloc::invalid_ptr),
	m_dataptr(NULL), m_ptrcount(0) {
#ifdef LIBTENSOR_DEBUG
	if(m_dims.get_size() == 0) {
		throw_exc("tensor(const tensor<T,Alloc>&)",
			"Zero tensor size is not allowed");
	}
#endif
	m_data = Alloc::allocate(m_dims.get_size());
}

template<typename T, typename Alloc>
inline tensor<T,Alloc>::~tensor() {
	if(m_dataptr) {
		Alloc::unlock(m_data);
		m_dataptr = NULL;
	}
	Alloc::deallocate(m_data);
}

template<typename T, typename Alloc>
inline const dimensions& tensor<T,Alloc>::get_dims() const {
	return m_dims;
}

template<typename T, typename Alloc>
void tensor<T,Alloc>::on_req_prefetch() throw(exception) {
	Alloc::prefetch(m_data);
}

template<typename T, typename Alloc>
T *tensor<T,Alloc>::on_req_dataptr() throw(exception) {
	if(is_immutable()) {
		throw_exc("on_req_dataptr()", "Tensor is immutable, writing "
			"operations are prohibited");
	}

	if(m_dataptr) {
		throw_exc("on_req_dataptr()",
			"Data pointer is already checked out for rw");
	}

	m_dataptr = Alloc::lock(m_data);
	return m_dataptr;
}

template<typename T, typename Alloc>
const T *tensor<T,Alloc>::on_req_const_dataptr() throw(exception) {
	if(m_dataptr) {
		if(m_ptrcount) {
			m_ptrcount++;
			return m_dataptr;
		}
		throw_exc("on_req_const_dataptr()",
			"Data pointer is already checked out for rw");
	}

	m_dataptr = Alloc::lock(m_data);
	m_ptrcount = 1;
	return m_dataptr;
}

template<typename T, typename Alloc>
void tensor<T,Alloc>::on_ret_dataptr(const element_t *p) throw(exception) {
	if(m_dataptr != p) {
		throw_exc("on_ret_dataptr(const element_t*)",
			"Unrecognized data pointer");
	}
	if(m_ptrcount > 0) m_ptrcount--;
	if(m_ptrcount == 0) {
		Alloc::unlock(m_data);
		m_dataptr = NULL;
	}
}

template<typename T, typename Alloc>
inline void tensor<T,Alloc>::throw_exc(const char *method,
	const char *msg) throw(exception) {
	char s[1024];
	snprintf(s, 1024, "[libtensor::tensor<T,Alloc>::%s] %s.",
		method, msg);
	throw exception(s);
}

} // namespace libtensor

#endif // LIBTENSOR_TENSOR_H

