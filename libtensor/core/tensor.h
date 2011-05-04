#ifndef LIBTENSOR_TENSOR_H
#define LIBTENSOR_TENSOR_H

#include <sstream>
#include <vector>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "immutable.h"
#include "permutation.h"
#include "tensor_i.h"

namespace libtensor {


/**	\brief Simple %tensor, which stores all its elements in memory

	\param N Tensor order.
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

	\ingroup libtensor_core
**/
template<size_t N, typename T, typename Alloc>
class tensor :
	public tensor_i<N, T>,
	public immutable,
	public timings< tensor<N, T, Alloc> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef T element_t; //!< Tensor element type
	typedef typename Alloc::ptr_t ptr_t; //!< Memory pointer type
	typedef typename tensor_i<N, T>::handle_t
		handle_t; //!< Session handle type

private:
	dimensions<N> m_dims; //!< Tensor %dimensions
	ptr_t m_data; //!< Pointer to data
	T *m_dataptr; //!< Pointer to checked out data
	size_t m_ptrcount; //!< Number of read-only data pointers given out
	std::vector<char> m_sessions; //!< Sessions
	std::vector<size_t> m_session_ptrcount; //!< Per-session data pointer counts

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates an empty %tensor
		\param dims Non-zero %tensor dimensions.
	 **/
	tensor(const dimensions<N> &dims);

	/**	\brief Creates an empty %tensor with the same %dimensions
			(data are not copied)
		\param t Another %tensor (tensor_i<N, T>).
	 **/
	tensor(const tensor_i<N, T> &t);

	/**	\brief Creates an empty %tensor with the same %dimensions
			(data are not copied)
		\param t Another %tensor (tensor<N, T, Alloc).
	 **/
	tensor(const tensor<N, T, Alloc> &t);

	/**	\brief Virtual destructor
	 **/
	virtual ~tensor();

	//@}


	//!	\name Implementation of libtensor::tensor_i<N, T>
	//@{

	/**	\brief Returns the %dimensions of the %tensor

		Returns the %dimensions of the %tensor.
	**/
	virtual const dimensions<N> &get_dims() const;

	//@}

protected:
	//!	\name Implementation of libtensor::tensor_i<N, T>
	//@{

	virtual handle_t on_req_open_session();
	virtual void on_req_close_session(const handle_t &h);
	virtual void on_req_prefetch(const handle_t &h);
	virtual T *on_req_dataptr(const handle_t &h);
	virtual void on_ret_dataptr(const handle_t &h, const T *p);
	virtual const T *on_req_const_dataptr(const handle_t &h);
	virtual void on_ret_const_dataptr(const handle_t &h, const T *p);

	//@}

	//!	\name Implementation of libtensor::immutable
	//@{
	virtual void on_set_immutable() { }
	//@}

	//!	\name Service functions
	//@{

	/**	\brief Verifies that the session identified by a handler
			exists and is open
		\param h Session handler.
		\throw bad_parameter If the handler is invalid or the session
			does not exist.
	 **/
	void verify_session(size_t h) throw(bad_parameter);

	/**	\brief Unlocks the physical pointer, returns it to the allocator
	 **/
	void unlock_dataptr();

	//@}

};


template<size_t N, typename T, typename Alloc>
const char *tensor<N, T, Alloc>::k_clazz = "tensor<N, T, Alloc>";


template<size_t N, typename T, typename Alloc>
tensor<N, T, Alloc>::tensor(const dimensions<N> &dims) :
	m_dims(dims), m_data(Alloc::invalid_ptr), m_dataptr(0), m_ptrcount(0),
	m_sessions(8, 0), m_session_ptrcount(8, 0) {

#ifdef LIBTENSOR_DEBUG
	if(m_dims.get_size() == 0) {
		throw_exc("tensor<N,T,Alloc>", "tensor(const dimensions&)",
			"Zero tensor size is not allowed");
	}
#endif // LIBTENSOR_DEBUG

	m_data = Alloc::allocate(m_dims.get_size());
}


template<size_t N, typename T, typename Alloc>
tensor<N, T, Alloc>::tensor(const tensor_i<N, T> &t) :

	m_dims(t.get_dims()), m_data(Alloc::invalid_ptr), m_dataptr(0),
	m_ptrcount(0), m_sessions(8, 0), m_session_ptrcount(8, 0) {

	m_data = Alloc::allocate(m_dims.get_size());
}


template<size_t N, typename T, typename Alloc>
tensor<N, T, Alloc>::tensor(const tensor<N, T, Alloc> &t) :

	m_dims(t.m_dims), m_data(Alloc::invalid_ptr), m_dataptr(0),
	m_ptrcount(0), m_sessions(8, 0), m_session_ptrcount(8, 0) {

	m_data = Alloc::allocate(m_dims.get_size());
}


template<size_t N, typename T, typename Alloc>
tensor<N, T, Alloc>::~tensor() {

	if(m_dataptr != 0) {
		Alloc::unlock(m_data);
		m_dataptr = 0;
	}
	Alloc::deallocate(m_data);
}


template<size_t N, typename T, typename Alloc>
const dimensions<N> &tensor<N, T, Alloc>::get_dims() const {

	return m_dims;
}


template<size_t N, typename T, typename Alloc>
typename tensor<N, T, Alloc>::handle_t
tensor<N, T, Alloc>::on_req_open_session() {

	size_t sz = m_sessions.size();

	for(register size_t i = 0; i < sz; i++) {
		if(m_sessions[i] == 0) {
			m_sessions[i] = 1;
			m_session_ptrcount[i] = 0;
			return i;
		}
	}

	m_sessions.resize(2 * sz, 0);
	m_session_ptrcount.resize(2 * sz, 0);
	m_sessions[sz] = 1;
	m_session_ptrcount[sz] = 0;
	return sz;
}


template<size_t N, typename T, typename Alloc>
void tensor<N, T, Alloc>::on_req_close_session(const handle_t &h) {

	verify_session(h);

	m_sessions[h] = 0;
	if(m_dataptr != 0) {
		m_ptrcount -= m_session_ptrcount[h];
		m_session_ptrcount[h] = 0;
		if(m_ptrcount == 0) unlock_dataptr();
	}
}


template<size_t N, typename T, typename Alloc>
void tensor<N, T, Alloc>::on_req_prefetch(const handle_t &h) {

	verify_session(h);

	if(m_dataptr == 0) Alloc::prefetch(m_data);
}


template<size_t N, typename T, typename Alloc>
T *tensor<N, T, Alloc>::on_req_dataptr(const handle_t &h) {

	static const char *method = "on_req_dataptr(const handle_t&)";

	verify_session(h);

	if(is_immutable()) {
		throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
			"");
	}

	if(m_dataptr != 0) {
		throw_exc(k_clazz, method,
			"Data pointer is already checked out for rw");
	}

	timings< tensor<N, T, Alloc> >::start_timer("lock");
	m_dataptr = Alloc::lock(m_data);
	timings< tensor<N, T, Alloc> >::stop_timer("lock");
	return m_dataptr;
}


template<size_t N, typename T, typename Alloc>
void tensor<N, T, Alloc>::on_ret_dataptr(const handle_t &h, const T *p) {

	static const char *method = "on_ret_dataptr(const handle_t&, const T*)";

	verify_session(h);

	if(m_dataptr == 0 || m_dataptr != p) {
		std::ostringstream ss;
		ss << "p[m_dataptr=" << m_dataptr << ",p=" << p << ",m_ptrcount=" << m_ptrcount << "]";
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			ss.str().c_str());
	}

	if(m_session_ptrcount[h] > 0) {
		m_session_ptrcount[h]--;
		m_ptrcount--;
	}
	if(m_ptrcount == 0) unlock_dataptr();
}


template<size_t N, typename T, typename Alloc>
const T *tensor<N,T,Alloc>::on_req_const_dataptr(const handle_t &h) {

	static const char *method = "on_req_const_dataptr(const handle_t&)";

	verify_session(h);

	if(m_dataptr != 0) {

		if(m_ptrcount == 0) {
			throw_exc(k_clazz, method,
				"Data pointer is already checked out for rw");
		}

		m_session_ptrcount[h]++;
		m_ptrcount++;
		return m_dataptr;
	}

	timings< tensor<N, T, Alloc> >::start_timer("lock");
	m_dataptr = Alloc::lock(m_data);
	timings< tensor<N, T, Alloc> >::stop_timer("lock");
	m_session_ptrcount[h] = 1;
	m_ptrcount = 1;
	return m_dataptr;
}


template<size_t N, typename T, typename Alloc>
void tensor<N, T, Alloc>::on_ret_const_dataptr(const handle_t &h, const T *p) {

	tensor<N, T, Alloc>::on_ret_dataptr(h, p);
}


template<size_t N, typename T, typename Alloc>
inline void tensor<N, T, Alloc>::verify_session(size_t h) throw(bad_parameter) {

	static const char *method = "verify_session(size_t)";

	if(h >= m_sessions.size() || m_sessions[h] == 0) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"h");
	}
}


template<size_t N, typename T, typename Alloc>
inline void tensor<N, T, Alloc>::unlock_dataptr() {

	Alloc::unlock(m_data);
	m_dataptr = 0;
}


} // namespace libtensor

#endif // LIBTENSOR_TENSOR_H
