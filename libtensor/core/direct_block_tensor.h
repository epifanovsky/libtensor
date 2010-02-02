#ifndef LIBTENSOR_DIRECT_BLOCK_TENSOR_H
#define LIBTENSOR_DIRECT_BLOCK_TENSOR_H

#include "../defs.h"
#include "../exception.h"
#include "block_tensor_i.h"
#include "direct_block_tensor_operation.h"

namespace libtensor {

/**	\brief Direct block %tensor
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Alloc Memory allocator type.

	\ingroup libtensor_core
**/
template<size_t N, typename T, typename Alloc>
class direct_block_tensor : public block_tensor_i<N, T> {
public:
	static const char *k_clazz; //!< Class name

public:
	typedef T element_t; //!< Tensor element type

private:
	//!	Underlying block tensor operation
	direct_block_tensor_operation<N, T> &m_op;

public:
	//!	\name Construction and destruction
	//@{

	direct_block_tensor(direct_block_tensor_operation<N, T> &op) :
		m_op(op) { }
	virtual ~direct_block_tensor() { }

	//@}

	//!	\name Implementation of libtensor::block_tensor_i<N, T>
	//@{

	virtual const block_index_space<N> &get_bis() const;

	//@}

protected:
	//!	\name Implementation of libtensor::block_tensor_i<N, T>
	//@{

	virtual const symmetry<N, T> &on_req_symmetry() throw(exception);
	virtual void on_req_sym_add_element(
		const symmetry_element_i<N, T> &elem) throw(exception);
	virtual void on_req_sym_remove_element(
		const symmetry_element_i<N, T> &elem) throw(exception);
	virtual bool on_req_sym_contains_element(
		const symmetry_element_i<N, T> &elem) throw(exception);
	virtual void on_req_sym_clear_elements() throw(exception);
	virtual tensor_i<N, T> &on_req_block(const index<N> &idx)
		throw(exception);
	virtual void on_ret_block(const index<N> &idx) throw(exception);
	virtual bool on_req_is_zero_block(const index<N> &idx)
		throw(exception);
	virtual void on_req_zero_block(const index<N> &idx)
		throw(exception);
	virtual void on_req_zero_all_blocks() throw(exception);

};


template<size_t N, typename T, typename Alloc>
const char *direct_block_tensor<N, T, Alloc>::k_clazz =
	"direct_block_tensor<N, T, Alloc>";


template<size_t N, typename T, typename Alloc>
const block_index_space<N> &direct_block_tensor<N, T, Alloc>::get_bis() const {

	return m_op.get_bis();
}


template<size_t N, typename T, typename Alloc>
const symmetry<N, T> &direct_block_tensor<N, T, Alloc>::on_req_symmetry()
	throw(exception) {

	return m_op.get_symmetry();
}


template<size_t N, typename T, typename Alloc>
void direct_block_tensor<N, T, Alloc>::on_req_sym_add_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	static const char *method =
		"on_req_sym_add_element(const symmetry_element_i<N, T>&)";

	throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
		"Immutable object cannot be modified.");
}


template<size_t N, typename T, typename Alloc>
void direct_block_tensor<N, T, Alloc>::on_req_sym_remove_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	static const char *method =
		"on_req_sym_remove_element(const symmetry_element_i<N, T>&)";

	throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
		"Immutable object cannot be modified.");
}


template<size_t N, typename T, typename Alloc>
bool direct_block_tensor<N, T, Alloc>::on_req_sym_contains_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	static const char *method =
		"on_req_sym_contains_element(const symmetry_element_i<N, T>&)";

	throw_exc(k_clazz, method, "NIY");
}


template<size_t N, typename T, typename Alloc>
void direct_block_tensor<N, T, Alloc>::on_req_sym_clear_elements()
	throw(exception) {

	static const char *method = "on_req_sym_clear_elements()";

	throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
		"Immutable object cannot be modified.");
}


template<size_t N, typename T, typename Alloc>
tensor_i<N, T> &direct_block_tensor<N, T, Alloc>::on_req_block(
	const index<N> &idx) throw(exception) {

	static const char *method = "on_req_block(const index<N>&)";

	throw_exc(k_clazz, method, "NIY");
}


template<size_t N, typename T, typename Alloc>
void direct_block_tensor<N, T, Alloc>::on_ret_block(const index<N> &idx)
	throw(exception) {

	static const char *method = "on_ret_block(const index<N>&)";

	throw_exc(k_clazz, method, "NIY");
}


template<size_t N, typename T, typename Alloc>
bool direct_block_tensor<N, T, Alloc>::on_req_is_zero_block(
	const index<N> &idx) throw(exception) {
	
	static const char *method = "on_req_is_zero_block(const index<N>&)";

	throw_exc(k_clazz, method, "NIY");
}


template<size_t N, typename T, typename Alloc>
void direct_block_tensor<N, T, Alloc>::on_req_zero_block(const index<N> &idx)
	throw(exception) {

	static const char *method = "on_req_zero_block(const index<N>&)";

	throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
		"Immutable object cannot be modified.");
}


template<size_t N, typename T, typename Alloc>
void direct_block_tensor<N, T, Alloc>::on_req_zero_all_blocks()
	throw(exception) {

	static const char *method = "on_req_zero_all_blocks()";

	throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
		"Immutable object cannot be modified.");
}


} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BLOCK_TENSOR_H
