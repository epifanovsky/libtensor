#ifndef LIBTENSOR_DIRECT_BTENSOR_H
#define LIBTENSOR_DIRECT_BTENSOR_H

#include <libvmm.h>
#include "defs.h"
#include "exception.h"
#include "core/direct_block_tensor.h"
#include "btensor_i.h"
#include "labeled_btensor.h"
#include "labeled_btensor_expr.h"

namespace libtensor {

template<typename T>
struct direct_btensor_traits {
	typedef T element_t;
	typedef libvmm::std_allocator<T> allocator_t;
};

/**	\brief User-friendly direct block %tensor

	\ingroup libtensor_iface
 **/
template<size_t N, typename T = double,
	typename Traits = direct_btensor_traits<T> >
class direct_btensor : public btensor_i<N, T> {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates the direct block %tensor using a %tensor
			expression
		\tparam LetterExpr Letter expression type for the label.
		\tparam Expr Tensor expression type.
	 **/
	template<typename LetterExpr, typename Expr>
	direct_btensor(const letter_expr<N, LetterExpr> &label,
		const labeled_btensor_expr<N, T, Expr> &expr)
		throw(exception);

	/**	\brief Virtual destructor
	 **/
	virtual ~direct_btensor();

	//@}

	/**	\brief Attaches a label to this %tensor and returns it as a
			labeled %tensor
	 **/
	template<typename ExprT>
	labeled_btensor<N, T, false, letter_expr<N, ExprT> > operator()(
		const letter_expr<N, ExprT> expr);

	//!	\name Implementation of tensor_i<N, T>
	//@{

	const dimensions<N> &get_dims() const;

	//@}

	//!	\name Implementation of block_tensor_i<N, T>
	//!{

	const block_index_space<N> &get_bis() const;

	//!}

protected:
	//!	\name Implementation of tensor_i<N, T>
	//@{
	virtual void on_req_prefetch() throw(exception);
	virtual T *on_req_dataptr() throw(exception);
	virtual const T *on_req_const_dataptr() throw(exception);
	virtual void on_ret_dataptr(const T *ptr) throw(exception);
	//@}

	//!	\name Implementation of block_tensor_i<N, T>
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
	virtual bool on_req_is_zero_block(const index<N> &idx) throw(exception);
	virtual void on_req_zero_block(const index<N> &idx) throw(exception);
	virtual void on_req_zero_all_blocks() throw(exception);
	//@}
};

template<size_t N, typename T, typename Traits>
template<typename LetterExpr, typename Expr>
direct_btensor<N, T, Traits>::direct_btensor(
	const letter_expr<N, LetterExpr> &label,
	const labeled_btensor_expr<N, T, Expr> &expr) throw(exception) {
}

template<size_t N, typename T, typename Traits>
direct_btensor<N, T, Traits>::~direct_btensor() {
}

template<size_t N, typename T, typename Traits> template<typename ExprT>
labeled_btensor<N, T, false, letter_expr<N, ExprT> >
direct_btensor<N, T, Traits>::operator()(const letter_expr<N, ExprT> expr) {
	return labeled_btensor<N, T, false, letter_expr<N, ExprT> >(
		*this, expr);
}

template<size_t N, typename T, typename Traits>
const dimensions<N> &direct_btensor<N, T, Traits>::get_dims() const {
	throw_exc("direct_btensor<N, T, Traits>", "get_dims()",
		"Not implemented");
}

template<size_t N, typename T, typename Traits>
const block_index_space<N> &direct_btensor<N, T, Traits>::get_bis() const {
	throw_exc("direct_btensor<N, T, Traits>", "get_bis()",
		"Not implemented");
}

template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_req_prefetch() throw(exception) {
}

template<size_t N, typename T, typename Traits>
T *direct_btensor<N, T, Traits>::on_req_dataptr() throw(exception) {
	throw_exc("direct_btensor<N, T, Traits>", "on_req_dataptr()",
		"Unhandled event");
	return NULL;
}

template<size_t N, typename T, typename Traits>
const T *direct_btensor<N, T, Traits>::on_req_const_dataptr() throw(exception) {
	throw_exc("direct_btensor<N, T, Traits>", "on_req_const_dataptr()",
		"Unhandled event");
	return NULL;
}

template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_ret_dataptr(const T *ptr) throw(exception) {
	throw_exc("direct_btensor<N, T, Traits>", "on_ret_dataptr(const T*)",
		"Unhandled event");
}

template<size_t N, typename T, typename Traits>
const symmetry<N, T> &direct_btensor<N, T, Traits>::on_req_symmetry()
	throw(exception) {

	throw_exc("direct_btensor<N, T, Traits>", "on_req_symmetry()", "NIY");
}

template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_req_sym_add_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	throw_exc("direct_btensor<N, T, Traits>", "on_req_sym_add_element()", "NIY");
}

template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_req_sym_remove_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	throw_exc("direct_btensor<N, T, Traits>", "on_req_sym_remove_element()", "NIY");
}

template<size_t N, typename T, typename Traits>
bool direct_btensor<N, T, Traits>::on_req_sym_contains_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	throw_exc("direct_btensor<N, T, Traits>", "on_req_sym_contains_element()", "NIY");
}

template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_req_sym_clear_elements()
	throw(exception) {
	throw_exc("direct_btensor<N, T, Traits>", "on_req_sym_clear_elements()", "NIY");
}

template<size_t N, typename T, typename Traits>
tensor_i<N, T> &direct_btensor<N, T, Traits>::on_req_block(const index<N> &idx)
	throw(exception) {
	throw_exc("direct_btensor<N, T, Traits>",
		"on_req_block(const index<N>&)", "Unhandled event");
}

template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_ret_block(const index<N> &idx)
	throw(exception) {
	throw_exc("direct_btensor<N, T, Traits>", "on_ret_block()", "NIY");
}

template<size_t N, typename T, typename Traits>
bool direct_btensor<N, T, Traits>::on_req_is_zero_block(const index<N> &idx)
	throw(exception) {
	throw_exc("direct_btensor<N, T, Traits>", "on_req_is_zero_block()", "NIY");
}

template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_req_zero_block(const index<N> &idx)
	throw(exception) {
	throw_exc("direct_btensor<N, T, Traits>", "on_req_zero_block()", "NIY");
}

template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_req_zero_all_blocks() throw(exception) {
	throw_exc("direct_btensor<N, T, Traits>", "on_req_zero_all_blocks()", "NIY");
}

} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BTENSOR_H
