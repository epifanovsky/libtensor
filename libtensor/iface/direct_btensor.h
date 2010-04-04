#ifndef LIBTENSOR_DIRECT_BTENSOR_H
#define LIBTENSOR_DIRECT_BTENSOR_H

#include <libvmm/libvmm.h>
#include "../defs.h"
#include "../exception.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/direct_block_tensor.h"
#include "btensor_i.h"
#include "labeled_btensor.h"
#include "labeled_btensor_expr.h"
#include "expr/expr.h"
#include "expr/eval_i.h"
#include "expr/evalfunctor.h"

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
private:
	typedef struct {
		labeled_btensor_expr::expr_i<N, T> *m_pexpr;
		labeled_btensor_expr::eval_i<N, T> *m_peval;
		labeled_btensor_expr::evalfunctor_i<N, T> *m_pfunc;
	} ptrs_t;

private:
	letter_expr<N> m_label;
	ptrs_t m_ptrs;
	direct_block_tensor<N, T, typename Traits::allocator_t> m_bt;
	block_tensor_ctrl<N, T> m_ctrl;

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates the direct block %tensor using a %tensor
			expression
		\tparam Expr Tensor expression type.
	 **/
	template<typename Core>
	direct_btensor(const letter_expr<N> &label,
		const labeled_btensor_expr::expr<N, T, Core> &expr);

	/**	\brief Virtual destructor
	 **/
	virtual ~direct_btensor();

	//@}

	/**	\brief Attaches a label to this %tensor and returns it as a
			labeled %tensor
	 **/
	labeled_btensor<N, T, false> operator()(const letter_expr<N> &expr);


	//!	\name Implementation of block_tensor_i<N, T>
	//@{

	/**	\copydoc block_tensor_i<N, T>::get_bis()
	 **/
	virtual const block_index_space<N> &get_bis() const;

	//@}

protected:

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
	virtual tensor_i<N, T> &on_req_aux_block(const index<N> &idx)
		throw(exception);
	virtual void on_ret_aux_block(const index<N> &idx) throw(exception);
	virtual bool on_req_is_zero_block(const index<N> &idx) throw(exception);
	virtual void on_req_zero_block(const index<N> &idx) throw(exception);
	virtual void on_req_zero_all_blocks() throw(exception);
	//@}

private:
	template<typename Core>
	static ptrs_t mk_func(const letter_expr<N> &label,
		const labeled_btensor_expr::expr<N, T, Core> &expr);
};


template<size_t N, typename T, typename Traits> template<typename Core>
direct_btensor<N, T, Traits>::direct_btensor(const letter_expr<N> &label,
	const labeled_btensor_expr::expr<N, T, Core> &expr) :

	m_label(label), m_ptrs(mk_func(m_label, expr)),
	m_bt(m_ptrs.m_pfunc->get_bto()), m_ctrl(m_bt) {

}


template<size_t N, typename T, typename Traits>
direct_btensor<N, T, Traits>::~direct_btensor() {

	delete m_ptrs.m_pfunc;
	delete m_ptrs.m_peval;
	delete m_ptrs.m_pexpr;
}


template<size_t N, typename T, typename Traits> template<typename Core>
typename direct_btensor<N, T, Traits>::ptrs_t
direct_btensor<N, T, Traits>::mk_func(const letter_expr<N> &label,
	const labeled_btensor_expr::expr<N, T, Core> &expr) {

	typedef labeled_btensor_expr::expr<N, T, Core> expression_t;
	typedef typename expression_t::eval_container_t eval_container_t;

	const size_t narg_tensor = eval_container_t::template narg<
		labeled_btensor_expr::tensor_tag>::k_narg;
	const size_t narg_oper = eval_container_t::template narg<
		labeled_btensor_expr::oper_tag>::k_narg;

	expression_t *pexpr = new expression_t(expr);
	eval_container_t *peval = new eval_container_t(*pexpr, label);
	peval->prepare();

	ptrs_t ptrs;
	ptrs.m_pexpr = pexpr;
	ptrs.m_peval = peval;
	ptrs.m_pfunc = new labeled_btensor_expr::evalfunctor<N, T, Core,
		narg_tensor, narg_oper>(*pexpr, *peval);
	return ptrs;
}


template<size_t N, typename T, typename Traits>
labeled_btensor<N, T, false> direct_btensor<N, T, Traits>::operator()(
	const letter_expr<N> &expr) {

	return labeled_btensor<N, T, false>(*this, expr);
}


template<size_t N, typename T, typename Traits>
const block_index_space<N> &direct_btensor<N, T, Traits>::get_bis() const {

	return m_bt.get_bis();
}


template<size_t N, typename T, typename Traits>
const symmetry<N, T> &direct_btensor<N, T, Traits>::on_req_symmetry()
	throw(exception) {

	return m_ctrl.req_symmetry();
}


template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_req_sym_add_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	m_ctrl.req_sym_add_element(elem);
}


template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_req_sym_remove_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	m_ctrl.req_sym_remove_element(elem);
}


template<size_t N, typename T, typename Traits>
bool direct_btensor<N, T, Traits>::on_req_sym_contains_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	return m_ctrl.req_sym_contains_element(elem);
}


template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_req_sym_clear_elements()
	throw(exception) {

	m_ctrl.req_sym_clear_elements();
}


template<size_t N, typename T, typename Traits>
tensor_i<N, T> &direct_btensor<N, T, Traits>::on_req_block(const index<N> &idx)
	throw(exception) {

	return m_ctrl.req_block(idx);
}


template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_ret_block(const index<N> &idx)
	throw(exception) {

	m_ctrl.ret_block(idx);
}


template<size_t N, typename T, typename Traits>
tensor_i<N, T> &direct_btensor<N, T, Traits>::on_req_aux_block(
	const index<N> &idx) throw(exception) {

	return m_ctrl.req_aux_block(idx);
}


template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_ret_aux_block(const index<N> &idx)
	throw(exception) {

	m_ctrl.ret_aux_block(idx);
}


template<size_t N, typename T, typename Traits>
bool direct_btensor<N, T, Traits>::on_req_is_zero_block(const index<N> &idx)
	throw(exception) {

	return m_ctrl.req_is_zero_block(idx);
}


template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_req_zero_block(const index<N> &idx)
	throw(exception) {

	m_ctrl.req_zero_block(idx);
}


template<size_t N, typename T, typename Traits>
void direct_btensor<N, T, Traits>::on_req_zero_all_blocks() throw(exception) {

	m_ctrl.req_zero_all_blocks();
}


} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BTENSOR_H
