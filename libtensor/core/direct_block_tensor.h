#ifndef LIBTENSOR_DIRECT_BLOCK_TENSOR_H
#define LIBTENSOR_DIRECT_BLOCK_TENSOR_H

#include "../defs.h"
#include "../exception.h"
#include "block_map.h"
#include "direct_block_tensor_base.h"
#include "direct_block_tensor_operation.h"

namespace libtensor {

/**	\brief Direct block %tensor
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Alloc Memory allocator type.

	\ingroup libtensor_core
**/
template<size_t N, typename T, typename Alloc>
class direct_block_tensor : public direct_block_tensor_base<N, T> {
public:
	static const char *k_clazz; //!< Class name
public:
	typedef direct_block_tensor_base<N,T> base_t; //!< Base class type
	typedef T element_t; //!< Tensor element type
	typedef std::map<size_t,unsigned char> map_t;
	typedef std::pair<size_t,unsigned char> pair_t;
private:
	block_tensor<N,T,Alloc> m_bt; //!< Data block tensor
	map_t m_req_map; //!< Map of requests

public:
	//!	\name Construction and destruction
	//@{

	direct_block_tensor(direct_block_tensor_operation<N, T> &op) :
		base_t(op), m_bt(op.get_bis()) {

		block_tensor_ctrl<N,T> ctrl(m_bt);
		for ( typename symmetry<N,T>::iterator it=op.get_symmetry().begin();
				it!=op.get_symmetry().end(); it++ ) {
			ctrl.req_sym_add_element(op.get_symmetry().get_element(it));
		}
	}

	virtual ~direct_block_tensor() { }

	//@}

protected:
	//!	\name Implementation of libtensor::block_tensor_i<N, T>
	//@{

	virtual bool on_req_is_zero_block(const index<N> &idx)
		throw(exception);
	virtual tensor_i<N, T> &on_req_block(const index<N> &idx)
		throw(exception);
	virtual void on_ret_block(const index<N> &idx) throw(exception);
	virtual tensor_i<N, T> &on_req_aux_block(const index<N> &idx)
		throw(exception);
	virtual void on_ret_aux_block(const index<N> &idx) throw(exception);

	//@}

private:
	//! \brief Performs calculation of the given block
	void perform(const index<N>& idx) throw(exception);
};


template<size_t N, typename T, typename Alloc>
const char *direct_block_tensor<N, T, Alloc>::k_clazz =
	"direct_block_tensor<N, T, Alloc>";

template<size_t N, typename T, typename Alloc>
bool direct_block_tensor<N, T, Alloc>::on_req_is_zero_block(const index<N> &idx)
	throw(exception) {

	static const char *method = "on_req_is_zero_block(const index<N>&)";

	size_t absidx = base_t::get_bis().get_dims().abs_index(idx);

	typename map_t::iterator it=m_req_map.find(absidx);
	if ( it != m_req_map.end() )
		return false;

	base_t::m_op.perform(m_bt,idx);
	if ( ! block_tensor_ctrl<N,T>(m_bt).req_is_zero_block(idx) ) {
		m_req_map.insert(pair_t(absidx,1));
		return false;
	}
	else
		return true;

}


template<size_t N, typename T, typename Alloc>
tensor_i<N, T> &direct_block_tensor<N, T, Alloc>::on_req_block(
	const index<N> &idx) throw(exception) {

	static const char *method = "on_req_block(const index<N>&)";

	size_t absidx = base_t::get_bis().get_dims().abs_index(idx);

	typename map_t::iterator it=m_req_map.find(absidx);
	if ( it != m_req_map.end() ) {
		it->second++;
	}
	else {
		base_t::m_op.perform(m_bt,idx);
		if ( block_tensor_ctrl<N,T>(m_bt).req_is_zero_block(idx) )
			throw immut_violation(g_ns,k_clazz,method,
					__FILE__,__LINE__,"Requesting zero block.");

		m_req_map.insert(pair_t(absidx,1));
	}
	return block_tensor_ctrl<N,T>(m_bt).req_block(idx);

}


template<size_t N, typename T, typename Alloc>
void direct_block_tensor<N, T, Alloc>::on_ret_block(const index<N> &idx)
	throw(exception) {

	static const char *method = "on_ret_block(const index<N>&)";

	size_t absidx = base_t::get_bis().get_dims().abs_index(idx);


	typename map_t::iterator it=m_req_map.find(absidx);
	if ( it != m_req_map.end() ) {
		if ( it->second == 1 ) {
			block_tensor_ctrl<N,T>(m_bt).req_zero_block(idx);
			m_req_map.erase(it);
		}
		else {
			it->second--;
		}
	}
}


template<size_t N, typename T, typename Alloc>
tensor_i<N, T> &direct_block_tensor<N, T, Alloc>::on_req_aux_block(
	const index<N> &idx) throw(exception) {

	static const char *method = "on_req_aux_block(const index<N>&)";

	throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
		"invalid_req");
}


template<size_t N, typename T, typename Alloc>
void direct_block_tensor<N, T, Alloc>::on_ret_aux_block(const index<N> &idx)
	throw(exception) {

	static const char *method = "on_ret_aux_block(const index<N>&)";

	throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
		"invalid_req");
}


} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BLOCK_TENSOR_H
